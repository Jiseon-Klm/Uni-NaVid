#!/usr/bin/env python3
# coding: utf-8
"""
ROS2 기반 Uni-NaVid 온라인 평가 스크립트 (Non-blocking v4 - 논문 완전 준수)
- 논문의 non-blocking navigation 완전 구현
- 각 액션 완료 후 명시적으로 새 이미지 캡처 및 전송
- 여러 명령 도착 시 최신 명령 우선 실행
- GPU 성능에 관계없이 동작하도록 비동기 처리
- ROS2로부터 이미지를 subscribe (10Hz, Gemini 336L 카메라)
- 이미지 히스토리 누적 (논문의 online token merging)
- 새 프레임만 전달하여 중복 추가 방지 (프레임 카운터 기반)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool, Empty
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import time
import argparse
import threading
from queue import Queue, Empty as QueueEmpty
from collections import deque
from enum import Enum

from uninavid.mm_utils import get_model_name_from_path
from uninavid.model.builder import load_pretrained_model
from uninavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from uninavid.conversation import conv_templates, SeparatorStyle
from uninavid.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria


class AgentState(Enum):
    """에이전트 상태"""
    IDLE = "idle"                    # 대기 중
    WAITING_FOR_IMAGE = "waiting_for_image"  # 새 이미지 대기 중
    INFERRING = "inferring"          # 추론 중
    EXECUTING_ACTIONS = "executing_actions"  # 액션 실행 중
    WAITING_ACTION_IMAGE = "waiting_action_image"  # 액션 완료 후 이미지 대기 중


class UniNaVid_Agent():
    """Uni-NaVid 에이전트 클래스"""
    def __init__(self, model_path):
        print("Initialize UniNaVid")
        
        self.conv_mode = "vicuna_v1"
        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, None, get_model_name_from_path(model_path)
        )
        
        assert self.image_processor is not None
        print("Initialization Complete")
        
        self.promt_template = "Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations and an image of the current observation <image>. Your assigned task is: '{}'. Analyze this series of images to determine your next four actions. The predicted action should be one of the following: forward, left, right, or stop."
        self.rgb_list = []
        self.count_id = 0
        self.reset()

    def process_images(self, rgb_list):
        """이미지 전처리 (new_frames는 act_with_history에서 이미 설정됨)"""
        batch_image = np.asarray(rgb_list)
        video = self.image_processor.preprocess(batch_image, return_tensors='pt')['pixel_values'].half().cuda()
        return [video]

    def predict_inference(self, prompt):
        question = prompt.replace(DEFAULT_IMAGE_TOKEN, '').replace('\n', '')
        qs = prompt

        VIDEO_START_SPECIAL_TOKEN = "<video_special>"
        VIDEO_END_SPECIAL_TOKEN = "</video_special>"
        IMAGE_START_TOKEN = "<image_special>"
        IMAGE_END_TOKEN = "</image_special>"
        NAVIGATION_SPECIAL_TOKEN = "[Navigation]"
        IAMGE_SEPARATOR = "<image_sep>"
        image_start_special_token = self.tokenizer(IMAGE_START_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_end_special_token = self.tokenizer(IMAGE_END_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_start_special_token = self.tokenizer(VIDEO_START_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_end_special_token = self.tokenizer(VIDEO_END_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        navigation_special_token = self.tokenizer(NAVIGATION_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_seperator = self.tokenizer(IAMGE_SEPARATOR, return_tensors="pt").input_ids[0][1:].cuda()

        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs.replace('<image>', '')
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs.replace('<image>', '')

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        token_prompt = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()
        indices_to_replace = torch.where(token_prompt == -200)[0]
        new_list = []
        while indices_to_replace.numel() > 0:
            idx = indices_to_replace[0]
            new_list.append(token_prompt[:idx])
            new_list.append(video_start_special_token)
            new_list.append(image_seperator)
            new_list.append(token_prompt[idx:idx + 1])
            new_list.append(video_end_special_token)
            new_list.append(image_start_special_token)
            new_list.append(image_end_special_token)
            new_list.append(navigation_special_token)
            token_prompt = token_prompt[idx + 1:]
            indices_to_replace = torch.where(token_prompt == -200)[0]
        if token_prompt.numel() > 0:
            new_list.append(token_prompt)
        input_ids = torch.cat(new_list, dim=0).unsqueeze(0)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        imgs = self.process_images(self.rgb_list)
        # rgb_list는 유지 (다음 추론에서 계속 누적)

        cur_prompt = question
        with torch.inference_mode():
            self.model.update_prompt([[cur_prompt]])
            output_ids = self.model.generate(
                input_ids,
                images=imgs,
                do_sample=True,
                temperature=0.5,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        return outputs

    def reset(self, task_type='vln'):
        self.transformation_list = []
        self.rgb_list = []
        self.last_action = None
        self.count_id += 1
        self.count_stop = 0
        self.pending_action_list = []
        self.task_type = task_type
        self.first_forward = False
        self.executed_steps = 0
        self.model.config.run_type = "eval"
        self.model.get_model().initialize_online_inference_nav_feat_cache()
        self.model.get_model().new_frames = 0

    def act_with_history(self, data):
        """새 프레임들을 받아 히스토리에 추가하고 추론 (중복 방지)"""
        new_frames = data["observations"]  # 새 프레임 리스트
        
        # 새 프레임만 추가 (중복 방지)
        for rgb in new_frames:
            self.rgb_list.append(rgb)
        
        # 메모리 관리: 히스토리가 너무 길어지면 오래된 것부터 제거
        # 논문: short-term buffer B=64, long-term은 자동으로 merge됨
        max_history = 256  # 여유있게 설정
        if len(self.rgb_list) > max_history:
            excess = len(self.rgb_list) - max_history
            self.rgb_list = self.rgb_list[excess:]
        
        # 새 프레임 수 설정 (논문의 online token merging용)
        # 이 값은 모델이 새로 추가된 프레임 수를 알기 위해 사용됨
        self.model.get_model().new_frames = len(new_frames)

        navigation_qs = self.promt_template.format(data["instruction"])
        navigation = self.predict_inference(navigation_qs)
        
        action_list = navigation.split(" ")

        traj = [[0.0, 0.0, 0.0]]
        for action in action_list: 
            if action == "stop":
                waypoint = [x + y for x, y in zip(traj[-1], [0.0, 0.0, 0.0])]
                traj = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                break
            elif action == "forward":
                waypoint = [x + y for x, y in zip(traj[-1], [0.5, 0.0, 0.0])]
                traj.append(waypoint)
            elif action == "left":
                waypoint = [x + y for x, y in zip(traj[-1], [0.0, 0.0, -np.deg2rad(30)])]
                traj.append(waypoint)
            elif action == "right":
                waypoint = [x + y for x, y in zip(traj[-1], [0.0, 0.0, np.deg2rad(30)])]
                traj.append(waypoint)
                                    
        if len(action_list) == 0:
            raise ValueError("No action found in the output")
            
        self.executed_steps += 1
        self.latest_action = {"step": self.executed_steps, "path": [traj], "actions": action_list}
        return self.latest_action.copy()


class UniNaVidROS2Node(Node):
    """ROS2 노드: Non-blocking navigation 완전 구현 (논문 준수)"""
    
    def __init__(self, model_path, instruction, image_topic='/camera/image_raw', 
                 action_topic='/uninavid/action', action_feedback_topic='/uninavid/action_feedback',
                 image_request_topic='/uninavid/request_image', action_timeout=5.0):
        super().__init__('uninavid_ros2_node')
        
        # 파라미터
        self.instruction = instruction
        self.action_timeout = action_timeout  # 액션 피드백 타임아웃 (초)
        
        # Uni-NaVid 에이전트 초기화
        self.agent = UniNaVid_Agent(model_path)
        self.agent.reset()
        
        # 이미지 처리
        self.bridge = CvBridge()
        
        # 상태 관리
        self.state = AgentState.IDLE
        self.lock = threading.Lock()
        
        # 이미지 관리
        self.latest_image = None
        self.latest_image_time = None
        self.image_received = False
        self.waiting_for_action_completion_image = False  # 액션 완료 후 이미지 대기 플래그
        
        # 이미지 히스토리 관리 (논문의 online token merging용)
        # 프레임 카운터 기반 접근법 (deque maxlen 동기화 문제 해결)
        self.image_history = []  # 일반 리스트 사용
        self.max_history = 128  # 논문: short-term buffer B=64 + 여유
        self.total_frames_received = 0  # 총 수신한 프레임 수
        self.last_sent_frame_count = 0  # 마지막으로 전송한 프레임 카운트
        self.last_action_time = None  # 액션 타임아웃 체크용
        
        # 액션 관리
        self.pending_actions = []  # 예측된 4개 액션
        self.current_action_index = 0  # 현재 실행 중인 액션 인덱스
        self.action_timestamp = {}  # 각 액션의 타임스탬프 (최신 명령 우선 처리용)
        self.next_action_id = 0  # 액션 ID (최신 명령 식별용)
        
        # 추론 관리 (비동기)
        self.inference_thread = None
        self.inference_queue = Queue()  # 추론 요청 큐
        self.inference_result_queue = Queue()  # 추론 결과 큐
        self.is_inferring = False
        
        # 통계
        self.stats = {
            'total_steps': 0,
            'total_inferences': 0,
            'total_actions_executed': 0,
            'actions_cancelled': 0,  # 최신 명령으로 인해 취소된 액션
            'actions_timeout': 0,  # 타임아웃 발생 횟수
            'avg_inference_time': 0.0,
            'inference_times': deque(maxlen=100)
        }
        
        # ROS2 Subscriber
        self.image_subscription = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )
        
        # 액션 완료 피드백 Subscriber
        self.action_feedback_subscription = self.create_subscription(
            Bool,
            action_feedback_topic,
            self.action_feedback_callback,
            10
        )
        
        # ROS2 Publisher
        self.action_publisher = self.create_publisher(
            String,
            action_topic,
            10
        )
        
        self.status_publisher = self.create_publisher(
            String,
            '/uninavid/status',
            10
        )
        
        # 이미지 요청 Publisher (논문: "Upon completing each action, the robot captures a new image")
        self.image_request_publisher = self.create_publisher(
            Empty,
            image_request_topic,
            10
        )
        
        # 타이머: 상태 머신 관리
        self.timer = self.create_timer(0.05, self.state_machine)  # 20Hz로 상태 체크
        
        # 추론 스레드 시작
        self.start_inference_thread()
        
        self.get_logger().info('Uni-NaVid ROS2 Node initialized (Non-blocking v4 - Paper Compliant)')
        self.get_logger().info(f'Instruction: {self.instruction}')
        self.get_logger().info(f'Image topic: {image_topic}')
        self.get_logger().info(f'Action topic: {action_topic}')
        self.get_logger().info(f'Image request topic: {image_request_topic}')
        self.get_logger().info('Waiting for first image...')
        
    def image_callback(self, msg):
        """이미지 콜백: 최신 이미지만 유지 (10Hz) + 히스토리 누적"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            current_time = time.time()
            
            with self.lock:
                self.latest_image = cv_image
                self.latest_image_time = current_time
                self.image_received = True
                
                # 히스토리에 추가 (논문의 online token merging용)
                self.image_history.append(cv_image)
                self.total_frames_received += 1
                
                # 히스토리 길이 제한 (오래된 것부터 제거)
                if len(self.image_history) > self.max_history:
                    self.image_history.pop(0)
                
                # 액션 완료 후 이미지 대기 중이었다면 플래그 해제
                if self.waiting_for_action_completion_image:
                    self.waiting_for_action_completion_image = False
                    self.get_logger().info(
                        f'Image received after action completion '
                        f'(history: {len(self.image_history)} frames, total: {self.total_frames_received})'
                    )
                
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
    
    def action_feedback_callback(self, msg):
        """액션 완료 피드백 콜백 - 논문: "Upon completing each action, the robot captures a new image" """
        if not msg.data:  # 액션 완료가 아님
            return
            
        with self.lock:
            if self.state != AgentState.EXECUTING_ACTIONS:
                return
            
            # 현재 실행 완료된 액션 확인 (인덱스 증가 전)
            current_action = (
                self.pending_actions[self.current_action_index] 
                if self.current_action_index < len(self.pending_actions) 
                else None
            )
            
            self.get_logger().info(
                f'Action {self.current_action_index + 1}/{len(self.pending_actions)} ({current_action}) completed'
            )
            
            # Stop 액션이면 네비게이션 완료 (인덱스 증가 전에 체크)
            if current_action == "stop":
                self.get_logger().info('Stop action executed. Navigation complete.')
                self.state = AgentState.IDLE
                self.pending_actions = []
                self.current_action_index = 0
                return
            
            # 논문 요구사항: 액션 완료 후 새 이미지 캡처 요청
            self.image_request_publisher.publish(Empty())  # 이미지 캡처 요청
            self.waiting_for_action_completion_image = True
            self.last_action_time = None  # 타임아웃 리셋
            
            self.current_action_index += 1
            
            # 모든 액션 완료
            if self.current_action_index >= len(self.pending_actions):
                self.pending_actions = []
                self.current_action_index = 0
                self.state = AgentState.WAITING_ACTION_IMAGE
                self.get_logger().info('All actions completed. Waiting for final image before new inference...')
            else:
                # 다음 액션은 이미지 도착 후 실행 (state_machine에서 처리)
                self.state = AgentState.WAITING_ACTION_IMAGE
    
    def start_inference_thread(self):
        """추론 전용 스레드 시작 (GPU 블로킹 방지)"""
        def inference_worker():
            while True:
                try:
                    # 추론 요청 대기
                    image_data = self.inference_queue.get(timeout=1.0)
                    if image_data is None:  # 종료 신호
                        break
                    
                    new_frames, instruction, inference_id = image_data
                    self.is_inferring = True
                    
                    # 추론 수행 (시간 측정)
                    inference_start = time.time()
                    try:
                        result = self.agent.act_with_history({
                            'instruction': instruction,
                            'observations': new_frames  # 새 프레임 리스트만 전달
                        })
                        inference_time = time.time() - inference_start
                        
                        # 통계 업데이트
                        with self.lock:
                            self.stats['total_inferences'] += 1
                            self.stats['inference_times'].append(inference_time)
                            if len(self.stats['inference_times']) > 0:
                                self.stats['avg_inference_time'] = np.mean(self.stats['inference_times'])
                        
                        # 결과 전달 (inference_id 포함)
                        self.inference_result_queue.put(('success', result, inference_time, inference_id))
                        self.get_logger().info(
                            f'Inference {inference_id} completed in {inference_time:.3f}s '
                            f'(processed {len(new_frames)} new frames)'
                        )
                        
                    except Exception as e:
                        self.inference_result_queue.put(('error', str(e), None, inference_id))
                        self.get_logger().error(f'Inference {inference_id} error: {str(e)}')
                    
                    finally:
                        self.is_inferring = False
                        
                except QueueEmpty:
                    continue
                except Exception as e:
                    self.get_logger().error(f'Inference worker error: {str(e)}')
        
        self.inference_thread = threading.Thread(target=inference_worker, daemon=True)
        self.inference_thread.start()
        self.get_logger().info('Inference thread started')
    
    def _process_inference_results(self):
        """추론 결과 처리 (최신 우선)"""
        while True:  # 큐에 있는 모든 결과 확인
            try:
                result_type, result_data, inference_time, inference_id = self.inference_result_queue.get_nowait()
                
                if result_type == 'success':
                    result, inf_time = result_data, inference_time
                    actions = result['actions']
                    
                    # Stop 액션이 첫 번째면 즉시 중단
                    if len(actions) > 0 and actions[0] == "stop":
                        with self.lock:
                            self.pending_actions = ["stop"]
                            self.current_action_index = 0
                            self.state = AgentState.EXECUTING_ACTIONS
                            self.stats['total_steps'] = result['step']
                        
                        self.execute_next_action()
                        self.get_logger().info('Stop action predicted. Executing immediately.')
                        continue
                    
                    with self.lock:
                        # 논문 요구사항: 여러 명령이 도착하면 최신 명령 우선
                        if self.state == AgentState.EXECUTING_ACTIONS:
                            # 기존 액션 중단하고 최신 명령 실행
                            cancelled_count = len(self.pending_actions) - self.current_action_index
                            if cancelled_count > 0:
                                self.stats['actions_cancelled'] += cancelled_count
                                self.get_logger().warn(
                                    f'Cancelling {cancelled_count} pending actions. '
                                    f'Executing most recent command (inference {inference_id})'
                                )
                        
                        # 최신 명령으로 업데이트
                        self.pending_actions = actions
                        self.current_action_index = 0
                        self.state = AgentState.EXECUTING_ACTIONS
                        self.stats['total_steps'] = result['step']
                        self.action_timestamp[inference_id] = time.time()
                        
                    # 첫 번째 액션 실행
                    self.execute_next_action()
                    
                    # 상태 publish
                    status_msg = String()
                    status_msg.data = f"Step {result['step']}, Actions: {', '.join(actions)}, Inference: {inf_time:.3f}s, ID: {inference_id}"
                    self.status_publisher.publish(status_msg)
                    
                    self.get_logger().info(
                        f"Step {result['step']}: Predicted {len(actions)} actions: {actions} (inference {inference_id})"
                    )
                    
                elif result_type == 'error':
                    self.get_logger().error(f'Inference {inference_id} failed: {result_data}')
                    with self.lock:
                        if self.state == AgentState.INFERRING:
                            self.state = AgentState.WAITING_FOR_IMAGE
                    
            except QueueEmpty:
                break
    
    def _request_inference_if_ready(self):
        """추론 요청 (새 프레임만 전달, 프레임 카운터 기반, lock 범위 최소화)"""
        if self.is_inferring or self.waiting_for_action_completion_image:
            return
        
        with self.lock:
            if not self.image_history:
                return
            
            # 새로 받은 프레임 수 계산 (프레임 카운터 기반)
            new_frame_count = self.total_frames_received - self.last_sent_frame_count
            
            if new_frame_count <= 0:
                return  # 새 프레임 없음
            
            # 새 프레임만 추출 (히스토리 끝에서)
            if new_frame_count <= len(self.image_history):
                new_frames = self.image_history[-new_frame_count:]
            else:
                # 히스토리가 갱신되어 일부 프레임이 삭제됨 - 전체 히스토리 사용
                new_frames = list(self.image_history)
                self.get_logger().warn(
                    f'History was truncated. Using all {len(new_frames)} frames '
                    f'(expected {new_frame_count} new frames)'
                )
            
            if not new_frames:
                return
            
            # 마지막 전송 프레임 카운트 업데이트
            self.last_sent_frame_count = self.total_frames_received
            inference_id = self.next_action_id
            self.next_action_id += 1
            self.state = AgentState.INFERRING
            
            # 로그용 값 저장 (lock 안에서)
            total_history = len(self.image_history)
            sent_frame_count = self.last_sent_frame_count
        
        # lock 밖에서 큐에 넣기 (lock 범위 최소화)
        self.inference_queue.put((new_frames, self.instruction, inference_id))
        self.get_logger().info(
            f'Inference {inference_id} requested with {len(new_frames)} new frames '
            f'(total history: {total_history}, sent_count: {sent_frame_count}/{self.total_frames_received})'
        )
    
    def state_machine(self):
        """상태 머신: Non-blocking navigation 로직 (논문 완전 준수)"""
        try:
            # 추론 결과 확인 (비동기) - 논문: "prioritizes and executes the most recent command"
            self._process_inference_results()
            
            # 액션 타임아웃 체크
            current_time = time.time()
            with self.lock:
                if (self.state == AgentState.EXECUTING_ACTIONS and 
                    self.last_action_time is not None and
                    current_time - self.last_action_time > self.action_timeout):
                    self.get_logger().warn(
                        f'Action timeout ({self.action_timeout}s). Moving to next action.'
                    )
                    self.stats['actions_timeout'] += 1
                    # 타임아웃 시 다음 액션으로 (이미지 요청은 이미 보냈을 것으로 가정)
                    self.current_action_index += 1
                    if self.current_action_index >= len(self.pending_actions):
                        self.pending_actions = []
                        self.current_action_index = 0
                        self.state = AgentState.WAITING_ACTION_IMAGE
                    else:
                        self.state = AgentState.WAITING_ACTION_IMAGE
                    self.last_action_time = None
            
            # 상태별 처리
            with self.lock:
                current_state = self.state
            
            if current_state == AgentState.IDLE:
                # 초기 상태: 첫 이미지 대기
                if self.image_received:
                    with self.lock:
                        self.state = AgentState.WAITING_FOR_IMAGE
                    self.get_logger().info('First image received. Starting navigation...')
            
            elif current_state == AgentState.WAITING_FOR_IMAGE:
                # 새 이미지 대기 및 추론 요청
                # 논문: 모든 액션 완료 후 새 이미지가 도착하면 추론 요청
                self._request_inference_if_ready()
            
            elif current_state == AgentState.INFERRING:
                # 추론 중: 결과 대기 (이미 위에서 처리)
                pass
            
            elif current_state == AgentState.WAITING_ACTION_IMAGE:
                # 액션 완료 후 이미지 대기
                if self.image_received and not self.waiting_for_action_completion_image:
                    with self.lock:
                        if len(self.pending_actions) > 0 and self.current_action_index < len(self.pending_actions):
                            # 아직 실행할 액션이 남음 → 다음 액션 실행
                            self.state = AgentState.EXECUTING_ACTIONS
                            self.execute_next_action()
                        else:
                            # 모든 액션 완료 → 새 추론 요청
                            self.state = AgentState.WAITING_FOR_IMAGE
            
            elif current_state == AgentState.EXECUTING_ACTIONS:
                # 액션 실행 중: 피드백 대기 (콜백에서 처리)
                # 타임아웃은 위에서 처리
                pass
                
        except Exception as e:
            self.get_logger().error(f'State machine error: {str(e)}')
    
    def execute_next_action(self):
        """다음 액션 실행"""
        with self.lock:
            if self.current_action_index < len(self.pending_actions):
                action = self.pending_actions[self.current_action_index]
                
                # ROS2 메시지로 액션 publish
                action_msg = String()
                action_msg.data = action
                self.action_publisher.publish(action_msg)
                
                self.stats['total_actions_executed'] += 1
                self.last_action_time = time.time()  # 타임아웃 체크용
                
                self.get_logger().info(
                    f"Executing action {self.current_action_index + 1}/{len(self.pending_actions)}: {action}"
                )
                return True
        return False
    
    def get_statistics(self):
        """통계 정보 반환"""
        with self.lock:
            return {
                'total_steps': self.stats['total_steps'],
                'total_inferences': self.stats['total_inferences'],
                'total_actions_executed': self.stats['total_actions_executed'],
                'actions_cancelled': self.stats['actions_cancelled'],
                'actions_timeout': self.stats['actions_timeout'],
                'avg_inference_time': self.stats['avg_inference_time'],
                'current_state': self.state.value,
                'pending_actions_count': len(self.pending_actions),
                'current_action_index': self.current_action_index,
                'image_history_length': len(self.image_history),
                'total_frames_received': self.total_frames_received,
                'last_sent_frame_count': self.last_sent_frame_count
            }
    
    def shutdown(self):
        """노드 종료"""
        self.get_logger().info('Shutting down Uni-NaVid ROS2 Node')
        # 추론 스레드 종료
        self.inference_queue.put(None)
        if self.inference_thread:
            self.inference_thread.join(timeout=2.0)
        
        # 최종 통계 출력
        stats = self.get_statistics()
        self.get_logger().info(f'Final statistics: {stats}')


def main(args=None):
    parser = argparse.ArgumentParser(description='Uni-NaVid ROS2 온라인 평가 (Non-blocking v4 - Paper Compliant)')
    parser.add_argument('--model_path', type=str, 
                       default='model_zoo/uninavid-7b-full-224-video-fps-1-grid-2',
                       help='모델 경로')
    parser.add_argument('--instruction', type=str, required=True,
                       help='네비게이션 지시사항')
    parser.add_argument('--image_topic', type=str, default='/camera/image_raw',
                       help='이미지 토픽 이름')
    parser.add_argument('--action_topic', type=str, default='/uninavid/action',
                       help='액션 publish 토픽 이름')
    parser.add_argument('--action_feedback_topic', type=str, default='/uninavid/action_feedback',
                       help='액션 완료 피드백 토픽 이름')
    parser.add_argument('--image_request_topic', type=str, default='/uninavid/request_image',
                       help='이미지 캡처 요청 토픽 이름 (논문: "Upon completing each action, the robot captures a new image")')
    parser.add_argument('--action_timeout', type=float, default=5.0,
                       help='액션 피드백 타임아웃 (초)')
    
    args = parser.parse_args()
    
    # ROS2 초기화
    rclpy.init(args=None)
    
    # 노드 생성
    node = UniNaVidROS2Node(
        model_path=args.model_path,
        instruction=args.instruction,
        image_topic=args.image_topic,
        action_topic=args.action_topic,
        action_feedback_topic=args.action_feedback_topic,
        image_request_topic=args.image_request_topic,
        action_timeout=args.action_timeout
    )
    
    try:
        # 노드 실행
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

