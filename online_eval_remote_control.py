# coding: utf-8
import os
import json
import cv2
import numpy as np
import imageio
import torch
import time
import argparse
import rclpy
from std_msgs.msg import String
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Try to import RealSense, but make it optional
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("Warning: pyrealsense2 not available. Will use webcam fallback.")

from uninavid.mm_utils import get_model_name_from_path
from uninavid.model.builder import load_pretrained_model
from uninavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from uninavid.conversation import conv_templates, SeparatorStyle
from uninavid.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

seed = 30
torch.manual_seed(seed)
np.random.seed(seed)

# 전역 변수 설정
current_frame = None
bridge = CvBridge()

def image_callback(msg):
    global current_frame
    try:
        # 압축된 데이터를 numpy array로 변환 후 OpenCV로 디코딩
        np_arr = np.frombuffer(msg.data, np.uint8)
        bgr_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        current_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Image Decoding Error: {e}")
        
class UniNaVid_Agent():
    def __init__(self, model_path):
        
        print("Initialize UniNaVid")
        
        self.conv_mode = "vicuna_v1"

        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, get_model_name_from_path(model_path))

        assert self.image_processor is not None

        print("Initialization Complete")
        
        self.promt_template = "Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations and an image of the current observation <image>. Your assigned task is: '{}'. Analyze this series of images to determine your next four actions. The predicted action should be one of the following: forward, left, right, or stop."
        self.rgb_list = []
        self.count_id = 0
        # Keep last raw model output for per-step logging/debugging (not drawn on GIF)
        self.last_navigation_output = None
        self.reset()

    def process_images(self, rgb_list):

        
        batch_image = np.asarray(rgb_list)
        self.model.get_model().new_frames = len(rgb_list)
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
        self.rgb_list = []

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

        self.last_navigation_output = outputs
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


    def act(self, data):
    
        rgb = data["observations"]
        self.rgb_list.append(rgb)


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

                                    
        if len(action_list)==0:
            raise ValueError("No action found in the output")
        
        # Accumulate actions in the queue
        self.pending_action_list.extend(action_list)
            
        self.executed_steps += 1
            
        self.latest_action = {"step": self.executed_steps, "path":[traj], "actions":action_list}
            
        return self.latest_action.copy()



def draw_traj_arrows_fpv(
    img,
    actions,
    arrow_len=10,                
    arrow_gap=2,                 
    arrow_color=(0, 255, 0),    
    arrow_thickness=2,
    tipLength=0.35,
    stop_color=(0, 0, 255),      
    stop_radius=5
):
 
    out = img.copy()
    h, w = out.shape[:2]

    base_x, base_y = w // 2, int(h * 0.95)

    for i, action in enumerate(actions):
        if action == "stop":
            waypoint = [0.0, 0.0, 0.0]
        elif action == "forward":
            waypoint = [0.5, 0.0, 0.0]
        elif action == "left":
            waypoint = [0.0, 0.0, -np.deg2rad(30)]
        elif action == "right":
            waypoint = [0.0, 0.0, np.deg2rad(30)]
        else:
            continue  

        x, y, yaw = waypoint

        start = (
            int(base_x),
            int(base_y - i * (arrow_len + arrow_gap))
        )

        if action == "stop":
            cv2.circle(out, start, stop_radius, stop_color, 2)
        else:
            end = (
                int(start[0] + arrow_len * np.sin(yaw)),
                int(start[1] - arrow_len * np.cos(yaw))
            )
            cv2.arrowedLine(out, start, end, arrow_color, arrow_thickness, tipLength=tipLength)
    
    return out


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Online evaluation with camera')
    parser.add_argument('--instruction', type=str, required=True, help='Navigation instruction/task description')
    parser.add_argument('--output_dir', type=str, default='Real_test_output', help='Output directory to save results (default: output_dir)')
    parser.add_argument('--max_steps', type=int, default=None, help='Maximum number of steps to process (default: unlimited)')
    parser.add_argument('--width', type=int, default=640, help='Camera width (default: 640)')
    parser.add_argument('--height', type=int, default=480, help='Camera height (default: 480)')
    parser.add_argument('--fps', type=int, default=1, help='Camera FPS for RealSense (default: 1, not used for webcam)')
    parser.add_argument('--model_path', type=str, default='./model_zoo/uninavid-7b-full-224-video-fps-1-grid-2', help='Path to model (default: ./model_zoo/uninavid-7b-full-224-video-fps-1-grid-2)')
    parser.add_argument('--save_video', action='store_true', help='Save visualization as video (MP4)')
    parser.add_argument('--display', action='store_true', help='Display frames in real-time')
    parser.add_argument('--camera_type', type=str, default='auto', choices=['auto', 'realsense', 'webcam'], 
                        help='Camera type: auto (try RealSense first, fallback to webcam), realsense, or webcam (default: auto)')
    parser.add_argument('--camera_id', type=int, default=0, help='Webcam camera ID (default: 0)')
    parser.add_argument('--no_step_log', action='store_true',
                        help='Disable per-step JSONL logging (default: enabled)')
    parser.add_argument('--step_log_name', type=str, default='step_log.jsonl',
                        help='Filename for per-step JSONL log inside output_dir (default: step_log.jsonl)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
        #ros
    rclpy.init(args=None)
    ros_node = rclpy.create_node('uninavid_host_node')
    publisher_sign = ros_node.create_publisher(String, 'sign', 10)

    qos_profile = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT, # 신뢰성보다 속도 우선 (최신성 유지)
        history=HistoryPolicy.KEEP_LAST,
        depth=1 # 큐에 딱 1개만 남김
    )
    
    # 이미지 구독자 추가 (확인된 실제 토픽명 적용)
    image_sub = ros_node.create_subscription(
    	CompressedImage, 
    	'/camera/camera/color/image_raw/compressed',
    	image_callback, 
    	qos_profile)
    print("*" * 10 + " ROS2 Subscriber Ready " + "*" * 10)
    
    # 기존 카메라 초기화 코드(pipeline, cap 등)는 모두 삭제/주석 처리
    camera_type = 'ros2_stream'
    # Initialize agent
    agent = UniNaVid_Agent(args.model_path)
    agent.reset()
        
    result_vis_list = []
    step_count = 0
    step_log_f = None
    step_log_path = None

    if not args.no_step_log:
        step_log_path = os.path.join(output_dir, args.step_log_name)
        step_log_f = open(step_log_path, 'a', encoding='utf-8')
        print(f"Step log enabled: writing JSONL to {step_log_path}")
    
    try:
        print(f"\nStarting online evaluation with instruction: '{args.instruction}'")
        print("Press Ctrl+C to stop\n")
        
        while rclpy.ok():
            rclpy.spin_once(ros_node, timeout_sec=0.01) # ROS2 이벤트 처리
            if current_frame is None:
                continue # 프레임이 올 때까지 대기
            frame = current_frame.copy()
            # Process frame through agent
            t_s = time.time()
            result = agent.act({'instruction': args.instruction, 'observations': frame})
            step_count += 1
            
            inference_time = time.time() - t_s
            print(f"Step {step_count}, inference time: {inference_time:.3f}s, actions: {result['actions']}")
            
            # Get trajectory and actions
            traj = result['path'][0]
            actions = result['actions']
            print("Current frame actions:", actions)
            
            # Publish the most recent action from pending_action_list
            msg = String()
            msg.data = actions[1]
            print(f"Publishing action: {msg.data} (inference time: {inference_time:.3f}s)")
            publisher_sign.publish(msg)

            # Draw visualization
            vis = draw_traj_arrows_fpv(frame, actions, arrow_len=20)
            # Match offline logic: store a stable per-step frame (avoid accidental aliasing)
            result_vis_list.append(vis.copy())

            # Per-step logging (raw model output + actions)
            if step_log_f is not None:
                step_record = {
                    "step": step_count,
                    "timestamp": time.time(),
                    "inference_time_sec": float(inference_time),
                    "model_output_raw": agent.last_navigation_output,
                    "actions": actions,
                    "published_action": msg.data,
                    "traj": traj,
                }
                step_log_f.write(json.dumps(step_record, ensure_ascii=False) + "\n")
                step_log_f.flush()
            
            # Display frame if requested
            if args.display:
                display_frame = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                cv2.imshow('UniNaVid Online Evaluation', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopped by user (pressed 'q')")
                    break
            
    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C)")
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop camera
        print("\nStopping camera...")

    
    # Save results if requested
    if args.save_video and len(result_vis_list) > 0:
        # Save as video (MP4)
        video_path = os.path.join(output_dir, "demo5.mp4")
        print(f"\nSaving visualization to {video_path}...")
    
        # Get frame dimensions from first frame
        h, w = result_vis_list[0].shape[:2]
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 2.0, (w, h))
        
        # Write frames (convert RGB to BGR for OpenCV)
        for frame in result_vis_list:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        
        video_writer.release()
        print(f"Saved {len(result_vis_list)} frames to {video_path}")
    
    print(f"\nCompleted {step_count} steps")
