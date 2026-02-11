import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from geometry_msgs.msg import TwistStamped
import math
import json
import threading # 스레드 잠금용

class RobotActionController(Node):
    def __init__(self):
        super().__init__('robot_action_controller')
        
        self.callback_group = ReentrantCallbackGroup()
        self.lock = threading.Lock() # 공유 자원 보호용 락

        # QoS 설정 (Best Effort) - depth=1로 설정하여 항상 최신 메시지만 유지
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=1  # 최신 메시지만 유지하여 중복 처리 방지
        )

        self.subscription = self.create_subscription(
            String, 'sign', self.listener_callback, qos_profile,
            callback_group=self.callback_group
        )
        
        self.publisher_ = self.create_publisher(
            TwistStamped, '/scout_mini_base_controller/cmd_vel', qos_profile
        )
        
        # 제어 주기 (10Hz)
        self.timer = self.create_timer(0.1, self.control_loop, callback_group=self.callback_group)
        
        # --- 상태 변수 및 설정 ---
        self.action_queue = []
        self.current_action_idx = -1
        self.action_end_time = 0.0  # 초기값을 0으로 설정하여 첫 메시지 즉시 실행
        self.is_moving = False
        self.new_queue_received = False  # 큐가 업데이트되었는지 플래그
        self.action_step_count = 0  # 실행한 액션의 step index

        # 액션별 파라미터 매핑 (속도, 각속도, 지속시간)
        # MOVE_DURATION=0.5s 기준
        self.ACTION_MAP = {
            'forward':  (0.5, 0.0, 0.5),
            'left':     (0.0, math.radians(30)/0.5, 0.5),
            'right':    (0.0, -math.radians(30)/0.5, 0.5),
            'stop':     (0.0, 0.0, 0.1)
        }

        self.current_twist = TwistStamped()
        self.current_twist.header.frame_id = 'base_link'

        print("Uni-NaVid Optimized Controller Ready.")

    def listener_callback(self, msg):
        """
        sign 토픽에서 액션 리스트를 받아 큐를 업데이트
        논문 요구사항: 새로운 액션이 들어오면 큐를 업데이트 (이전 것은 버리고 최신 것으로)
        """
        try:
            # 홑따옴표 처리 및 JSON 파싱
            raw_data = msg.data.replace("'", '"')
            new_actions = json.loads(raw_data)
            
            if isinstance(new_actions, list) and new_actions:
                # 소문자화/공백제거 전처리
                processed_actions = [a.lower().strip() for a in new_actions]
                
                # 모든 공유 변수 접근을 락으로 보호
                with self.lock:
                    # 논문 요구사항: 새로운 액션이 들어오면 큐를 업데이트 (이전 것은 버림)
                    # depth=1 QoS로 이미 최신 메시지만 받지만, 명시적으로 큐 업데이트
                    self.action_queue = processed_actions
                    self.new_queue_received = True
                    
                    # 첫 번째 메시지이거나 현재 액션이 없으면 즉시 시작
                    # (action_end_time이 0이거나 현재 시간보다 작으면)
                    current_time_msg = self.get_clock().now()
                    now_sec = current_time_msg.nanoseconds / 1e9
                    
                    # 모든 상태 변수를 락 내에서 체크
                    if self.action_end_time == 0.0 or now_sec >= self.action_end_time:
                        if not self.is_moving or self.current_action_idx == -1:
                            # 큐가 업데이트되었고 액션이 없으면 첫 액션부터 시작
                            self.current_action_idx = 0
                            self.new_queue_received = False
                            # 새 큐를 받아서 실행할 때는 큐 내용도 출력
                            self._start_next_action_locked(now_sec, current_time_msg, show_queue=True)
                
                # 로그는 락 밖에서 출력 (간단한 포맷)
                # print(f"Queue Updated: {processed_actions}")  # 필요시 주석 해제
        except Exception as e:
            print(f"Parse Error: {e}")

    def control_loop(self):
        # 시간 정보 한 번만 가져오기
        current_time_msg = self.get_clock().now()
        now_sec = current_time_msg.nanoseconds / 1e9

        # 모든 공유 변수 접근을 락으로 보호
        with self.lock:
            # action_end_time 체크도 락 내에서 수행
            action_should_transition = (now_sec >= self.action_end_time)
            
            # 정지 메시지 초기화
            stop_msg = None
            
            if action_should_transition:
                # 논문 요구사항: 액션이 끝났을 때 큐가 업데이트되었으면 첫 액션부터, 아니면 다음 액션
                if self.new_queue_received:
                    # 큐가 업데이트되었으면 첫 액션부터 순차적으로 실행
                    self.current_action_idx = 0
                    self.new_queue_received = False
                    # 새 큐를 받아서 실행할 때는 큐 내용도 출력
                    self._start_next_action_locked(now_sec, current_time_msg, show_queue=True)
                
                # 현재 액션이 진행 중이고 다음 액션이 있으면 다음 액션으로
                # 인덱스 체크와 접근을 단일 락 블록 내에서 수행
                elif self.current_action_idx != -1:
                    # 큐 길이 체크도 락 내에서 수행
                    queue_len = len(self.action_queue)
                    if self.current_action_idx < queue_len - 1:
                        # 현재 큐에서 다음 액션 실행 (큐 내용 출력 안 함)
                        self.current_action_idx += 1
                        self._start_next_action_locked(now_sec, current_time_msg, show_queue=False)
                    # 모든 액션이 완료되었으면 정지
                    elif self.is_moving:
                        stop_msg = self._stop_robot_locked(current_time_msg)
                        self.is_moving = False
                        self.current_action_idx = -1
                        self.action_end_time = 0.0  # 다음 메시지를 위해 리셋
            
            # 움직임 유지 (Heartbeat) - 락 내에서 상태 확인 후 락 밖에서 발행
            is_moving_now = self.is_moving
            if is_moving_now:
                # current_twist는 락 내에서 복사 (읽기)
                twist_to_publish = TwistStamped()
                twist_to_publish.header.frame_id = self.current_twist.header.frame_id
                twist_to_publish.header.stamp = current_time_msg.to_msg()
                twist_to_publish.twist.linear.x = self.current_twist.twist.linear.x
                twist_to_publish.twist.angular.z = self.current_twist.twist.angular.z
            else:
                twist_to_publish = None
        
        # 발행은 락 밖에서 수행 (성능 최적화 및 데드락 방지)
        if stop_msg is not None:
            self.publisher_.publish(stop_msg)
            print(">> Finished.")
        if twist_to_publish is not None:
            self.publisher_.publish(twist_to_publish)

    def _start_next_action_locked(self, start_time, time_msg, show_queue=False):
        """
        락이 이미 획득된 상태에서 호출되는 내부 메서드
        모든 공유 변수 접근이 락 내에서 수행됨
        """
        # 큐와 인덱스 체크를 단일 락 블록 내에서 수행
        if not self.action_queue:
            return
        
        # 인덱스 유효성 검사도 락 내에서 수행
        if self.current_action_idx < 0 or self.current_action_idx >= len(self.action_queue):
            idx = self.current_action_idx
            queue_len = len(self.action_queue)
            self.is_moving = False
            # 로그는 락 밖에서 출력 (간단한 포맷)
            print(f"Invalid action index: {idx}, queue length: {queue_len}")
            return
        
        cmd = self.action_queue[self.current_action_idx]
        params = self.ACTION_MAP.get(cmd)
        
        # 큐 내용 복사 (출력용)
        queue_copy = self.action_queue.copy() if show_queue else None

        if params:
            v_x, v_z, duration = params
            # 모든 공유 변수 쓰기를 락 내에서 수행
            self.current_twist.twist.linear.x = v_x
            self.current_twist.twist.angular.z = v_z
            self.current_twist.header.frame_id = 'base_link'
            self.current_twist.header.stamp = time_msg.to_msg()
            self.action_end_time = start_time + duration
            self.is_moving = True
            # step index 증가 및 로그용 정보 저장
            self.action_step_count += 1
            action_num = self.current_action_idx + 1
            step_count = self.action_step_count
            invalid_cmd = None
        else:
            self.is_moving = False
            action_num = None
            step_count = None
            invalid_cmd = cmd  # 로그용 저장
        
        # 로그는 락 밖에서 출력 (간단한 포맷, step index 포함)
        if invalid_cmd is not None:
            print(f"Invalid action: {invalid_cmd}")
        elif action_num is not None:
            if show_queue and queue_copy:
                queue_str = ' '.join(queue_copy)
                print(f"Step {step_count}: Action [{action_num}/4] {cmd} | Queue: {queue_str}")
            else:
                print(f"Step {step_count}: Action [{action_num}/4] {cmd}")

    def _stop_robot_locked(self, time_msg):
        """
        락이 이미 획득된 상태에서 호출되는 내부 메서드
        정지 메시지 생성만 수행하고 반환 (발행은 호출자가 락 밖에서 수행)
        """
        # 정지 메시지 생성 (발행은 호출자가 락 밖에서 수행)
        stop_msg = TwistStamped()
        stop_msg.header.frame_id = 'base_link'
        stop_msg.header.stamp = time_msg.to_msg()
        # 속도는 이미 0으로 초기화됨
        
        # current_twist도 0으로 설정 (일관성 유지)
        self.current_twist.twist.linear.x = 0.0
        self.current_twist.twist.angular.z = 0.0
        self.current_twist.header.stamp = time_msg.to_msg()
        
        # 메시지를 반환하여 락 밖에서 발행하도록 함
        return stop_msg

def main(args=None):
    rclpy.init(args=args)
    node = RobotActionController()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
