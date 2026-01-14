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

        # QoS 설정 (Best Effort)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
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
        self.action_end_time = 0.0
        self.is_moving = False
        self.new_queue_received = False

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

        self.get_logger().info("Uni-NaVid Optimized Controller Ready.")

    def listener_callback(self, msg):
        try:
            # 홑따옴표 처리 및 JSON 파싱
            raw_data = msg.data.replace("'", '"')
            new_actions = json.loads(raw_data)
            
            if isinstance(new_actions, list) and new_actions:
                with self.lock: # 데이터 쓰기 시 락 획득
                    # 콜백 단계에서 소문자화/공백제거 전처리 수행
                    self.action_queue = [a.lower().strip() for a in new_actions]
                    self.new_queue_received = True
                self.get_logger().info(f"Queue Updated: {self.action_queue}")
        except Exception as e:
            self.get_logger().error(f"Parse Error: {e}")

    def control_loop(self):
        # 시간 정보 한 번만 가져오기
        current_time_msg = self.get_clock().now()
        now_sec = current_time_msg.nanoseconds / 1e9

        # 액션 교체 로직
        if now_sec >= self.action_end_time:
            with self.lock: # 공유 자원 접근 시 락
                if self.new_queue_received:
                    self.current_action_idx = 0
                    self.new_queue_received = False
                    self.start_next_action(now_sec)
                
                elif self.current_action_idx != -1 and self.current_action_idx < len(self.action_queue) - 1:
                    self.current_action_idx += 1
                    self.start_next_action(now_sec)
                
                elif self.is_moving:
                    self.stop_robot(current_time_msg)
                    self.is_moving = False
                    self.current_action_idx = -1

        # 움직임 유지 (Heartbeat)
        if self.is_moving:
            self.current_twist.header.stamp = current_time_msg.to_msg()
            self.publisher_.publish(self.current_twist)

    def start_next_action(self, start_time):
        if not self.action_queue: return

        cmd = self.action_queue[self.current_action_idx]
        params = self.ACTION_MAP.get(cmd)

        if params:
            v_x, v_z, duration = params
            self.current_twist.twist.linear.x = v_x
            self.current_twist.twist.angular.z = v_z
            self.action_end_time = start_time + duration
            self.is_moving = True
            self.get_logger().info(f"Action [{self.current_action_idx+1}/4]: {cmd}")
        else:
            self.get_logger().warn(f"Invalid action: {cmd}")
            self.is_moving = False

    def stop_robot(self, time_msg):
        msg = TwistStamped()
        msg.header.frame_id = 'base_link'
        msg.header.stamp = time_msg.to_msg()
        self.publisher_.publish(msg)
        self.get_logger().info(">> Finished.")

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
