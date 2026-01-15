import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from geometry_msgs.msg import TwistStamped
import math
import json
import threading
import sys
import tty
import termios

class RobotCalibrationController(Node):
    def __init__(self):
        super().__init__('robot_calibration_controller')
        
        self.callback_group = ReentrantCallbackGroup()
        self.lock = threading.Lock()

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        self.publisher_ = self.create_publisher(
            TwistStamped, '/scout_mini_base_controller/cmd_vel', qos_profile
        )
        
        # 10Hz 제어 루프
        self.timer = self.create_timer(0.1, self.control_loop, callback_group=self.callback_group)
        
        # --- [튜닝 파라미터] 이 값을 수정하며 테스트하세요 ---
        self.T_LINEAR_VEL = 0.5      # 목표: 25cm 이동을 위한 속도
        self.T_ANGULAR_DEG = 30.0    # 목표 각도 (도 단위)
        self.T_DURATION = 0.5        # 동작 지속 시간
        # -----------------------------------------------

        self.action_queue = []
        self.current_action_idx = -1
        self.action_end_time = 0.0
        self.is_moving = False
        self.new_queue_received = False
        self.current_twist = TwistStamped()
        self.current_twist.header.frame_id = 'base_link'

        self.get_logger().info("\n" + "="*40 + 
                               "\nCalibration Mode Ready" +
                               "\nW: Forward | A: Left 30 deg | D: Right 30 deg" +
                               "\n" + "="*40)

    def trigger_manual_action(self, action_name):
        """키보드 입력 시 액션 큐를 강제로 업데이트"""
        with self.lock:
            self.action_queue = [action_name]
            self.new_queue_received = True
            self.get_logger().info(f"Manual Trigger: {action_name.upper()}")

    def control_loop(self):
        current_time_msg = self.get_clock().now()
        now_sec = current_time_msg.nanoseconds / 1e9

        if now_sec >= self.action_end_time:
            with self.lock:
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

        if self.is_moving:
            self.current_twist.header.stamp = current_time_msg.to_msg()
            self.publisher_.publish(self.current_twist)

    def start_next_action(self, start_time):
        cmd = self.action_queue[self.current_action_idx]
        
        v_x, v_z = 0.0, 0.0
        # 튜닝된 파라미터를 기반으로 계산
        if cmd == 'forward':
            v_x = self.T_LINEAR_VEL
        elif cmd == 'left':
            v_z = math.radians(self.T_ANGULAR_DEG) / self.T_DURATION
        elif cmd == 'right':
            v_z = -math.radians(self.T_ANGULAR_DEG) / self.T_DURATION

        self.current_twist.twist.linear.x = v_x
        self.current_twist.twist.angular.z = v_z
        self.action_end_time = start_time + self.T_DURATION
        self.is_moving = True

    def stop_robot(self, time_msg):
        msg = TwistStamped()
        msg.header.frame_id = 'base_link'
        msg.header.stamp = time_msg.to_msg()
        self.publisher_.publish(msg)

def get_key():
    """터미널에서 키 입력을 비동기적으로 읽기 위한 함수"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def keyboard_listener_thread(node):
    """별도 스레드에서 키보드 입력을 무한 루프로 감시"""
    while rclpy.ok():
        key = get_key().lower()
        if key == 'w':
            node.trigger_manual_action('forward')
        elif key == 'a':
            node.trigger_manual_action('left')
        elif key == 'd':
            node.trigger_manual_action('right')
        elif key == '\x03': # Ctrl+C
            break

def main(args=None):
    rclpy.init(args=args)
    node = RobotCalibrationController()
    
    # 키보드 리스너 스레드 시작
    key_thread = threading.Thread(target=keyboard_listener_thread, args=(node,))
    key_thread.daemon = True
    key_thread.start()

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
