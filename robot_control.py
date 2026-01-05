import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import TwistStamped  # Twist -> TwistStamped 변경!
import time

class RobotActionController(Node):
    def __init__(self):
        super().__init__('robot_action_controller')
        
        # 1. Subscriber (명령 받는 곳)
        self.subscription = self.create_subscription(
            String,
            'sign',
            self.listener_callback,
            10
        )
        
        # 2. Publisher (수정됨: 토픽 이름 & 메시지 타입)
        self.publisher_ = self.create_publisher(
            TwistStamped, 
            '/scout_mini_base_controller/cmd_vel', 
            10
        )
        
        # 3. Control Timer (0.1s 마다 실행)
        self.timer = self.create_timer(0.1, self.control_loop)
        
        # State Variables
        self.current_twist = TwistStamped() # 타입 변경
        self.current_twist.header.frame_id = 'base_link' # 필수 설정
        
        self.action_end_time = 0.0
        self.is_moving = False

        self.get_logger().info("Scout Mini Controller Ready. Waiting for /sign topic...")

    def listener_callback(self, msg):
        command = msg.data.lower().strip()
        self.get_logger().info(f"Received: '{command}'")
        
        # TwistStamped는 내부에 twist가 한 겹 더 있음
        new_msg = TwistStamped()
        new_msg.header.frame_id = 'base_link'
        
        duration = 0.0

        if command in ['straight', 'forward']:
            new_msg.twist.linear.x = 0.2  # 구조 변경: .twist.linear.x
            duration = 3.0  # 테스트용으로 3초로 줄임
            
        elif command == 'left':
            new_msg.twist.angular.z = 0.5
            duration = 2.0
            
        elif command == 'right':
            new_msg.twist.angular.z = -0.5
            duration = 2.0
            
        elif command == 'stop':
            new_msg.twist.linear.x = 0.0
            new_msg.twist.angular.z = 0.0
            duration = 0.0
            
        else:
            self.get_logger().warn(f"Unknown: {command}")
            return

        # 상태 업데이트
        self.current_twist = new_msg
        
        # 종료 시간 설정
        now_sec = self.get_clock().now().nanoseconds / 1e9
        self.action_end_time = now_sec + duration
        self.is_moving = True

    def control_loop(self):
        # 움직이는 중이 아니면 아무것도 안 함 (불필요한 트래픽 방지)
        if not self.is_moving:
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9
        
        if now_sec < self.action_end_time:
            # 시간 갱신 (Stamped 메시지는 시간이 최신이어야 함)
            self.current_twist.header.stamp = self.get_clock().now().to_msg()
            self.publisher_.publish(self.current_twist)
        else:
            # 시간 종료 -> 정지
            self.stop_robot()
            self.is_moving = False

    def stop_robot(self):
        stop_msg = TwistStamped()
        stop_msg.header.frame_id = 'base_link'
        stop_msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher_.publish(stop_msg)
        self.get_logger().info(">> Action Finished (Stopped).")

def main(args=None):
    rclpy.init(args=args)
    node = RobotActionController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop_robot()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
