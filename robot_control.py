import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from geometry_msgs.msg import TwistStamped
import time

class RobotActionController(Node):
    def __init__(self):
        super().__init__('robot_action_controller')
        
        # Subscriber
        self.subscription = self.create_subscription(
            String,
            'sign',
            self.listener_callback,
            10
        )
        
        # Publisher (Best Effort 유지)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )
        self.publisher_ = self.create_publisher(
            TwistStamped, 
            '/scout_mini_base_controller/cmd_vel', 
            qos_profile
        )
        
        # 0.1초마다 루프 (충분함)
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.current_twist = TwistStamped()
        self.current_twist.header.frame_id = 'base_link'
        
        self.action_end_time = 0.0
        self.is_moving = False

        # [핵심] 명령 갱신 주기(0.5s)보다 약간 길게 설정 (안전 마진)
        self.CMD_DURATION = 0.8 

        self.get_logger().info("Robot Controller Ready. Duration set to 0.8s (Safety Watchdog).")

    def listener_callback(self, msg):
        command = msg.data.lower().strip()
        self.get_logger().info(f"Received: '{command}'")
        
        new_msg = TwistStamped()
        new_msg.header.frame_id = 'base_link'
        
        # 명령이 들어오면 무조건 시간을 갱신해줌
        # (계속 forward가 들어오면 멈추지 않고 계속 감)
        
        if command in ['straight', 'forward']:
            new_msg.twist.linear.x = 0.4  # 속도
            
        elif command == 'left':
            new_msg.twist.angular.z = 0.8 # 회전 속도
            
        elif command == 'right':
            new_msg.twist.angular.z = -0.8
            
        elif command == 'stop':
            new_msg.twist.linear.x = 0.0
            new_msg.twist.angular.z = 0.0
            # stop은 즉시 멈춰야 하므로 duration을 짧게 주거나 즉시 만료시킴
            # 여기선 0.0으로 둬서 다음 루프 때 바로 stop_robot() 호출되게 함
            self.action_end_time = 0.0 
            self.is_moving = True
            self.current_twist = new_msg
            return 
            
        else:
            return

        # 상태 업데이트
        self.current_twist = new_msg
        
        # [중요] 종료 시간 = 현재시간 + 0.8초
        now_sec = self.get_clock().now().nanoseconds / 1e9
        self.action_end_time = now_sec + self.CMD_DURATION
        self.is_moving = True

    def control_loop(self):
        if not self.is_moving:
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9
        
        if now_sec < self.action_end_time:
            # 아직 유효 시간 안쪽이면 계속 명령 전송 (Keep Alive)
            self.current_twist.header.stamp = self.get_clock().now().to_msg()
            self.publisher_.publish(self.current_twist)
        else:
            # 유효 시간(0.8초) 동안 새 명령이 안 왔다? -> 비상 정지
            self.stop_robot()
            self.is_moving = False

    def stop_robot(self):
        stop_msg = TwistStamped()
        stop_msg.header.frame_id = 'base_link'
        stop_msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher_.publish(stop_msg)
        self.get_logger().info(">> Timeout/Stop -> Robot Halted.")

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
