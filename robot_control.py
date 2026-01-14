import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy  # 1. 이거 추가!
from std_msgs.msg import String
from geometry_msgs.msg import TwistStamped
import time

class RobotActionController(Node):
    def __init__(self):
        super().__init__('robot_action_controller')
        
        # ==================================================
        # [수정] 1. Subscriber QoS 설정 변경
        # 누가 보내든 다 받기 위해 Best Effort로 설정
        # ==================================================
        qos_profile_subscriber = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        self.subscription = self.create_subscription(
            String,
            'sign',
            self.listener_callback,
            qos_profile_subscriber  # <-- 숫자 10 대신 이걸 넣으세요!
        )
        
        # 2. Publisher (핵심 수정!)
        # 로봇(Scout Mini)이 Best Effort만 받아주므로, 우리도 똑같이 맞춰야 함
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        self.publisher_ = self.create_publisher(
            TwistStamped, 
            '/scout_mini_base_controller/cmd_vel', 
            qos_profile  # 숫자 10 대신 qos_profile 객체 사용
        )
        
        # 3. Control Timer (0.1s = 10Hz)
        # 로직 제어용으로는 10Hz도 충분해
        self.timer = self.create_timer(0.1, self.control_loop)
        
        # State Variables
        self.current_twist = TwistStamped()
        self.current_twist.header.frame_id = 'base_link'
        
        self.action_end_time = 0.0
        self.is_moving = False

        self.get_logger().info("Scout Mini Logic Controller Ready (Best Effort Mode). Waiting for /sign...")

    def listener_callback(self, msg):
        command = msg.data.lower().strip()
        self.get_logger().info(f"Received: '{command}'")
        
        new_msg = TwistStamped()
        new_msg.header.frame_id = 'base_link'
        
        # ==================================================
        # [수정] 0.5초 동안 동작하도록 설정
        # ==================================================
        MOVER_DURATION = 0.5  # 0.5초로 변경
        duration = 0.0

        if command in ['straight', 'forward']:
            # 목표: 0.5초 동안 25cm(0.25m) 이동
            # 속도 = 거리 / 시간 = 0.25m / 0.5s = 0.5 m/s
            new_msg.twist.linear.x = 0.5  
            duration = MOVER_DURATION
            
        elif command == 'left':
            # 목표: 0.5초 동안 30도 회전
            target_angle = math.radians(30) 
            # 각속도 = 목표각도 / 시간 = rad(30) / 0.5s
            new_msg.twist.angular.z = target_angle / MOVER_DURATION
            duration = MOVER_DURATION
            
        elif command == 'right':
            # 목표: 0.5초 동안 30도 회전 (우회전은 -)
            target_angle = math.radians(30)
            new_msg.twist.angular.z = -(target_angle / MOVER_DURATION)
            duration = MOVER_DURATION
            
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
        # 움직이는 중이 아니면 아무것도 안 함
        if not self.is_moving:
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9
        
        if now_sec < self.action_end_time:
            # 시간 갱신 (중요: Stamped 메시지는 시간이 흘러가야 함)
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
