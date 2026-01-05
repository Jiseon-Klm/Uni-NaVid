import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import TwistStamped
import sys, select, termios, tty

# 설정값
LINEAR_SPEED = 0.4  # 전진 속도
ANGULAR_SPEED = 0.8 # 회전 속도

msg = """
---------------------------
🎮 Scout Mini Teleop (Deadman Switch)
---------------------------
   w : 전진 (누르고 있는 동안만)
   s : 후진
   a : 좌회전
   d : 우회전

   CTRL-C : 종료
---------------------------
"""

class TeleopNode(Node):
    def __init__(self):
        super().__init__('scout_teleop_node')
        
        # QoS 설정 (Best Effort 필수!)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        self.publisher_ = self.create_publisher(
            TwistStamped, 
            '/scout_mini_base_controller/cmd_vel', 
            qos_profile
        )
        self.print_manual()

    def print_manual(self):
        print(msg)

    def send_velocity(self, linear, angular):
        twist = TwistStamped()
        twist.header.frame_id = 'base_link'
        twist.header.stamp = self.get_clock().now().to_msg()
        
        twist.twist.linear.x = float(linear)
        twist.twist.angular.z = float(angular)
        
        self.publisher_.publish(twist)

def get_key(settings):
    tty.setraw(sys.stdin.fileno())
    # 0.1초 동안 키 입력을 기다림
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def main():
    settings = termios.tcgetattr(sys.stdin)
    rclpy.init()
    
    node = TeleopNode()
    
    try:
        while True:
            # 1. 루프 시작할 때마다 속도를 0으로 초기화 (이게 핵심!)
            target_linear = 0.0
            target_angular = 0.0
            
            # 2. 키 입력 확인
            key = get_key(settings)
            
            if key == 'w':
                target_linear = LINEAR_SPEED
                print("⬆️", end='\r') # 상태 표시
            elif key == 's':
                target_linear = -LINEAR_SPEED
                print("⬇️", end='\r')
            elif key == 'a':
                target_angular = ANGULAR_SPEED
                print("⬅️", end='\r')
            elif key == 'd':
                target_angular = -ANGULAR_SPEED
                print("➡️", end='\r')
            elif key == '\x03': # Ctrl-C
                break
            
            # 키를 아무것도 안 눌렀으면 target 변수는 0인 상태 그대로 내려옴.
            
            # 3. 결정된 속도(이동 혹은 0)를 로봇에게 전송
            node.send_velocity(target_linear, target_angular)

    except Exception as e:
        print(e)

    finally:
        node.send_velocity(0.0, 0.0)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
