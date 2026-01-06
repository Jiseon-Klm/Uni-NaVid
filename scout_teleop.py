import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import TwistStamped
import sys, select, termios, tty
import time

# ==========================================
# ⚙️ 설정값
# ==========================================
LINEAR_SPEED = 0.4   # m/s
ANGULAR_SPEED = 0.8  # rad/s

# 반응 속도 튜닝
# 입력 감지 주기 (초): 짧을수록 반응이 빠름 (0.02s = 50Hz)
POLLING_RATE = 0.02  

# 키 입력 유지 시간 (초): 
# 키를 떼도 아주 잠깐 명령을 유지해서 부드럽게 주행 (0.15초 추천)
KEY_PERSISTENCE = 0.15 
# ==========================================

msg = """
=============================================
      🚀 SCOUT MINI TELEOP CONTROL
=============================================
    [W]       Forward
 [A][S][D]    Left / Back / Right

  SPACE       Emergency Stop
  CTRL-C      Quit
=============================================
waiting for input...
"""

class TeleopNode(Node):
    def __init__(self):
        super().__init__('scout_teleop_node')
        
        # 1. QoS 설정 (건드리지 않음: Best Effort)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        # 2. Publisher 설정 (건드리지 않음: TwistStamped)
        self.publisher_ = self.create_publisher(
            TwistStamped, 
            '/scout_mini_base_controller/cmd_vel', 
            qos_profile
        )
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
    # select 타임아웃을 POLLING_RATE로 설정해서 반응속도 높임
    rlist, _, _ = select.select([sys.stdin], [], [], POLLING_RATE)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def print_status(status, lin, ang):
    # 화면을 지우지 않고 한 줄에서 갱신 (\r 사용)
    # f-string으로 자릿수를 고정해서 글자가 깨지지 않게 함
    print(f"Status: {status:<10} | Lin: {lin:>5.2f} m/s | Ang: {ang:>5.2f} rad/s      ", end='\r')

def main():
    settings = termios.tcgetattr(sys.stdin)
    rclpy.init()
    
    node = TeleopNode()
    
    # 상태 변수
    target_linear = 0.0
    target_angular = 0.0
    last_key_time = 0.0 # 마지막으로 키를 누른 시간
    status_msg = "IDLE"

    try:
        while True:
            key = get_key(settings)
            current_time = time.time()
            
            # 1. 키 입력 처리
            if key in ['w', 's', 'a', 'd', ' ']:
                last_key_time = current_time # 키 누른 시간 갱신
                
                if key == 'w':
                    target_linear = LINEAR_SPEED
                    target_angular = 0.0
                    status_msg = "FORWARD ⬆️"
                elif key == 's':
                    target_linear = -LINEAR_SPEED
                    target_angular = 0.0
                    status_msg = "BACKWARD ⬇️"
                elif key == 'a':
                    target_linear = 0.0
                    target_angular = ANGULAR_SPEED
                    status_msg = "LEFT ⬅️"
                elif key == 'd':
                    target_linear = 0.0
                    target_angular = -ANGULAR_SPEED
                    status_msg = "RIGHT ➡️"
                elif key == ' ':
                    target_linear = 0.0
                    target_angular = 0.0
                    status_msg = "STOP 🛑"
            
            elif key == '\x03': # Ctrl-C
                break

            # 2. 로직 처리 (데드맨 스위치 + 잔상 효과)
            # 키를 누른지 얼마 안 됐으면(Persistence 시간 내) -> 속도 유지
            if (current_time - last_key_time) < KEY_PERSISTENCE:
                pass # 값 유지
            else:
                # 시간이 지났으면 -> 정지
                target_linear = 0.0
                target_angular = 0.0
                status_msg = "IDLE ⏸️"

            # 3. 명령 전송
            node.send_velocity(target_linear, target_angular)
            
            # 4. UI 출력 (깔끔하게 한 줄 갱신)
            print_status(status_msg, target_linear, target_angular)

    except Exception as e:
        print(f"\nError: {e}")

    finally:
        # 종료 시 확실하게 정지
        node.send_velocity(0.0, 0.0)
        print("\n\n🛑 Teleop Closed. Robot Stopped.")
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
   
