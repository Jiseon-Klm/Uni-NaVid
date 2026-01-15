import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import TwistStamped
import sys, select, termios, tty
import time
import sys
import shutil
# ==========================================
# ⚙️ 설정값
# ==========================================
LINEAR_SPEED = 0.6   # m/s
ANGULAR_SPEED = 0.6  # rad/s

# 반응 속도 튜닝
# 입력 감지 주기 (초): 짧을수록 반응이 빠름 (0.02s = 50Hz)
POLLING_RATE = 0.02  

# 키 입력 유지 시간 (초): 
# 키를 떼도 아주 잠깐 명령을 유지해서 부드럽게 주행 (0.15초 추천)
KEY_PERSISTENCE = 0.1
# ==========================================

msg = """
=============================================
      🚀 SCOUT MINI TELEOP CONTROL
=============================================
    [W]       Forward
 [A][S][D]    Left / Back / Right
 [W]+[A]      Forward + Left (동시 입력)
 [W]+[D]      Forward + Right (동시 입력)

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
    """키 입력을 받아서 반환 (비어있으면 None)"""
    tty.setraw(sys.stdin.fileno())
    # select 타임아웃을 POLLING_RATE로 설정해서 반응속도 높임
    rlist, _, _ = select.select([sys.stdin], [], [], POLLING_RATE)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = None
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def print_status(status, lin, ang):
    # 터미널 폭에 맞춰 줄바꿈(wrap) 방지
    cols = shutil.get_terminal_size((80, 20)).columns

    s = f"Status: {status:<10} | Lin: {lin:>5.2f} m/s | Ang: {ang:>5.2f} rad/s"
    # 너무 길면 잘라서 wrap 자체를 못 하게 막기
    if len(s) > cols - 1:
        s = s[:cols - 1]

    # \r: 줄 맨 앞으로, \033[2K: 현재 줄 전체 삭제
    sys.stdout.write("\r\033[2K" + s)
    sys.stdout.flush()

def main():
    settings = termios.tcgetattr(sys.stdin)
    rclpy.init()
    
    node = TeleopNode()
    
    # 상태 변수
    target_linear = 0.0
    target_angular = 0.0
    pressed_keys = set()  # 현재 눌린 키들을 추적하는 집합
    last_key_time = {}  # 각 키별로 마지막으로 누른 시간
    status_msg = "IDLE"

    try:
        while True:
            key = get_key(settings)
            current_time = time.time()
            
            # 1. 키 입력 처리 (키 상태 추적)
            if key == '\x03':  # Ctrl-C
                break
            elif key == '\x1b':  # ESC 키 (키를 떼는 신호로 사용)
                # ESC는 무시하거나 특별 처리
                pass
            elif key is not None:
                # 키가 입력되었을 때
                if key in ['w', 's', 'a', 'd', ' ']:
                    if key == ' ':  # 스페이스는 즉시 정지
                        pressed_keys.clear()
                        target_linear = 0.0
                        target_angular = 0.0
                        status_msg = "STOP 🛑"
                        last_key_time.clear()
                    else:
                        # 키를 누름 (집합에 추가)
                        pressed_keys.add(key)
                        last_key_time[key] = current_time
            
            # 2. 키 상태 업데이트 (떼어진 키 제거)
            # 키 입력이 없으면 (key is None) 모든 키의 마지막 입력 시간 확인
            # KEY_PERSISTENCE 시간이 지나면 키를 제거 (키를 떼었다고 간주)
            keys_to_remove = []
            for k in list(pressed_keys):  # 리스트로 복사해서 순회 (집합 변경 방지)
                if k in last_key_time:
                    # 키를 누른지 KEY_PERSISTENCE 시간이 지났으면 제거
                    if (current_time - last_key_time[k]) >= KEY_PERSISTENCE:
                        keys_to_remove.append(k)
            for k in keys_to_remove:
                pressed_keys.discard(k)
                if k in last_key_time:
                    del last_key_time[k]
            
            # 3. 눌린 키 조합에 따라 속도 계산
            # 우선순위: 조합 > 단일 키
            if 'w' in pressed_keys and 'a' in pressed_keys:
                # 좌회전하면서 직진 (같은 시간, 같은 각속도/선속도)
                target_linear = LINEAR_SPEED
                target_angular = ANGULAR_SPEED
                status_msg = "FORWARD+LEFT ↗️"
            elif 'w' in pressed_keys and 'd' in pressed_keys:
                # 우회전하면서 직진 (같은 시간, 같은 각속도/선속도)
                target_linear = LINEAR_SPEED
                target_angular = -ANGULAR_SPEED
                status_msg = "FORWARD+RIGHT ↗️"
            elif 'w' in pressed_keys:
                target_linear = LINEAR_SPEED
                target_angular = 0.0
                status_msg = "FORWARD ⬆️"
            elif 's' in pressed_keys:
                target_linear = -LINEAR_SPEED
                target_angular = 0.0
                status_msg = "BACKWARD ⬇️"
            elif 'a' in pressed_keys:
                target_linear = 0.0
                target_angular = ANGULAR_SPEED
                status_msg = "LEFT ⬅️"
            elif 'd' in pressed_keys:
                target_linear = 0.0
                target_angular = -ANGULAR_SPEED
                status_msg = "RIGHT ➡️"
            elif len(pressed_keys) == 0:
                # 모든 키가 떼어졌으면 정지
                target_linear = 0.0
                target_angular = 0.0
                status_msg = "IDLE ⏸️"

            # 4. 명령 전송
            node.send_velocity(target_linear, target_angular)
            
            # 5. UI 출력 (깔끔하게 한 줄 갱신)
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
   
