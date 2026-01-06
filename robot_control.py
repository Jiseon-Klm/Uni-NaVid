import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import time

class RobotActionController(Node):
    def __init__(self):
        super().__init__('robot_action_controller')
        
        # 1. Subscriber (Just updates state)
        self.subscription = self.create_subscription(
            String,
            'sign',
            self.listener_callback,
            10
        )
        
        # 2. Publisher
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # 3. Control Timer (10Hz - Runs every 0.1s)
        # This replaces the 'for loop' and 'sleep'
        self.timer = self.create_timer(0.1, self.control_loop)
        
        # State Variables
        self.current_twist = Twist()
        self.action_end_time = 0.0 # When the current action should stop
        self.is_moving = False

        self.get_logger().info("Async Controller Ready.")

    def listener_callback(self, msg):
        command = msg.data.lower().strip()
        self.get_logger().info(f"Received: '{command}' (Overriding previous action)")
        
        new_twist = Twist()
        duration = 0.0

        if command == 'forward':
            new_twist.linear.x = 0.2
            duration = 3 # Keep moving for 3s (unless interrupted)
            
        elif command == 'left':
            new_twist.angular.z = 0.5
            duration = 3
            
        elif command == 'right':
            new_twist.angular.z = -0.5
            duration = 3
            
        elif command == 'stop':
            new_twist.linear.x = 0.0
            new_twist.angular.z = 0.0
            duration = 0.0 # Stop immediately
            
        else:
            self.get_logger().warn(f"Unknown: {command}")
            return

        # Update State (Immediate Override)
        self.current_twist = new_twist
        
        # Set End Time (Current Time + Duration)
        now_sec = self.get_clock().now().nanoseconds / 1e9
        self.action_end_time = now_sec + duration
        self.is_moving = True

    def control_loop(self):
        if not self.is_moving:
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9
        
        if now_sec < self.action_end_time:
            # Still within duration -> Publish Velocity
            self.publisher_.publish(self.current_twist)
        else:
            # Time is up -> Stop
            self.stop_robot()
            self.is_moving = False

    def stop_robot(self):
        stop_twist = Twist()
        self.publisher_.publish(stop_twist)
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
        # Avoid "rcl_shutdown already called" if something else already shut down the context.
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()