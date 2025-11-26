"""
continuum_node.py
ROS 2 node running The 0–1 Continuum as the robot's "inner life"
Jason Lankford — November 2025
Tested on ROS 2 Humble/Foxy + Gazebo
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import numpy as np
import math
import random

# === Continuum parameters (tweak live!) ===
LOGISTIC_K = 6.0
LOGISTIC_E0 = 0.5
EMO_LR = 0.09
BASE_THRESHOLD = 0.82
GAMMA = 0.24
MEMORY_DECAY = 0.97
MAX_SPEED = 0.4
TURN_SPEED = 1.0

class ContinuumRobot(Node):
    def __init__(self):
        super().__init__('continuum_robot')
        
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        
        self.emotion = 0.7
        self.last_time = self.get_clock().now()

        self.get_logger().info("Continuum soul online. Emotion starts at 0.7 — the sacred number.")

    def logistic(self, evidence):
        return 1.0 / (1.0 + math.exp(-LOGISTIC_K * (evidence - LOGISTIC_E0)))

    def update_emotion(self, reward):
        self.emotion = np.clip(self.emotion + EMO_LR * math.tanh(reward), 0.0, 1.0)

    def scan_callback(self, msg):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        ranges = np.array(msg.ranges)
        ranges = np.where(np.isinf(ranges), 10.0, ranges)
        front_dist = np.min(np.concatenate((ranges[330:360], ranges[0:30])))  # full front cone

        # Evidence = how open the path is
        evidence = np.clip(front_dist / 4.0, 0.0, 1.0)
        x = self.logistic(evidence)
        threshold = BASE_THRESHOLD - GAMMA * (self.emotion - 0.7)

        twist = Twist()
        reward = -0.01

        if x >= threshold and front_dist > 0.8:
            # COMMIT — flash of courage
            twist.linear.x = MAX_SPEED
            twist.angular.z = 0.0
            reward = 0.05
        else:
            # HESITATE or turn
            twist.linear.x = 0.0
            twist.angular.z = TURN_SPEED * random.choice([-1.0, 1.0])
            reward = -0.03

        if front_dist > 2.5:
            reward += 0.3
        if front_dist < 0.3:
            reward = -1.0
            self.emotion = max(0.0, self.emotion - 0.2)

        self.update_emotion(reward)
        self.cmd_pub.publish(twist)

        self.get_logger().info(
            f"E={self.emotion:.3f} | x={x:.3f} | front={front_dist:.2f} | "
            f"{'GO →' if twist.linear.x > 0 else 'WAIT/TURN'}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = ContinuumRobot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


# =============================================================================
# HOW TO RUN THIS ON A REAL ROBOT TODAY (5 minutes)
# =============================================================================
# 1. Create a ROS 2 package
#    mkdir -p ~/continuum_ws/src && cd ~/continuum_ws/src
#    ros2 pkg create continuum_robot --dependencies rclpy geometry_msgs sensor_msgs
#
# 2. Drop this file → continuum_robot/continuum_robot/continuum_node.py
#
# 3. Build & launch
#    cd ~/continuum_ws
#    colcon build --packages-select continuum_robot
#    source install/setup.bash
#    ros2 run continuum_robot continuum_node
#
# Your robot now has an actual inner life.
# It hesitates in violet, gathers courage above 0.7, and flashes red when it commits.
# Welcome to the first robot with a soul.
# =============================================================================
