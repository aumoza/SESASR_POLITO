#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
import numpy as np
from scipy.spatial.transform import Rotation as R

class Controller_Burger(Node):
    def __init__(self):
        super().__init__('controller_burger')
        self.get_logger().info('Controller_Burger node started.')

        # Publishers & subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.lidar_callback, qos_profile_sensor_data)
        self.sub_gt = self.create_subscription(Odometry, '/ground_truth', self.ground_truth_callback, 10)

        # Parameters (declare + read)
        self.declare_parameter('max_angular', 1.0)
        self.declare_parameter('max_vel', 0.1)
        self.declare_parameter('control_frequency', 10.0)
        self.declare_parameter('obstacle_distance_threshold', 0.35)  # meters

        self.max_angular_velocity = float(self.get_parameter('max_angular').get_parameter_value().double_value)
        self.max_linear_velocity = float(self.get_parameter('max_vel').get_parameter_value().double_value)
        self.control_frequency = float(self.get_parameter('control_frequency').get_parameter_value().double_value)
        self.obstacle_distance_threshold = float(self.get_parameter('obstacle_distance_threshold').get_parameter_value().double_value)

        # State variables (initialized)
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.ground_truth_x = None
        self.ground_truth_y = None
        self.ground_truth_yaw = None

        self.lidar_data = None
        self.range_min = None
        self.range_max = None
        self.left_min = float('inf')
        self.right_min = float('inf')
        self.close_to_wall = False

        # Timer to run control loop (movement) at configured frequency
        period = 1.0 / max(1e-3, self.control_frequency)
        self.create_timer(period, self.movement)

        self.get_logger().info(f'Controller initialized: max_vel={self.max_linear_velocity}, max_ang={self.max_angular_velocity}, control_freq={self.control_frequency} Hz')

    # -----------------------
    # Callbacks
    # -----------------------
    def odom_callback(self, msg: Odometry):
        # store pose
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = [msg.pose.pose.orientation.x,
             msg.pose.pose.orientation.y,
             msg.pose.pose.orientation.z,
             msg.pose.pose.orientation.w]
        try:
            # as_euler('xyz')[2] => yaw
            self.yaw = R.from_quat(q).as_euler('xyz', degrees=False)[2]
        except Exception as e:
            self.get_logger().warning(f'Failed to convert quaternion to euler: {e}')
            self.yaw = 0.0

        # light logging at debug level to avoid flooding console
        self.get_logger().debug(f"Odom -> x:{self.x:.2f}, y:{self.y:.2f}, yaw:{np.degrees(self.yaw):.1f}°")

    def lidar_callback(self, msg: LaserScan):
        # store raw scan (but clean it when used)
        self.lidar_data = list(msg.ranges)
        self.range_min = msg.range_min
        self.range_max = msg.range_max

    def ground_truth_callback(self, msg: Odometry):
        pose = msg.pose.pose
        self.ground_truth_x = pose.position.x
        self.ground_truth_y = pose.position.y
        q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        try:
            self.ground_truth_yaw = R.from_quat(q).as_euler('xyz', degrees=False)[2]
        except Exception:
            self.ground_truth_yaw = 0.0

        self.get_logger().info(
            f"Ground Truth -> x:{self.ground_truth_x:.2f}, y:{self.ground_truth_y:.2f}, yaw:{np.degrees(self.ground_truth_yaw):.1f}°"
        )

    # -----------------------
    # Movement helpers
    # -----------------------
    def check_obstacle(self):
        """Process the latest lidar_data and update left_min/right_min/front_min flags safely."""
        if not self.lidar_data:
            return

        # Clean the scan: replace inf/nan with large value (range_max), clamp into [range_min, range_max]
        cleaned = []
        for r in self.lidar_data:
            try:
                if r is None:
                    cleaned.append(self.range_max if self.range_max is not None else 10.0)
                elif np.isinf(r) or np.isnan(r):
                    cleaned.append(self.range_max if self.range_max is not None else 10.0)
                else:
                    # clamp
                    if self.range_min is not None:
                        r = max(r, self.range_min)
                    if self.range_max is not None:
                        r = min(r, self.range_max)
                    cleaned.append(r)
            except Exception:
                cleaned.append(self.range_max if self.range_max is not None else 10.0)

        n = len(cleaned)
        if n == 0:
            return

        # safe index ranges for front/left/right sectors
        # front: first and last 10 measurements (wrap-around)
        front_idx = 10 if n >= 20 else max(1, n // 10)
        front = cleaned[:front_idx] + cleaned[-front_idx:]
        left_start = int(n * 0.25)
        left_end = int(n * 0.30)
        right_start = int(n * 0.70)
        right_end = int(n * 0.75)

        left = cleaned[left_start:left_end] if left_end > left_start else cleaned[left_start:left_start+1]
        right = cleaned[right_start:right_end] if right_end > right_start else cleaned[right_start:right_start+1]

        # compute minima (use large default if empty)
        front_min = min(front) if front else float('inf')
        self.left_min = min(left) if left else float('inf')
        self.right_min = min(right) if right else float('inf')

        # obstacle check
        if front_min < self.obstacle_distance_threshold:
            self.close_to_wall = True
            self.get_logger().debug("Obstacle detected in front")
        else:
            self.close_to_wall = False
            self.get_logger().debug("No obstacle in front")

    def movement(self):
        """Main control loop executed periodically by timer."""
        # update lidar-based obstacle detections if data present
        if self.lidar_data is not None:
            self.check_obstacle()

        # decide motion
        if getattr(self, 'close_to_wall', False):
            # choose turning direction based on which side is closer
            if (self.left_min < self.right_min) or (self.left_min < self.obstacle_distance_threshold):
                self.turn_right()
            elif (self.right_min < self.left_min) or (self.right_min < self.obstacle_distance_threshold):
                self.turn_left()
            else:
                # fallback: rotate in place to search
                self.turn_left()
        else:
            self.move_straight()

    def move_straight(self):
        twist = Twist()
        twist.linear.x = float(self.max_linear_velocity)
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
        self.get_logger().debug(f"Moving straight: {self.max_linear_velocity:.2f} m/s")

    def turn_left(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = float(self.max_angular_velocity)
        self.cmd_pub.publish(twist)
        self.get_logger().info(f"Turning left: {self.max_angular_velocity:.2f} rad/s")

    def turn_right(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = -float(self.max_angular_velocity)
        self.cmd_pub.publish(twist)
        self.get_logger().info(f"Turning right: {-self.max_angular_velocity:.2f} rad/s")


def main(args=None):
    rclpy.init(args=args)
    controller = Controller_Burger()
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
