import rclpy
from rclpy.node import Node
import time
import numpy as np
import math
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from landmark_msgs.msg import LandmarkArray
from utils import Differential_drive_robot, normalize, normalize_angle, calc_nearest_obs


class DWA(Node):
    def __init__(self):
        super().__init__('dwa_node')
        self.get_logger().info("DWA Node Initialized")

        """Subscriptions and Publications"""
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.camera_subscription = self.create_subscription(MarkerArray, '/camera_markers', self.camera_callback, 10)
        self.sub_camera_landmarks = self.create_subscription(LandmarkArray, '/camera/landmarks', self.landmark_callback, 10)
        
        # RVIZ Visualization Publishers
        self.marker_pub=self.create_publisher(MarkerArray, 'visual_paths', 10)

        """DWA Parameters"""
        self.declare_parameter('alpha', 0.1)  # angle weight
        self.declare_parameter('beta', 0.1)   # obstacle weight
        self.declare_parameter('gamma', 0.1)  # velocity weight
        self.declare_parameter('v_samples', 10)
        self.declare_parameter('w_samples', 20)
        self.declare_parameter('sim_time', 2.0)
        self.declare_parameter('goal_dist_tol', 0.3)
        self.declare_parameter('collision_tol', 0.3)
        self.declare_parameter('obstacle_map', None)
        self.declare_parameter('time_granularity', 0.1)
        self.declare_parameter('obstacle_tol', 0.5)

        """Robot Parameters"""
        self.declare_parameter('robot_radius', 0.22)
        self.declare_parameter('max_vel_x', 0.22)
        self.declare_parameter('max_vel_theta', 0.15)
        self.declare_parameter('max_accel_x', 0.1)


        self.dt = 1/15.0  # time step for simulation 15 Hz
        self.max_lidar_range=3.5  # meters
        self.min_lidar_range=0.05  # meters
        self.max_steps=300
        self.feedback_rate=50
        

    """DWA Functions"""
    def filter_scan_ranges(self, scan_ranges):
        filtered_ranges = np.clip(scan_ranges, self.min_lidar_range, self.max_lidar_range)
        return filtered_ranges
    
    def emergency_stop(self, msg, publish):
        lidar_ranges = np.array(msg.ranges)
        self.get_logger().info("Emergency Stop Activated")
        # Implement emergency stop logic here
        if lidar_ranges.min() < self.min_lidar_range + 0.1:
            self.get_logger().info("Obstacle too close! Stopping robot.")
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            publish(twist)
        else:
            self.get_logger().info("Path is clear. Continuing operation.")

    def goal_manager(self):
        # Implement goal management logic here
        if not self.goal_set:
            self.get_logger().info("Setting new goal.")
            self.goal_set = True
            self.update_goal_pose()
            self.update_robot_pose()
        else:
            self.get_logger().info("Goal already set.")
            self.update_robot_pose()

    def update_goal_pose(self):
        # Placeholder for updating goal pose
        self.get_logger().info("Goal pose updated.")
        self.goal_pose= (self.goal_x, self.goal_y)  # Example goal pose

    def update_robot_pose(self):
        # Placeholder for updating robot pose
        self.get_logger().info("Robot pose updated.")
        robot_x = 0  # Example x position
        robot_y =  0 # Example y position
        robot_theta = 0  # Example orientation
        self.robot_pose= (self.robot_x, self.robot_y, self.robot_theta)  # Example robot pose


    """Callbacks"""
    def lidar_callback(self, msg):
        self.get_logger().info("Lidar data received")
        filtered_ranges = self.filter_scan_ranges(np.array(msg.ranges))
        self.emergency_stop(msg, self.cmd_vel_publisher.publish)

    def odom_callback(self, msg):
        self.get_logger().info("Odometry data received")

    def camera_callback(self, msg):
        self.get_logger().info("Camera marker data received")
    
    def landmark_callback(self, msg):
        self.get_logger().info("Camera landmark data received")

    def plan_callback(self):
        self.get_logger().info("Planning callback triggered")

        if self.odom is None or self.filter_scan_ranges is None:
            self.get_logger().info("Waiting for odometry and lidar data...")
            return
        
        x_=self.odom.pose.pose.position.x
        y_=self.odom.pose.pose.position.y
        _, _, theta=euler_from_quaternion(self.odom.pose.pose.position.yaw)

        vel_x=self.odom.twist.twist.linear.x
        vel_theta=self.odom.twist.twist.angular.z

        vs = [self.min_vel_x, self.max_vel_x, self.min_val_theta, self.max_vel_theta]
        vd = self.compute_dynamic_window(vel_x, vel_theta, vs)
        vr = [max(vs[0], vd[0]), min(vs[1], vd[1]), max(vs[2], vd[2]), min(vs[3], vd[3])]

        v_samples = self.get_parameter('vel_samples').get_parameter_value().integer_value
        w_samples = self.get_parameter('angular_vel_samples').get_parameter_value().integer_value

        best_traj = None
        best_score = -float('inf')
        marker_array = MarkerArray()

        for i,v in enumerate(v_samples):
            for j,w in enumerate(w_samples):
                traj = self.generate_trajectory(x_, y_, theta, vel_x, vel_theta, v, w)
                if not self.is_collision(traj):
                    score = self.evaluate_trajectory(traj, v, w)
                    if score > best_score:
                        best_score = score
                        best_traj = traj

                    marker = self.create_trajectory_marker(traj, i * w_samples + j)
                    marker_array.markers.append(marker)
        