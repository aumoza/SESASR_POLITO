import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Bool
import time
import tf_transformations
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from rclpy.qos import qos_profile_sensor_data

class Controller_Burger(Node):
    def __init__(self):
        # Initialize the node
        super().__init__('controller_burger')
        self.get_logger().info('Controller_burger node has been started.')
        
        # Create publishers and subscribers
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscriber_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.subscriber_lidar = self.create_subscription(LaserScan, '/scan', self.lidar_callback, qos_profile_sensor_data) # implemented qus_profile_sensor_data for LIDAR
        self.subscriber_gt = self.create_subscription(Odometry, '/ground_truth', self.ground_truth_callback, 10)
        self.subscriber_ps = self.create_subscription(PoseStamped, '/move_base_simple/goal', self.goal_pose_callback, 10)
       
       # Declare parameters (with default values)
        self.declare_parameter('max_angular', 1.5)
        self.declare_parameter('max_vel', 0.22)
        self.declare_parameter('control_frequency', 10.0)
        self.max_angular_velocity = self.get_parameter('max_angular').get_parameter_value().double_value
        self.max_linear_velocity = self.get_parameter('max_vel').get_parameter_value().double_value
        self.control_frequency = self.get_parameter('control_frequency').get_parameter_value().double_value

        # Internal state variables
        self.timer = self.create_timer(1.0 / self.control_frequency, self.movement)
        self.close_to_wall = False
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.lidar_data = []
        self.ground_truth_x = 0.0
        self.ground_truth_y = 0.0
        self.at_goal_flag = False
   
    # define methods
    # LIDAR callback to process incoming laser scan data
    def lidar_callback(self, msg: LaserScan):
        self.lidar_data = msg.ranges
        self.range_min = msg.range_min
        self.range_max = msg.range_max

    # Check for obstacles using LIDAR data
    def check_obstacle(self):
        if self.lidar_data is None or len(self.lidar_data) == 0:
            return

        elif len(self.lidar_data) >= 280:
            front_ranges = self.lidar_data[0:10] + self.lidar_data[-10:]
            left_ranges = self.lidar_data[80:100]
            right_ranges = self.lidar_data[260:280]
            front_min = min(front_ranges)
            self.left_min = min(left_ranges)
            self.right_min = min(right_ranges)
        else:
            front_ranges = self.lidar_data[0:5] + self.lidar_data[-5:]
            left_ranges = self.lidar_data[40:60]
            right_ranges = self.lidar_data[-60:-40]
            front_min = min(front_ranges)
            self.left_min = min(left_ranges)
            self.right_min = min(right_ranges)

        if front_min < 0.5:
            self.close_to_wall = True
            print("Obstacle detected in front!")
        else:
            self.close_to_wall = False
            print("No obstacle in front.")

    # Movement logic based on obstacle detection
    def movement(self):
        if hasattr(self, 'lidar_data'):
            self.check_obstacle()

        self.at_goal() # Check if the robot has reached its goal
        if not self.at_goal_flag:
            if self.close_to_wall:
                if self.left_min < self.right_min:
                    self.turn_right()
                else:
                    self.turn_left()
            else:
                self.move_straight()
        else:
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.publisher_.publish(twist)
            self.get_logger().info("Robot has reached the goal and stopped.")
            time.sleep(10)
            self.destroy_node()
            rclpy.shutdown()

    # Move straight method
    def move_straight(self):
        twist = Twist()
        twist.linear.x = self.max_linear_velocity
        self.centring()
        twist.angular.z = 0.0
        self.publisher_.publish(twist)
        self.get_logger().info(f"Moving straight with linear velocity {self.max_linear_velocity} m/s")

    # Turn LEFT method
    def turn_left(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = self.max_angular_velocity
        self.publisher_.publish(twist)
        self.get_logger().info(f"Turning with angular velocity {self.max_angular_velocity} rad/s")

    # Turn RIGHT method
    def turn_right(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = -self.max_angular_velocity
        self.publisher_.publish(twist)
        self.get_logger().info(f"Turning with angular velocity {-self.max_angular_velocity} rad/s")

    # Check if the robot has reached its goal
    def at_goal(self):
        # Implement logic to check if the robot has reached its goal
        goal_x = 6.5  # Example goal x-coordinate
        goal_y = 3.0  # Example goal y-coordinate
        threshold = 0.2  # Distance threshold to consider as "at goal"
        if (abs(self.ground_truth_x - goal_x) < threshold) and (abs(self.ground_truth_y - goal_y) < threshold):
            self.get_logger().info("Goal reached!")
            self.at_goal_flag = True
            
        self.get_logger().info("Goal not reached yet.")
        self.at_goal_flag = False
    
    # Odometry callback to update robot's position and orientation
    def odom_callback(self, msg: Odometry):
        # Extract robot pose
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        # Orientation is a quaternion — convert to yaw
        orientation_q = msg.pose.pose.orientation
        (_, _, yaw) = tf_transformations.euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])
        self.yaw = yaw
        # Optional: log for debugging
        self.get_logger().info(f"Odometry -> x: {self.x:.2f}, y: {self.y:.2f}, yaw: {np.degrees(self.yaw):.1f}°")
    
    # Ground truth callback to get accurate position from Gazebo
    def ground_truth_callback(self, msg: Odometry):
        pose = msg.pose.pose
        self.ground_truth_x = pose.position.x
        self.ground_truth_y = pose.position.y
        orientation_q = pose.orientation
        (_, _, yaw) = tf_transformations.euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z ,orientation_q.w])
        self.get_logger().info(f"Ground Truth → x: {self.ground_truth_x:.2f}, y: {self.ground_truth_y:.2f}, yaw: {np.degrees(yaw):.1f}°")

    # Goal pose callback to receive goal position
    def goal_pose_callback(self, msg: PoseStamped):
        if msg is None:
            print("No goal pose received.")
        else:
            goal_pose = msg.pose
            self.get_logger().info(f"Received new goal pose: x={goal_pose.position.x}, y={goal_pose.position.y}")
            print(f"Received new goal pose: x={goal_pose.position.x}, y={goal_pose.position.y}")
        # Implement logic to set the goal position
    

def main(args=None):
    rclpy.init(args=args)
    controller_burger = Controller_Burger()
    rclpy.spin(controller_burger)
    controller_burger.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()