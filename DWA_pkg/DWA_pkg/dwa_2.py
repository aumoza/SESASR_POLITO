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
from DWA_pkg.utils import Differential_drive_robot, normalize, normalize_angle, calc_nearest_obs
import os
from ament_index_python.packages import get_package_share_directory
import yaml
import tf_transformations

class DWA(Node):
    def __init__(self, **kwargs):
        super().__init__('dwa_node')
        self.get_logger().info("DWA Node Initialized")

        """Subscriptions and Publications"""
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.odom_subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.camera_subscription = self.create_subscription(MarkerArray, '/camera_markers', self.camera_callback, 10)
        self.sub_camera_landmarks = self.create_subscription(LandmarkArray, '/camera/landmarks', self.landmark_callback, 10)
        self.sub_goal_pose=self.create_subscription(Odometry, '/dynamic_goal_pose', self.goal_pose_callback, 10)

        # RVIZ Visualization Publishers
        self.marker_pub=self.create_publisher(MarkerArray, 'visual_paths', 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        """DWA Parameters"""
        self.alpha=self.declare_parameter('alpha', 0.1).value  # angle weight
        self.beta=self.declare_parameter('beta', 0.1).value   # obstacle weight
        self.gamma=self.declare_parameter('gamma', 0.1).value  # velocity weight
        self.v_samples=self.declare_parameter('v_samples', 10).value
        self.w_samples=self.declare_parameter('w_samples', 20).value
        self.sim_time=self.declare_parameter('sim_time', 2.0).value
        self.goal_dist_tol=self.declare_parameter('goal_dist_tol', 0.3).value
        self.collision_tol=self.declare_parameter('collision_tol', 0.3).value
        #self.obstacle_map=self.declare_parameter('obstacle_map', None)
        self.time_granularity=self.declare_parameter('time_granularity', 0.1).value
        self.obstacle_tol=self.declare_parameter('obstacle_tol', 0.5).value

        """Robot Parameters"""
        self.robot_radius=self.declare_parameter('robot_radius', 0.22).value
        self.max_vel_x=self.declare_parameter('max_vel_x', 0.22).value
        self.max_vel_theta=self.declare_parameter('max_vel_theta', 0.15).value
        self.max_angular_acc=self.declare_parameter('max_angular_acc', math.pi/2).value


        self.dt = 1.0/15.0  # time step for simulation 15 Hz
        self.max_lidar_range=3.5  # meters
        self.min_lidar_range=0.05  # meters
        self.max_steps=300
        self.feedback_rate=50
        self.robot=Differential_drive_robot(**kwargs)
        self.sim_step=round(self.sim_time/self.time_granularity)
        
        # try to load map (non-fatal)
        try:
            self.obstacle_map = self.get_map()
        except Exception as e:
            self.get_logger().warning(f"Could not load map: {e}")
            self.obstacle_map = None

        self.weight_angle=self.alpha
        self.weight_obs=self.beta
        self.weight_vel=self.gamma

         # internal state
        self.last_scan = None
        self.goal_pose = None    # np.array([x,y,yaw])
        self.dwa_active = False
        self.current_step = 0
        self.dwa_active=False

        # create a periodic timer for the DWA loop (non-blocking)
        self.create_timer(self.dt, self.dwa_timer_cb)


    def go_to_pose(self, goal_pose):
        if goal_pose is None:
            if not hasattr(self, "goal_pose"):
                self.get_logger().warn("No goal pose set yet.")
                return False, []
            goal_pose = self.goal_pose
        else:
            goal_pose = np.array(goal_pose, dtype=float)

        success=False
        dist_to_goal=np.linalg.norm(self.robot.pose[0:2]-goal_pose[0:2])

        print('Initial distance to goal: ', dist_to_goal)
        print('Initial robot pose: ', self.robot.pose)

        steps=1
        for steps in range(1, self.max_steps + 1):
            # 1. Check if Goal reached
            dist_to_goal = np.linalg.norm(self.robot.pose[0:2] - goal_pose[0:2])
            if dist_to_goal < self.goal_dist_tol:
                success = True
                print("Goal reached!")
                break

            # 2. Get new observations for obstacles
            # Use lidar data for obstacles
            obstacles = self.process_lidar_data()

            # 3. Compute command for the robot with DWA controller
            u = self.compute_cmd(goal_pose, self.robot.pose, obstacles)

            # 4. Publish the command
            cmd = Twist()
            cmd.linear.x = u[0]
            cmd.angular.z = u[1]
            self.cmd_vel_publisher.publish(cmd)
            self.get_logger().info(f"Publishing cmd: v={cmd.linear.x:.3f}, w={cmd.angular.z:.3f}")
            self.robot.update_state(u, self.dt)

            # 5. Provide intermediate task feedback
            if steps % self.feedback_rate == 0:
                dist_to_goal = np.linalg.norm(self.robot.pose[0:2] - goal_pose[0:2])
                print("Current distance to goal ", dist_to_goal, " at step ", steps)
                print("Current Robot pose: ", self.robot.pose)

        if steps > self.max_steps:
            print("Timeout! Goal not reached.")

        return success, self.robot.trajectory
    
    def compute_cmd(self, goal_pose, robot_state, obstacles):
        # create path
        paths, velocities = self.get_trajectories(robot_state)

        # eval paths
        opt_idx = self.evaluate_paths(paths, velocities, goal_pose, robot_state, obstacles)
        u = velocities[opt_idx]
        return u
    
    def get_trajectories(self, robot_pose):
        # calculate reachable range of velocity and angular velocity in the dynamic window
        min_lin_vel, max_lin_vel, min_ang_vel, max_ang_vel = self.compute_dynamic_window(self.robot.vel)

        v_values = np.linspace(min_lin_vel, max_lin_vel, int(self.v_samples))
        w_values = np.linspace(min_ang_vel, max_ang_vel, int(self.w_samples))

        # list of all paths and velocities
        n_paths = w_values.shape[0]*v_values.shape[0]
        sim_paths = np.zeros((n_paths, self.sim_step, robot_pose.shape[0]))
        velocities = np.zeros((n_paths, 2))

        # evaluate all velocities and angular velocities combination
        vv, ww = np.meshgrid(v_values, w_values)
        velocities = np.dstack([vv,ww]).reshape(n_paths,2)
        sim_paths=self.simulate_paths(n_paths, robot_pose, velocities)

        return sim_paths, velocities
    
    def simulate_paths(self, n_paths, pose, u):
        sim_paths = np.zeros((n_paths, self.sim_step, pose.shape[0]))
        sim_paths[:, 0] = pose.copy()

        for i in range(1, self.sim_step):
            sim_paths[:, i, 0] = sim_paths[:, i - 1, 0] + u[:, 0] * np.cos(sim_paths[:, i - 1, 2]) * self.dt
            sim_paths[:, i, 1] = sim_paths[:, i - 1, 1] + u[:, 0] * np.sin(sim_paths[:, i - 1, 2]) * self.dt
            sim_paths[:, i, 2] = sim_paths[:, i - 1, 2] + u[:, 1] * self.dt

        return sim_paths
    
    def compute_dynamic_window(self, robot_vel):
        # linear velocity
        min_vel = robot_vel[0] - self.dt * self.robot.max_linear_acc
        max_vel = robot_vel[0] + self.dt * self.robot.max_linear_acc
        # minimum
        if min_vel < self.robot.min_lin_vel:
            min_vel = self.robot.min_lin_vel
        # maximum
        if max_vel > self.robot.max_lin_vel:
            max_vel = self.robot.max_lin_vel

        # angular velocity
        min_ang_vel = robot_vel[1] - self.dt * self.robot.max_ang_acc
        max_ang_vel = robot_vel[1] + self.dt * self.robot.max_ang_acc
        # minimum
        if min_ang_vel < self.robot.min_ang_vel:
            min_ang_vel = self.robot.min_ang_vel
        # maximum
        if max_ang_vel > self.robot.max_ang_vel:
            max_ang_vel = self.robot.max_ang_vel

        return min_vel, max_vel, min_ang_vel, max_ang_vel
    
    def evaluate_paths(self, paths, velocities, goal_pose, robot_pose, obstacles):
        # detect nearest obstacle
        nearest_obs = calc_nearest_obs(robot_pose, obstacles)

        # Compute the scores for the generated path
        # (1) heading_angle and goal distance
        score_heading_angles = self.score_heading_angle(paths, goal_pose)
        # (2) velocity
        score_vel = self.score_vel(velocities, paths, goal_pose)
        # (3) obstacles
        score_obstacles = self.score_obstacles(paths, nearest_obs)

        # Scores Normalization
        score_heading_angles = normalize(score_heading_angles)
        score_vel = normalize(score_vel)
        score_obstacles = normalize(score_obstacles)

        # Weighted Sum
        total_score=(
            self.weight_angle * score_heading_angles +
            self.weight_vel * score_vel +
            self.weight_obs * score_obstacles
        )

        if total_score.size==0 or np.all(np.isnan(total_score)):
            raise Exception("Not possible to find an optimal path!")
        
        opt_idx=np.argmax(total_score)
        return opt_idx

    def score_heading_angle(self, path, goal_pose):
        last_x = path[:, -1, 0]
        last_y = path[:, -1, 1]
        last_th = path[:, -1, 2]

        # calculate angle
        angle_to_goal = np.arctan2(goal_pose[1] - last_y, goal_pose[0] - last_x)

        # calculate score
        score_angle = angle_to_goal - last_th
        score_angle = np.fabs(normalize_angle(score_angle))
        heading_error = normalize_angle(angle_to_goal - last_th)
        score_angle = np.cos(heading_error)
        return score_angle
    
    def score_vel(self, u, path, goal_pose):
        vel = u[:,0] # linear velocity value is the score itself (to maximize)
        goal_xy = goal_pose[0:2]  # use only x,y
        dist_to_goal = np.linalg.norm(path[:, -1, 0:2] - goal_xy, axis=-1)
        score = vel + np.exp(-dist_to_goal / self.goal_dist_tol)
        return score
    
    def score_obstacles(self, path, obstacles):
        score_obstacle = 2.0*np.ones((path.shape[0]))

        for obs in obstacles:
            dx = path[:, :, 0] - obs[0]
            dy = path[:, :, 1] - obs[1]
            dist = np.hypot(dx, dy)

            min_dist = np.min(dist, axis=-1)
            score_obstacle[min_dist < score_obstacle] = min_dist[min_dist < score_obstacle]
        
            # collision with obstacle
            score_obstacle[score_obstacle < self.robot.radius + self.collision_tol] = -100 # heavy penalty for collision
               
        return score_obstacle

    def filter_scan_ranges(self, scan_ranges):
        filtered_ranges = np.clip(scan_ranges, self.min_lidar_range, self.max_lidar_range)
        return filtered_ranges
    
    def stop_robot(self):
        twist=Twist()
        twist.linear.x=0.0
        twist.angular.z=0.0
        self.cmd_vel_publisher.publish
        self.get_logger().info("Goal reached and stopped")

    def emergency_stop(self, msg, publish):
        lidar_ranges = np.array(msg.ranges)
        self.get_logger().info("Emergency Stop Activated")
        # Implement emergency stop logic here
        if lidar_ranges.min() < self.min_lidar_range + 0.3:
            self.get_logger().info("Obstacle too close! Stopping robot.")
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            publish(twist)
        else:
            self.get_logger().info("Path is clear. Continuing operation.")

    def process_lidar_data(self):
        # store latest scan in lidar_callback
        if self.last_scan is None:
            return np.empty((0, 2))
         
        scan = self.last_scan
        ranges = np.array(scan.ranges, dtype=float)
        ranges = self.filter_scan_ranges(ranges)

        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment

        # consider only finite points within max range
        valid = np.isfinite(ranges) & (ranges < self.max_lidar_range)
        ranges = ranges[valid]
        angles = angles[valid]
        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        
        th = self.robot.pose[2]
        Rx = self.robot.pose[0]
        Ry = self.robot.pose[1]

        world_x = Rx + xs*np.cos(th) - ys*np.sin(th)
        world_y = Ry + xs*np.sin(th) + ys*np.cos(th)
        obstacles = np.stack([world_x, world_y], axis=-1)
        return obstacles

    def get_map(self):
        pkg_path = get_package_share_directory('turtlebot3_gazebo')
        map_path = os.path.join(pkg_path, 'maps', 'project_map', 'map.yaml')
        # Load YAML
        with open(map_path, 'r') as file:
            map_data = yaml.safe_load(file)

        return map_data
    
    '''CALLBACKS'''
    def lidar_callback(self, msg):
        #self.get_logger().info("Lidar data received")
        if self.last_scan is None:
            self.get_logger().info("First lidar data received")
        self.last_scan = msg
        filtered_ranges = self.filter_scan_ranges(np.array(msg.ranges))

    def odom_callback(self, msg):
        #self.get_logger().info("Odometry data received")
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        (_, _, yaw) = tf_transformations.euler_from_quaternion(
            [ori.x, ori.y, ori.z, ori.w]
        )
        self.robot.pose = np.array([pos.x, pos.y, yaw])
    
    def camera_callback(self, msg):
        self.get_logger().info("Camera marker data received")
    
    def landmark_callback(self, msg):
        self.get_logger().info("Camera landmark data received")

    def goal_pose_callback(self,msg):
        self.get_logger().info("Goal Pose recieved")
        pos = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation
        (_, _, yaw) = tf_transformations.euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])
        self.yaw = yaw
        self.goal_pose = np.array([pos.x, pos.y, yaw])
        self.dwa_active = True
        self.current_step = 0
        self.get_logger().info("DWA activated toward new goal")

    def dwa_timer_cb(self):
        if not self.dwa_active:
            self.stop_robot()
            return
        
        if self.goal_pose is None:
            self.get_logger().warn("DWA active but no goal pose yet.")
            self.stop_robot()
            self.dwa_active=False
            return
        
        # compute distance to goal
        dist_to_goal = np.linalg.norm(self.robot.pose[0:2] - self.goal_pose[0:2])
        if dist_to_goal < self.goal_dist_tol:
            self.get_logger().info("Goal reached (timer). Stopping DWA.")
            self.emergency_stop(self.last_scan, self.cmd_vel_publisher.publish)
            self.stop_robot()
            self.dwa_active = False
            return
        
        # get obstacles from latest scan
        obstacles = self.process_lidar_data()
        
        # compute command
        try:
            self.emergency_stop(self.last_scan, self.cmd_vel_publisher.publish)
            u=self.compute_cmd(self.goal_pose, self.robot.pose, obstacles)
        except Exception as e:
            self.get_logger().warn(f"DWA Compute failed {e}")
            self.stop_robot()
            self.dwa_active=False
            return
        
        # publish cmd and update internal simulated robot (so the planner's internal model moves)
        cmd = Twist()
        cmd.linear.x = float(u[0])
        cmd.angular.z = float(u[1])
        self.cmd_vel_publisher.publish(cmd)
        # update the internal robot (keeps simulation consistent with chosen commands)
        try:
            self.robot.update_state(u, self.dt)
        except Exception as e:
            self.get_logger().warn(f"robot.update_state failed: {e}")

        # visualization: publish chosen trajectory marker
        self._publish_trajectory_marker(u)

        # periodic debug
        self.current_step += 1
        if self.current_step % self.feedback_rate == 0:
            dist = np.linalg.norm(self.robot.pose[0:2] - self.goal_pose[0:2])
            self.get_logger().info(f"DWA step {self.current_step}: dist_to_goal={dist:.3f}, pose={self.robot.pose}")

# ---------- small RViz visualization for selected trajectory ----------
    def _publish_trajectory_marker(self, u):
        # Simulate a short trajectory for the chosen u for visualization only
        steps = min(self.sim_step, 20)
        pts = []
        x, y, th = float(self.robot.pose[0]), float(self.robot.pose[1]), float(self.robot.pose[2])
        vx, wz = float(u[0]), float(u[1])

        for _ in range(steps):
            x = x + vx * math.cos(th) * self.dt
            y = y + vx * math.sin(th) * self.dt
            th = th + wz * self.dt
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.0
            pts.append(p)

        marker_array = MarkerArray()
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "dwa_path"
        m.id = 0
        m.type = m.LINE_STRIP
        m.action = m.ADD
        m.scale.x = 0.03
        m.pose.orientation.w = 1.0
        m.color.a = 1.0

        # ← restored color settings like previous version
        m.color.r = 1.0
        m.color.g = 0.5
        m.color.b = 0.0

        m.points = pts
        marker_array.markers.append(m)

        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    kwargs = dict(
        init_pose=np.array([0,0,0]),
        max_linear_acc=0.5, # m/s^2
        max_ang_acc=math.pi, # rad/s^2
        max_lin_vel=0.5, # m/s
        min_lin_vel=0.0, # m/s
        max_ang_vel=2.82, # rad/s 
        min_ang_vel=-2.82, # rad/s 
        radius=0.2, # m
    )
    dwa_node=DWA(**kwargs)
    rclpy.spin(dwa_node)
    dwa_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()