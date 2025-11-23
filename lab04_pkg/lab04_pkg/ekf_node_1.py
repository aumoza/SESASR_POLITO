import rclpy
from rclpy.node import Node
import numpy as np
from numpy.linalg import inv
from nav_msgs.msg import Odometry
from landmark_msgs.msg import LandmarkArray
import yaml
import os
from math import sin, cos, atan2, sqrt, pi

class EKFNode_1(Node):
    def __init__(self):
        super().__init__('ekf_node_1')
        self.get_logger().info('EKF node has been started.')

        # timer for periodic EKF updates
        self.create_timer(0.05, self.timer_callback)  # 20 Hz timer

        # publishers / subscribers
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # subscribe to both topics used in doc: /landmarks (sim) and /camera/landmarks (real)
        self.sub_landmarks = self.create_subscription(LandmarkArray, '/landmarks', self.landmark_callback, 10)
        self.sub_camera_landmarks = self.create_subscription(LandmarkArray, '/camera/landmarks', self.landmark_callback, 10)
        self.pub_ekf = self.create_publisher(Odometry, '/ekf_1', 10)

        # --- EKF state initialization ---
        self.declare_parameter('landmarks_yaml', '')
        lm_yaml = self.get_parameter('landmarks_yaml').get_parameter_value().string_value
        if not lm_yaml:
            # default path inside workspace; change as needed
            lm_yaml = os.path.expanduser('~/ros2_ws/src/turtlebot3_perception/turtlebot3_perception/config/landmarks.yaml')
            self.get_logger().info(f'Using default landmarks YAML path: {lm_yaml}')

        self.dt = 0.05  # time step (20 Hz)
        self.x = np.zeros((3, 1))  # state: [x, y, theta]
        self.P = np.eye(3) * 0.1   # covariance matrix
        self.mu = np.zeros((3, 1), dtype=float)  # mean state estimate
        self.Sigma = np.eye(3) * 0.1  # covariance state estimate
        self.Mt=np.diag(np.array([0.1, np.deg2rad(5)])**2) # control noise covariance
        self.Q=np.diag(np.array([0.1, np.deg2rad(5)])**2)  # measurement noise covariance
        self.val = 0.0  # linear velocity
        self.omega = 0.0  # angular velocity
        self.landmarks=self.load_landmarks(lm_yaml) # load landmarks from yaml
        self._I = np.eye(3) # identity matrix

    def load_landmarks(self, yaml_path):
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
        landmarks = {}
        for lm in data['landmarks']:
            landmarks[lm['id']] = (lm['x'], lm['y'])
        self.get_logger().info(f'Loaded {len(landmarks)} landmarks from {yaml_path}')
        return landmarks
    
    def odom_callback(self, msg: Odometry):
        self.val = msg.twist.twist.linear.x
        self.omega = msg.twist.twist.angular.z

    def landmark_callback(self, msg: LandmarkArray):
        measurements = []
        self.get_logger().info(f'Received {len(msg.landmarks)} landmarks.')
        landmark_updated=False
        for lm in msg.landmarks:
            lm_id = lm.id
            if lm_id in self.landmarks:
                lm_x, lm_y = self.landmarks[lm_id]
                dx = lm_x - self.mu[0, 0]
                dy = lm_y - self.mu[1, 0]
                r = sqrt(dx**2 + dy**2)
                phi = atan2(dy, dx) - self.mu[2, 0]
                measurements.append((lm_id, r, phi))
        self.update(measurements)
    
    def timer_callback(self):
        self.predict_ekf()
        self.update()
        self.publish_ekf()

    def publish_ekf(self):
        ekf_msg = Odometry()
        ekf_msg.header.stamp = self.get_clock().now().to_msg()
        ekf_msg.header.frame_id = 'odom'
        ekf_msg.pose.pose.position.x = self.mu[0,0]
        ekf_msg.pose.pose.position.y = self.mu[1,0]
        ekf_msg.pose.pose.orientation.z = sin(self.mu[2,0]/2.0)
        ekf_msg.pose.pose.orientation.w = cos(self.mu[2,0]/2.0)
        self.pub_ekf.publish(ekf_msg)

    def predict_ekf(self, u, sigma_u, g_extra_args=()):
        """
        Update the state prediction using the control input u and compute the relative uncertainty ellipse
        Parameters
        ----------

        u : np.array
            command for this step.

        sigma_u : np.array
            std dev for each component of the command signal

        extra_args : tuple
            any additional required parameter: dt

        Modified variables:
            self.mu: the state prediction
            self.Sigma: the covariance matrix of the state prediction
        """
        # Update the state prediction evaluating the motion model
        self.mu = self.eval_gux(self.mu, u, sigma_u, *g_extra_args)

        args = (*self.mu, *u)
        # Update the covariance matrix of the state prediction,
        # you need to evaluate the Jacobians Gt and Vt

        Gt = self.eval_Gt(*args, *g_extra_args)
        Vt = self.eval_Vt(*args, *g_extra_args)
        self.Sigma = Gt @ self.Sigma @ Gt.T + Vt @ self.Mt @ Vt.T

    def update(self, z, eval_hx, eval_Ht, Qt, Ht_args=(), hx_args=(),  residual=np.subtract, **kwargs):
        """Performs the update innovation of the extended Kalman filter.

        Parameters
        ----------

        z : np.array
            measurement for this step.

        lmark : [x, y] list-like
            Landmark location in cartesian coordinates.

        residual : function (z, z2), optional
            Optional function that computes the residual (difference) between
            the two measurement vectors. If you do not provide this, then the
            built in minus operator will be used. You will normally want to use
            the built in unless your residual computation is nonlinear (for
            example, if they are angles)
        """

        # Convert the measurement to a vector if necessary. Needed for the residual computation
        if np.isscalar(z):
            z = np.asarray([z], float)
            Qt = np.atleast_2d(Qt).astype(float)
            Ht = np.atleast_2d(Ht).astype(float)

        # Compute the Kalman gain, you need to evaluate the Jacobian Ht
        Ht = eval_Ht(*Ht_args)
        SigmaHT = self.Sigma @ Ht.T
        self.S = Ht @ SigmaHT + Qt
        self.K = SigmaHT @ inv(self.S)

        # Evaluate the expected measurement and compute the residual, then update the state prediction
        z_hat = eval_hx(*hx_args)
        if np.isscalar(z_hat):
            z_hat = np.asarray([z_hat], float)

        # if the z measurement include an angle update, we need to specify the positional index to normalize the residual
        y = residual(z, z_hat, **kwargs)
        self.mu = self.mu + self.K @ y

        # P = (I-KH)P(I-KH)' + KRK' is more numerically stable and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.
        # Note that I is the identity matrix.
        I_KH = self._I - self.K @ Ht
        self.Sigma = I_KH @ self.Sigma @ I_KH.T + self.K @ Qt @ self.K.T


    def eval_gux():
        # Evaluate the motion model
        pass
    
    def eval_Gt():
        # Evaluate the Jacobian Gt
        pass

    def eval_Vt():
        # Evaluate the Jacobian Vt
        pass
    