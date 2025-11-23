import rclpy
from rclpy.node import Node
import numpy as np
from numpy.linalg import inv
from nav_msgs.msg import Odometry
from landmark_msgs.msg import LandmarkArray
import yaml
import os
from math import sin, cos, atan2, sqrt, pi

class EKFNode(Node):
    def __init__(self):
        # Initialize the node
        super().__init__('ekf_node')
        self.get_logger().info('EKF node has been started.')

        # publishers / subscribers
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # subscribe to both topics used in doc: /landmarks (sim) and /camera/landmarks (real)
        self.sub_landmarks = self.create_subscription(LandmarkArray, '/landmarks', self.landmark_callback, 10)
        self.sub_camera_landmarks = self.create_subscription(LandmarkArray, '/camera/landmarks', self.landmark_callback, 10)
        self.pub_ekf = self.create_publisher(Odometry, '/ekf', 10)
        
        # timer for periodic EKF updates
        self.create_timer(0.05, self.timer_callback)  # 20 Hz timer

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
        self.v=0.0
        self.omega=0.0
        self.x_new=0.0
        self.y_new=0.0
        self.theta_new=0.0

    # Load landmarks from a YAML file
    def load_landmarks(self, yaml_path):
        try:
            with open(yaml_path, 'r') as f:
                cfg = yaml.safe_load(f)

            lm_map = {}

            if 'landmarks' in cfg:
                ids = cfg['landmarks']['id']
                xs  = cfg['landmarks']['x']
                ys  = cfg['landmarks']['y']
                # optional z
                zs  = cfg['landmarks'].get('z', None)

                for i in range(len(ids)):
                    lm_id = int(ids[i])
                    x = float(xs[i])
                    y = float(ys[i])
                    if zs is not None:
                        z = float(zs[i])
                        lm_map[lm_id] = (x, y, z)
                    else:
                        lm_map[lm_id] = (x, y)
            else:
                self.get_logger().warn(f'No "landmarks" key found in YAML; file contents: {cfg}')

            self.get_logger().info(f'Loaded {len(lm_map)} landmarks')
            return lm_map

        except Exception as e:
            self.get_logger().error(f'Failed to load landmarks YAML: {e}')
            return {}
            
    def odom_callback(self, msg: Odometry):
        # Process odometry data
        #self.get_logger().info('Received odometry data.')
        self.v=msg.twist.twist.linear.x
        self.omega=msg.twist.twist.angular.z

    def landmark_callback(self, msg: LandmarkArray):
        # Process landmark data
        self.get_logger().info('Received landmark data.')
        # Here you would typically update the EKF with the landmark observations
        updated=False
        for lm in msg.landmarks:
            try:
                self.get_logger().info(f'Processing landmark ID: {lm.id}')
                lm_id = lm.id
            except Exception:
                self.get_logger().warn('Landmark ID missing or invalid; skipping.')
                lm_id = int(lm.id)
            
            if lm_id not in self.landmarks:
                self.get_logger().warn(f'Unknown landmark ID {lm_id} observed; skipping.')
                continue  # skip unknown landmarks

            r=getattr(lm, 'range', getattr(lm, 'r', None))
            b=getattr(lm, 'bearing', getattr(lm, 'bearing', None))

            if r is None or b is None:
                self.get_logger().warn(f'Landmark {lm_id} missing range/bearing attributes.')
                continue

            z = np.array([[r], [b]])
            
            hx_lambda = lambda: self.eval_hx(self.mu[0,0], self.mu[1,0], self.mu[2,0], self.landmarks[lm_id])
            Ht_lambda = lambda: self.eval_Ht(self.mu[0,0], self.mu[1,0], self.mu[2,0], self.landmarks[lm_id])

            self.update(z,
                        eval_hx=hx_lambda,
                        eval_Ht=Ht_lambda,
                        Qt=self.Q,
                        Ht_args=(),    # our lambdas capture current mu and landmark
                        hx_args=(),
                        residual=self.residual_angle)
            
            updated = True
            if updated:
                self.get_logger().info('EKF updated with landmark observations.')
                self.publish_ekf()

    def timer_callback(self):
        # Periodically publish the EKF estimated pose
        u = np.array([self.v, self.omega])
        sigma_u = np.array([0.1, 0.05])
        self.predict(u, sigma_u, g_extra_args=(self.dt,))
        self.publish_ekf()
    
    def publish_ekf(self):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'
        msg.pose.pose.position.x = self.mu[0, 0]
        msg.pose.pose.position.y = self.mu[1, 0]

        # Convert yaw to quaternion
        yaw = float(self.mu[2])
        qz = sin(yaw/2.0)
        qw = cos(yaw/2.0)
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        # Optionally fill covariance (pose covariance is 6x6 flattened) - put Sigma into x,y,yaw slots
        cov = np.zeros((6,6))
        cov[0:2, 0:2] = self.Sigma[0:2, 0:2]
        cov[5,5] = self.Sigma[2,2]
        msg.pose.covariance = cov.flatten().tolist()
        msg.twist.twist.linear.x = self.v
        msg.twist.twist.angular.z = self.omega

        self.pub_ekf.publish(msg)

    def residual_angle(self, z, z_hat, angle_index=1):
    # z and z_hat are column vectors
        y = z - z_hat
    # normalize bearing difference
        y[angle_index, 0] = self.angle_wrap(y[angle_index, 0])
        return y
    
    # ---------------------------
    #  Motion and measurement models + Jacobians
    # ---------------------------
    # Motion model g(mu, u) and Jacobians Gt, Vt
    # Measurement model h(x) and Jacobian Ht
    def eval_gux(self, mu, u, sigma_u, dt):
        """Motion model g(mu, u) returning new mu (3x1)."""
        x = mu[0,0]; y = mu[1,0]; theta = mu[2,0]
        v = float(u[0]); w = float(u[1])

        if abs(w) < 1e-6:
            x_new = x + v * dt * cos(theta)
            y_new = y + v * dt * sin(theta)
            theta_new = theta
        else:
            x_new = x + (v / w) * (sin(theta + w*dt) - sin(theta))
            y_new = y + (v / w) * (cos(theta) - cos(theta+w*dt))
            theta_new = theta + w * dt

        theta_new = self.angle_wrap(theta_new)
        return np.array([[x_new], [y_new], [theta_new]])

    # Jacobians
    def eval_Gt(self, x, y, theta, v, w, dt):
        """Jacobian of motion wrt state (3x3)."""
        G = np.eye(3) # initialize as identity
        if abs(w) > 1e-6:
            G[0,2] = (v/w)*(cos(theta + w*dt) - cos(theta))
            G[1,2] = (v/w)*(sin(theta + w*dt) - sin(theta))
        else:
            G[0,2] = -v * dt * sin(theta)
            G[1,2] = v * dt * cos(theta)
        return G

    def eval_Vt(self, x, y, theta, v, w, dt):
        """Jacobian of motion wrt control (3x2)."""
        V = np.zeros((3,2))
        if abs(w) > 1e-6:
            V[0,0] = (sin(theta + w*dt) - sin(theta)) / w
            V[0,1] = v * (sin(theta) - sin(theta + w*dt)) / (w**2) + v * cos(theta + w*dt) * dt / w
            V[1,0] = (-cos(theta + w*dt) + cos(theta)) / w
            V[1,1] = v * (cos(theta + w*dt) - cos(theta)) / (w**2) + v * sin(theta + w*dt) * dt / w
            V[2,1] = dt
        else:
            V[0,0] = dt * cos(theta)
            V[1,0] = dt * sin(theta)
            V[2,1] = dt
        return V

    def eval_hx(self, x, y, theta, landmark):
        """Measurement model h(x) for a given landmark -> [range, bearing]^T."""
        lx, ly, _ = landmark
        dx = lx - x
        dy = ly - y
        r = sqrt(dx*dx + dy*dy)
        b = self.angle_wrap(atan2(dy, dx) - theta)
        return np.array([[r], [b]])

    def eval_Ht(self, x, y, theta, landmark):
        """Jacobian of measurement wrt state: 2x3."""
        lx, ly, _ = landmark
        dx = lx - x
        dy = ly - y
        q = dx*dx + dy*dy
        r = sqrt(q)
        if q == 0:
            # degenerate; avoid div by zero
            return np.zeros((2,3))
        H = np.array([
            [-dx / r, -dy / r, 0.0],
            [ dy / q, -dx / q, -1.0]
        ])
        return H


    def predict(self, u, sigma_u, g_extra_args=()):
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

    @staticmethod
    def angle_wrap(angle):
        return (angle + pi) % (2 * pi) - pi
    
def main(args=None):
    rclpy.init(args=args)
    ekf_node = EKFNode()
    rclpy.spin(ekf_node)
    ekf_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()