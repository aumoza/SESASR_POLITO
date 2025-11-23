#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import argparse
import csv
import os
import yaml
import math
import numpy as np
import matplotlib.pyplot as plt
from threading import Lock


def covariance_to_ellipse(P, n_std=2.0):
    """
    Convert 2x2 covariance matrix P to ellipse parameters (width, height, angle)
    n_std defines scale factor (2 → 95% confidence)
    """
    eigvals, eigvecs = np.linalg.eigh(P)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    angle = math.atan2(eigvecs[1, 0], eigvecs[0, 0])
    width  = 2 * n_std * math.sqrt(eigvals[0])
    height = 2 * n_std * math.sqrt(eigvals[1])
    return width, height, angle


class TrajectoryPlotNode(Node):
    def __init__(self, landmark_file):
        super().__init__('trajectory_plot_node')
        self.lock = Lock()

        # Trajectories
        self.odom_xy = []
        self.gt_xy   = []
        self.ekf_xy  = []

        # Covariance storage
        self.ekf_cov = []          # list of 2×2 XY covariance matrices

        # Subscribe
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(Odometry, '/ground_truth', self.gt_cb, 10)
        self.create_subscription(Odometry, '/ekf', self.ekf_cb, 10)

        # Landmarks
        self.landmarks_list = []   # [(id, x, y, z)]
        self.landmarks_dict = {}   # {id: (x, y, z)}
        self.load_landmarks(landmark_file)

        self.get_logger().info('Trajectory Plot Node started.')

    # ---------------------------
    # Callbacks
    # ---------------------------
    def odom_cb(self, msg):
        with self.lock:
            p = msg.pose.pose.position
            self.odom_xy.append((p.x, p.y))

    def gt_cb(self, msg):
        with self.lock:
            p = msg.pose.pose.position
            self.gt_xy.append((p.x, p.y))

    def ekf_cb(self, msg):
        with self.lock:
            p = msg.pose.pose.position
            self.ekf_xy.append((p.x, p.y))

            # Extract 2D position covariance (first 2×2 block)
            P = np.array(msg.pose.covariance).reshape(6, 6)[0:2, 0:2]
            self.ekf_cov.append(P)

    # ---------------------------
    # Load Landmarks
    # ---------------------------
    def load_landmarks(self, path):
        if not os.path.exists(path):
            self.get_logger().warn(f"Landmark file not found: {path}")
            return

        if path.endswith('.yaml') or path.endswith('.yml'):
            self.load_yaml_landmarks(path)
        else:
            self.load_csv_landmarks(path)

        self.get_logger().info(f"Loaded {len(self.landmarks_list)} landmarks.")

    def load_yaml_landmarks(self, path):
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f)
            ids = cfg['landmarks']['id']
            xs  = cfg['landmarks']['x']
            ys  = cfg['landmarks']['y']
            zs = cfg['landmarks']['z']
            for i in range(len(ids)):
                self.landmarks_dict[int(ids[i])] = (float(xs[i]), float(ys[i]))
                self.landmarks_list.append((int(ids[i]), float(xs[i]), float(ys[i])))
        except Exception as e:
            self.get_logger().error(f"Failed YAML load: {e}")

    def load_csv_landmarks(self, path):
        try:
            with open(path) as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row or row[0].startswith('#'):
                        continue
                    if len(row) == 3:
                        lm_id = row[0]
                        x, y = float(row[1]), float(row[2])
                    else:
                        lm_id = None
                        x, y = float(row[0]), float(row[1])
                    self.landmarks_list.append((lm_id, x, y))
        except Exception as e:
            self.get_logger().error(f"Failed CSV load: {e}")

    # ---------------------------
    # Plotting
    # ---------------------------
    def plot_covariances(self, ax, step=500):
        """Plot covariance ellipses every 'step' EKF messages."""
        if not self.ekf_xy or not self.ekf_cov:
            print("No EKF XY and not EKF COV")
            return

        for i in range(0, len(self.ekf_xy), step):
            x, y = self.ekf_xy[i]
            P = self.ekf_cov[i]

            width, height, angle = covariance_to_ellipse(P)
            ellipse = plt.matplotlib.patches.Ellipse(
                xy=(x, y),
                width=width,
                height=height,
                angle=np.degrees(angle),
                edgecolor='blue',
                facecolor='none',
                linewidth=0.5
            )
            ax.add_patch(ellipse)

    def plot_and_save(self, save_path='trajectory.png', show_plot=True):
        with self.lock:
            fig, ax = plt.subplots(figsize=(8, 8))

            def plot_xy(data, label, color, linewidth=1.0):
                if not data:
                    return
                xs = [p[0] for p in data]
                ys = [p[1] for p in data]
                ax.plot(xs, ys, label=label, color=color)

            plot_xy(self.gt_xy,   'ground truth', 'blue', 2)
            plot_xy(self.odom_xy, 'odom', 'green', 2)
            plot_xy(self.ekf_xy,  'ekf', 'red')

            # Plot covariance ellipses
            self.plot_covariances(ax, step=20)

            # Landmarks
            for lm_id, x, y in self.landmarks_list:
                ax.scatter([x], [y], marker='x', color='black')
                if lm_id is not None:
                    ax.annotate(str(lm_id), (x, y), xytext=(3, 3), textcoords='offset points')

            ax.set_aspect('equal')
            ax.grid(True)
            ax.legend()
            plt.tight_layout()
            plt.savefig(save_path, dpi=200)
            self.get_logger().info(f"Saved plot to {save_path}")

            if show_plot:
                plt.show()
            plt.close(fig)


def main(args=None):
    parser = argparse.ArgumentParser()
    lm_yaml = os.path.expanduser('~/ros2_ws/src/turtlebot3_perception/turtlebot3_perception/config/landmarks.yaml')
    parser.add_argument('--save', default='trajectory.png')
    parser.add_argument('--no-show', dest='show', action='store_false')
    parsed, _ = parser.parse_known_args(args)

    rclpy.init()
    node = TrajectoryPlotNode(lm_yaml)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down → plotting…")
        node.plot_and_save(save_path=parsed.save, show_plot=parsed.show)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
