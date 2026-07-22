#!/usr/bin/env python3
"""Re-zero an Odometry stream so it starts at the origin (its first pose).

OdometryPublisher reports the robot's ABSOLUTE world pose (starts at the spawn
position, e.g. x=-3), whereas wheel /odom and /odom_ekf are spawn-relative
(start at 0,0). This node subtracts the first ground-truth pose so the re-zeroed
trajectory lines up with the others in the 'odom' frame for RViz comparison.
The original /odom_truth (absolute) is left untouched for the Gazebo marker math.

It also re-zeros the planned path: /vlm_path_truth (anchored in the absolute
ground-truth odom frame) is republished as /vlm_path_truth_rezero using the SAME
locked transform, so the path overlays the re-zeroed truth/odom/ekf trails in
RViz.
"""

import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from transforms3d.euler import euler2quat, quat2euler


class OdomRezero(Node):
    def __init__(self):
        super().__init__('odom_rezero')
        self.in_topic = self.declare_parameter('odom_in_topic', '/odom_truth').value
        self.out_topic = self.declare_parameter('odom_out_topic', '/odom_truth_rezero').value
        self.path_in_topic = self.declare_parameter(
            'path_in_topic', '/vlm_path_truth').value
        self.path_out_topic = self.declare_parameter(
            'path_out_topic', '/vlm_path_truth_rezero').value
        self.frame_id = self.declare_parameter('frame_id', 'odom').value

        self._x0 = self._y0 = self._z0 = self._yaw0 = None
        self._last_path = None  # remember the latest path until the pose is locked

        self.pub = self.create_publisher(Odometry, self.odom_out_topic, 10)
        self.path_pub = self.create_publisher(Path, self.path_out_topic, 10)
        self.create_subscription(Odometry, self.odom_in_topic, self._cb, 10)
        self.create_subscription(Path, self.path_in_topic, self._path_cb, 10)
        self.get_logger().info(
            f're-zeroing {self.odom_in_topic} -> {self.odom_out_topic} and '
            f'{self.path_in_topic} -> {self.path_out_topic}')

    def _rezero_xy(self, x, y):
        """Express (x, y) in the locked initial-pose frame (rotate -yaw0, translate)."""
        c, s = math.cos(-self._yaw0), math.sin(-self._yaw0)
        dx, dy = x - self._x0, y - self._y0
        return c * dx - s * dy, s * dx + c * dy

    def _cb(self, msg):
        p = msg.pose.pose
        o = p.orientation
        yaw = quat2euler((o.w, o.x, o.y, o.z))[2]
        first = self._x0 is None
        if first:
            self._x0, self._y0 = p.position.x, p.position.y
            self._z0, self._yaw0 = p.position.z, yaw
            self.get_logger().info(
                f'locked initial pose ({self._x0:.2f}, {self._y0:.2f}, '
                f'yaw={self._yaw0:.2f})')

        rx, ry = self._rezero_xy(p.position.x, p.position.y)

        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.frame_id
        out.child_frame_id = msg.child_frame_id
        out.pose.pose.position.x = rx
        out.pose.pose.position.y = ry
        out.pose.pose.position.z = p.position.z - self._z0
        ryaw = yaw - self._yaw0
        qw, _, _, qz = euler2quat(0.0, 0.0, ryaw)
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw
        out.twist = msg.twist  # body-frame velocities are unaffected
        self.pub.publish(out)

        # A path may arrive before the first /odom_truth message; republish it
        # once the transform is locked.
        if first and self._last_path is not None:
            self._publish_rezeroed_path(self._last_path)

    def _path_cb(self, msg):
        self._last_path = msg
        if self._x0 is None:
            self.get_logger().warn(
                f'got {self.path_in_topic} before {self.in_topic} lock; '
                f'will republish once the initial pose is locked')
            return
        self._publish_rezeroed_path(msg)

    def _publish_rezeroed_path(self, msg):
        out = Path()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.frame_id
        for ps in msg.poses:
            rx, ry = self._rezero_xy(ps.pose.position.x, ps.pose.position.y)
            nps = PoseStamped()
            nps.header.stamp = ps.header.stamp
            nps.header.frame_id = self.frame_id
            nps.pose.position.x = rx
            nps.pose.position.y = ry
            nps.pose.position.z = ps.pose.position.z - self._z0
            o = ps.pose.orientation
            ryaw = quat2euler((o.w, o.x, o.y, o.z))[2] - self._yaw0
            qw, _, _, qz = euler2quat(0.0, 0.0, ryaw)
            nps.pose.orientation.z = qz
            nps.pose.orientation.w = qw
            out.poses.append(nps)
        self.path_pub.publish(out)


def main():
    rclpy.init()
    node = OdomRezero()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
