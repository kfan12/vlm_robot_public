#!/usr/bin/env python3
"""Broadcast a TF from an Odometry topic so RViz can render a ghost robot at the
ground-truth pose. Publishes parent_frame -> child_frame from /odom_truth, where
child_frame (e.g. 'truth/base_link') is the prefixed root used by a second
robot_state_publisher (frame_prefix: 'truth/')."""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class OdomTfBroadcaster(Node):
    def __init__(self):
        super().__init__('odom_tf_broadcaster')
        self.odom_topic = self.declare_parameter('odom_topic', '/odom_truth').value
        self.child_frame = self.declare_parameter('child_frame', 'truth/base_link').value
        # Empty parent_frame => use the Odometry message's own header.frame_id.
        self.parent_frame = self.declare_parameter('parent_frame', '').value

        self.br = TransformBroadcaster(self)
        self.create_subscription(Odometry, self.odom_topic, self._cb, 10)
        self.get_logger().info(
            f'broadcasting TF {self.parent_frame or "<odom.frame_id>"} -> '
            f'{self.child_frame} from {self.odom_topic}')

    def _cb(self, msg):
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = self.parent_frame or msg.header.frame_id or 'odom'
        t.child_frame_id = self.child_frame
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation
        self.br.sendTransform(t)


def main():
    rclpy.init()
    node = OdomTfBroadcaster()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
