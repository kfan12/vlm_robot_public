#!/usr/bin/env python3
"""fpv_truth_overlay — make the RViz FPV camera overlay track GROUND TRUTH.

RViz's Camera display positions its virtual camera at the image's header
frame_id, resolved through TF. Normally that frame flows through the EKF
(odom->base_link), so as the EKF drifts from the true pose the overlay slides
off the real cones. This node republishes the camera image/info anchored to a
ground-truth optical frame so the overlay locks onto the scene regardless of
localization drift.

It:
  * broadcasts TF <parent_frame> -> <truth_frame> = (ground-truth base pose from
    /odom_truth) composed with the static base_link->optical camera transform;
  * republishes the RGB image on <out_ns>/image_raw stamped with <truth_frame>;
  * republishes a corrected CameraInfo on <out_ns>/camera_info (true intrinsics
    for the rendered resolution — Gazebo reports stale 320x240 defaults; RViz
    ignores cx but a future VLM reading K will need the right values).

Point the RViz "FPV Overlay" Camera display at <out_ns>/image_raw; it will
auto-subscribe to <out_ns>/camera_info for intrinsics.

Assumes the 'odom' frame origin coincides with the world origin (robot spawned
at the origin), so the absolute /odom_truth pose is published directly under
<parent_frame>. Run with ROS sourced (system python). Images are passed through
untouched (only the header frame_id changes), so no cv_bridge dependency.
"""
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
import tf2_ros
from transforms3d.quaternions import quat2mat, mat2quat


class FpvTruthOverlay(Node):
    def __init__(self):
        super().__init__('fpv_truth_overlay')
        self.truth_topic = self.declare_parameter('truth_topic', '/odom_truth').value
        self.image_in = self.declare_parameter('image_in', '/camera/front/image_raw').value
        self.info_in = self.declare_parameter('info_in', '/camera/front/camera_info').value
        out_ns = self.declare_parameter('out_ns', '/fpv_truth').value
        self.parent_frame = self.declare_parameter('parent_frame', 'odom').value
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value
        self.optical_frame = self.declare_parameter(
            'optical_frame', 'front_camera_optical_frame').value
        self.truth_frame = self.declare_parameter(
            'truth_frame', 'front_camera_optical_frame_truth').value
        # /odom_truth reports the ABSOLUTE world pose; world_frame is the absolute
        # frame that cone_marker_node ties to odom via a static odom->world TF.
        self.world_frame = self.declare_parameter('world_frame', 'world').value
        self.hfov = float(self.declare_parameter('horizontal_fov', 1.047).value)
        self.fix_intrinsics = bool(self.declare_parameter('fix_intrinsics', True).value)

        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf, self)
        self.br = tf2_ros.TransformBroadcaster(self)
        self._T_base_opt = None    # static base_link->optical (4x4)
        self._T_odom_world = None  # static odom->world (4x4), folds in spawn offset
        self._ow_tries = 0
        # Resolve the static transforms off the hot path (lookups can block).
        self._static_timer = self.create_timer(0.5, self._resolve_static)

        self.image_pub = self.create_publisher(Image, out_ns + '/image_raw', 10)
        self.info_pub = self.create_publisher(CameraInfo, out_ns + '/camera_info', 10)
        self.create_subscription(Odometry, self.truth_topic, self._truth_cb, 10)
        # Camera streams over ros_gz commonly use best-effort QoS; subscribe with
        # sensor-data QoS so we don't silently miss frames.
        self.create_subscription(Image, self.image_in, self._image_cb,
                                 qos_profile_sensor_data)
        self.create_subscription(CameraInfo, self.info_in, self._info_cb,
                                 qos_profile_sensor_data)
        self.get_logger().info(
            f'FPV truth overlay: {self.image_in} -> {out_ns}/image_raw in frame '
            f'{self.truth_frame}, anchored to {self.truth_topic}')

    def _lookup(self, target, source):
        """Return the static target<-source transform as a 4x4, or None."""
        try:
            tf = self.tf_buf.lookup_transform(
                target, source, rclpy.time.Time(), timeout=Duration(seconds=0.1))
        except Exception:
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        M = np.eye(4)
        M[:3, :3] = quat2mat([q.w, q.x, q.y, q.z])
        M[:3, 3] = [t.x, t.y, t.z]
        return M

    def _resolve_static(self):
        """Cache base->optical (URDF) and odom->world (spawn offset) once."""
        if self._T_base_opt is None:
            self._T_base_opt = self._lookup(self.base_frame, self.optical_frame)
            if self._T_base_opt is not None:
                self.get_logger().info('cached static base->optical transform')
        if self._T_odom_world is None:
            self._T_odom_world = self._lookup(self.parent_frame, self.world_frame)
            self._ow_tries += 1
            if self._T_odom_world is not None:
                self.get_logger().info(
                    f'cached static {self.parent_frame}->{self.world_frame} (spawn offset)')
            elif self._ow_tries >= 10:  # ~5 s: no cone_marker / world frame
                self.get_logger().warn(
                    f'no {self.parent_frame}->{self.world_frame} TF; assuming '
                    f'identity (odom == world)')
                self._T_odom_world = np.eye(4)
        if self._T_base_opt is not None and self._T_odom_world is not None:
            self._static_timer.cancel()

    def _truth_cb(self, msg):
        if self._T_base_opt is None:
            return
        T_ow = self._T_odom_world if self._T_odom_world is not None else np.eye(4)
        p = msg.pose.pose
        T_wb = np.eye(4)  # base pose in the absolute world frame (= /odom_truth)
        T_wb[:3, :3] = quat2mat([p.orientation.w, p.orientation.x,
                                 p.orientation.y, p.orientation.z])
        T_wb[:3, 3] = [p.position.x, p.position.y, p.position.z]
        # odom -> truth optical = (odom->world) @ (world->base) @ (base->optical)
        T_oo = T_ow @ T_wb @ self._T_base_opt
        qw, qx, qy, qz = mat2quat(T_oo[:3, :3])

        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = self.parent_frame
        t.child_frame_id = self.truth_frame
        t.transform.translation.x = float(T_oo[0, 3])
        t.transform.translation.y = float(T_oo[1, 3])
        t.transform.translation.z = float(T_oo[2, 3])
        t.transform.rotation.w = float(qw)
        t.transform.rotation.x = float(qx)
        t.transform.rotation.y = float(qy)
        t.transform.rotation.z = float(qz)
        self.br.sendTransform(t)

    def _image_cb(self, msg):
        msg.header.frame_id = self.truth_frame  # re-anchor, image data untouched
        self.image_pub.publish(msg)

    def _info_cb(self, msg):
        msg.header.frame_id = self.truth_frame
        if self.fix_intrinsics and msg.width and msg.height:
            fx = (msg.width / 2.0) / math.tan(self.hfov / 2.0)
            cx, cy = msg.width / 2.0, msg.height / 2.0
            msg.k = [fx, 0.0, cx, 0.0, fx, cy, 0.0, 0.0, 1.0]
            msg.p = [fx, 0.0, cx, 0.0, 0.0, fx, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.info_pub.publish(msg)


def main():
    rclpy.init()
    node = FpvTruthOverlay()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
