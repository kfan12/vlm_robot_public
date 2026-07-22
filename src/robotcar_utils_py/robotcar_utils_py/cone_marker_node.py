#!/usr/bin/env python3
"""cone_marker_node — show the world's cones in RViz.

RViz only renders ROS topics, so the Gazebo world cones are invisible to it.
This node publishes:
  1. a latched MarkerArray of the cones (in the 'world' frame), and
  2. a static TF odom -> world derived from the robot's spawn pose, so the
     cones land at the correct place relative to RViz's 'odom' fixed frame.

The spawn pose is taken from parameters (passed by the spawn launch), so the
cones stay correct even when you change spawn_x / spawn_y / spawn_yaw.

Cone positions mirror src/robotcar_gazebo/worlds/cone_lane.world.sdf — keep them
in sync if you edit the world.
"""

import math
import xml.etree.ElementTree as ET

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import StaticTransformBroadcaster
from transforms3d.euler import euler2quat

# Fallback used only if the world file can't be parsed.
# (x, y, z, height, radius, (r, g, b)) in the WORLD frame — from cone_lane.world.sdf
_FALLBACK_CONES = [
    (4.0,  0.4, 0.1, 0.2, 0.05, (1.0, 0.9, 0.0)),   # left row (yellow)
    (5.0,  0.4, 0.1, 0.2, 0.05, (1.0, 0.9, 0.0)),
    (6.0,  0.4, 0.1, 0.2, 0.05, (1.0, 0.9, 0.0)),
    (4.0, -0.4, 0.1, 0.2, 0.05, (1.0, 0.9, 0.0)),   # right row (yellow)
    (5.0, -0.4, 0.1, 0.2, 0.05, (1.0, 0.9, 0.0)),
    (6.0, -0.4, 0.1, 0.2, 0.05, (1.0, 0.9, 0.0)),
    (7.0,  0.0, 0.125, 0.25, 0.06, (1.0, 0.0, 0.0)),  # goal (red)
]


def parse_world_cones(world_file):
    """Parse every <model name='cone*'> from an SDF world into
    (x, y, z, height, radius, (r, g, b)) tuples. Returns None on any failure so
    the caller can fall back to the hardcoded list."""
    try:
        root = ET.parse(world_file).getroot()
    except (ET.ParseError, OSError):
        return None

    cones = []
    for model in root.iter('model'):
        name = model.get('name', '')
        if not name.startswith('cone'):
            continue
        pose_el = model.find('pose')
        if pose_el is None or not pose_el.text:
            continue
        vals = pose_el.text.split()
        x, y, z = float(vals[0]), float(vals[1]), float(vals[2])

        # visual comes before collision, so .// picks the visual cylinder.
        cyl = model.find('.//cylinder')
        radius = float(cyl.find('radius').text) if cyl is not None else 0.05
        length = float(cyl.find('length').text) if cyl is not None else 0.2

        diffuse = model.find('.//material/diffuse')
        if diffuse is not None and diffuse.text:
            c = diffuse.text.split()
            color = (float(c[0]), float(c[1]), float(c[2]))
        else:
            color = (1.0, 0.9, 0.0)

        cones.append((x, y, z, length, radius, color))

    return cones if cones else None


class ConeMarkerNode(Node):
    def __init__(self):
        super().__init__('cone_marker_node')

        self.world_frame = self.declare_parameter('world_frame', 'world').value
        self.odom_frame = self.declare_parameter('odom_frame', 'odom').value
        # Robot spawn pose in the world frame (must match the spawn launch args).
        self.spawn_x = float(self.declare_parameter('spawn_x', -3.0).value)
        self.spawn_y = float(self.declare_parameter('spawn_y', 0.0).value)
        self.spawn_yaw = float(self.declare_parameter('spawn_yaw', 0.0).value)
        # Source of truth for cone poses: parse the world SDF, fall back to the
        # hardcoded list if the path is empty/unreadable.
        world_file = self.declare_parameter('world_file', '').value
        self.cones = parse_world_cones(world_file) if world_file else None
        if self.cones is None:
            self.cones = _FALLBACK_CONES
            self.get_logger().warn(
                f'Could not parse cones from world_file="{world_file}"; '
                f'using {len(self.cones)} hardcoded fallback cones.')
        else:
            self.get_logger().info(
                f'Parsed {len(self.cones)} cones from {world_file}.')

        # Publish the world-cone markers for RViz? The static odom->world TF is
        # ALWAYS published (fpv_truth_overlay depends on it for its spawn offset);
        # only the visual MarkerArray is gated. Set False to keep RViz clear of
        # world objects while preserving the transform.
        self.publish_markers = bool(self.declare_parameter('publish_markers', True).value)

        # Latched QoS so RViz receives the markers even though we publish once.
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.marker_pub = self.create_publisher(MarkerArray, '/cone_markers', latched)

        self.static_tf = StaticTransformBroadcaster(self)
        self._publish_static_tf()
        if self.publish_markers:
            self.marker_pub.publish(self._build_markers())

        self.get_logger().info(
            f'Published {"cone markers + " if self.publish_markers else ""}'
            f'static {self.odom_frame}->{self.world_frame} TF for spawn '
            f'({self.spawn_x}, {self.spawn_y}, yaw={self.spawn_yaw}) '
            f'[markers={"on" if self.publish_markers else "off"}].')

    def _publish_static_tf(self):
        """odom -> world = inverse of the robot's spawn pose (odom origin sits at
        the spawn point in the world)."""
        inv_yaw = -self.spawn_yaw
        # inverse translation = -R(-yaw) * spawn_translation
        tx = -(math.cos(inv_yaw) * self.spawn_x - math.sin(inv_yaw) * self.spawn_y)
        ty = -(math.sin(inv_yaw) * self.spawn_x + math.cos(inv_yaw) * self.spawn_y)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.odom_frame      # parent
        t.child_frame_id = self.world_frame      # child
        t.transform.translation.x = tx
        t.transform.translation.y = ty
        t.transform.translation.z = 0.0
        # transforms3d returns (w, x, y, z) for the Z-axis rotation.
        qw, _, _, qz = euler2quat(0.0, 0.0, inv_yaw)
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.static_tf.sendTransform(t)

    def _build_markers(self):
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for i, (x, y, z, height, radius, (r, g, b)) in enumerate(self.cones):
            m = Marker()
            m.header.frame_id = self.world_frame
            m.header.stamp = stamp
            m.ns = 'cones'
            m.id = i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = z
            m.pose.orientation.w = 1.0
            m.scale.x = 2.0 * radius
            m.scale.y = 2.0 * radius
            m.scale.z = height
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 1.0
            arr.markers.append(m)
        return arr


def main(args=None):
    rclpy.init(args=args)
    node = ConeMarkerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
