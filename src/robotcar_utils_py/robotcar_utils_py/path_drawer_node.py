#!/usr/bin/env python3
"""path_drawer_node — draw a curve on a canvas and publish it as a path.

Opens a tkinter canvas drawn in the ROBOT BODY frame (top-down): the stroke
origin = the robot position, screen-up = +X (robot forward, along its heading),
screen-left = +Y (robot left). You draw a curve with the mouse; on Publish the
curve is:
  * anchored so its FIRST point sits at the vehicle's current odom position,
  * the remaining points are body-frame offsets ROTATED by the robot's yaw into
    the odom frame (so "up" on the canvas follows wherever the robot is facing),
  * resampled to an even spacing and published as nav_msgs/Path on /vlm_path_truth.

The vehicle drives the path with pure_pursuit_node and stops at the final point
(pure_pursuit_node brakes within its goal_tolerance).

Run (ROS sourced, NOT the venv — needs rclpy + tkinter, no transformers/numpy):
  ros2 run robotcar_utils_py path_drawer
Do not run fake_path_node at the same time (both publish /vlm_path_truth).
"""

import math
import threading
import tkinter as tk

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
from transforms3d.euler import euler2quat, quat2euler


def resample(points, spacing):
    """Resample a polyline (list of (x, y)) to roughly even arc-length spacing.

    Accumulates distance ACROSS segments, so it works regardless of how densely
    the raw points are spaced (drawing slowly produces many sub-`spacing`
    segments; the old version collapsed those to a straight line)."""
    if len(points) < 2 or spacing <= 0.0:
        return list(points)
    out = [points[0]]
    acc = 0.0  # distance accumulated since the last emitted sample
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        x0, y0 = float(x0), float(y0)
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg < 1e-12:
            continue
        # Emit a sample each time the running distance crosses `spacing`.
        while acc + seg >= spacing:
            t = (spacing - acc) / seg
            x0 = x0 + t * (x1 - x0)
            y0 = y0 + t * (y1 - y0)
            out.append((x0, y0))
            seg = math.hypot(x1 - x0, y1 - y0)
            acc = 0.0
        acc += seg
    if out[-1] != points[-1]:
        out.append(points[-1])
    return out


class PathDrawerNode(Node):
    def __init__(self):
        super().__init__('path_drawer_node')
        self.span_m = self.declare_parameter('span_m', 20.0).value
        self.canvas_px = self.declare_parameter('canvas_px', 600).value
        self.spacing_m = self.declare_parameter('spacing_m', 0.15).value
        self.frame_id = self.declare_parameter('frame_id', 'odom').value
        self.grid_m = self.declare_parameter('grid_m', 2.0).value
        # Anchor the drawn path on ground-truth pose (OdometryPublisher) so the
        # path/marker don't drift with wheel odometry.
        self.odom_topic = self.declare_parameter('odom_topic', '/odom_truth').value
        # Built-in relay: also publish the path relocated into the controller's
        # localization frame (/odom_ekf by default), so path_relay_node isn't
        # needed. Must match pure_pursuit's odom_topic (set to /odom for raw
        # wheel control).
        self.wheel_odom_topic = self.declare_parameter('wheel_odom_topic', '/odom_ekf').value
        self.relay_topic = self.declare_parameter('relay_topic', '/vlm_path_odom').value
        self.relay_frame_id = self.declare_parameter('relay_frame_id', 'odom').value

        self.m_per_px = self.span_m / float(self.canvas_px)

        self._lock = threading.Lock()
        self._robot = None   # ground-truth pose (x, y, yaw)
        self._wheel = None   # wheel-odom pose (x, y, yaw)

        self.pub = self.create_publisher(Path, '/vlm_path_truth', 10)
        self.pub_odom = self.create_publisher(Path, self.relay_topic, 10)
        self.create_subscription(Odometry, self.odom_topic, self._on_odom, 10)
        self.create_subscription(Odometry, self.wheel_odom_topic,
                                 self._on_wheel_odom, 10)
        self._last_odom_path = None
        self.create_timer(0.5, self._republish)

    def _on_odom(self, msg):
        p = msg.pose.pose
        o = p.orientation
        with self._lock:
            self._robot = (p.position.x, p.position.y,
                           quat2euler((o.w, o.x, o.y, o.z))[2])

    def _on_wheel_odom(self, msg):
        p = msg.pose.pose
        o = p.orientation
        with self._lock:
            self._wheel = (p.position.x, p.position.y,
                           quat2euler((o.w, o.x, o.y, o.z))[2])

    def robot_pose(self):
        with self._lock:
            return self._robot

    def wheel_pose(self):
        with self._lock:
            return self._wheel

    def _build_path(self, pts, stamp, frame_id, tf=None, yaw_offset=0.0):
        """Build a Path from (x, y) points. tf optionally maps each point to
        another frame; yaw_offset is added to the tangent heading."""
        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = frame_id
        for i, (x, y) in enumerate(pts):
            nx, ny = pts[min(i + 1, len(pts) - 1)]
            px, py = pts[max(i - 1, 0)]
            yaw = math.atan2(ny - py, nx - px) + yaw_offset
            X, Y = tf(x, y) if tf else (x, y)
            qw, qx, qy, qz = euler2quat(0.0, 0.0, yaw)
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = X
            ps.pose.position.y = Y
            ps.pose.orientation.x = qx
            ps.pose.orientation.y = qy
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            path.poses.append(ps)
        return path

    def publish_curve(self, world_points, truth_pose, wheel_pose):
        """Publish the path in the ground-truth frame (/vlm_path_truth) and, when the
        wheel pose is known, its relocation into the wheel-odom frame
        (/vlm_path_odom) — the same transform path_relay_node applies."""
        pts = resample(world_points, self.spacing_m)
        stamp = self.get_clock().now().to_msg()

        truth_path = self._build_path(pts, stamp, self.frame_id)
        self.pub.publish(truth_path)

        n_relay = 0
        if wheel_pose is not None:
            gx, gy, gyaw = truth_pose
            px, py, pyaw = wheel_pose
            dth = pyaw - gyaw
            c, s = math.cos(dth), math.sin(dth)

            def to_wheel(x, y):
                rx, ry = x - gx, y - gy
                return (px + c * rx - s * ry, py + s * rx + c * ry)

            odom_path = self._build_path(pts, stamp, self.relay_frame_id,
                                         tf=to_wheel, yaw_offset=dth)
            self.pub_odom.publish(odom_path)
            self._last_odom_path = odom_path
            n_relay = len(odom_path.poses)

        self.get_logger().info(
            f'Published {len(truth_path.poses)} poses on /vlm_path_truth'
            + (f' and {n_relay} on {self.relay_topic}' if n_relay
               else f' (no {self.wheel_odom_topic} yet — skipped relay)'))
        return len(truth_path.poses), n_relay

    def _republish(self):
        if self._last_odom_path is not None:
            self._last_odom_path.header.stamp = self.get_clock().now().to_msg()
            self.pub_odom.publish(self._last_odom_path)

    # ── tkinter GUI (runs on the main thread) ──────────────────────────────
    def run_gui(self):
        root = tk.Tk()
        root.title('Path Drawer — draw a curve, then Publish')
        W = self.canvas_px

        self.canvas = tk.Canvas(root, width=W, height=W, bg='white',
                                cursor='crosshair')
        self.canvas.grid(row=0, column=0, columnspan=4)
        self._draw_grid()

        self._pixels = []      # current stroke, list of (px, py)

        self.canvas.bind('<ButtonPress-1>', self._on_press)
        self.canvas.bind('<B1-Motion>', self._on_motion)

        tk.Button(root, text='Publish', command=self._on_publish,
                  width=12).grid(row=1, column=0, pady=4)
        tk.Button(root, text='Clear', command=self._on_clear,
                  width=12).grid(row=1, column=1, pady=4)
        tk.Button(root, text='Quit', command=root.destroy,
                  width=12).grid(row=1, column=2, pady=4)

        self.status = tk.Label(root, text='', anchor='w', justify='left')
        self.status.grid(row=2, column=0, columnspan=4, sticky='we')

        self._tick_status(root)
        root.protocol('WM_DELETE_WINDOW', root.destroy)
        root.mainloop()

    def _draw_grid(self):
        W = self.canvas_px
        c = W / 2.0
        step_px = self.grid_m / self.m_per_px
        n = int((W / 2.0) / step_px) + 1

        # grid lines + distance labels (metres from canvas center)
        for k in range(-n, n + 1):
            off = k * step_px
            x = c + off
            if 0 <= x <= W:
                self.canvas.create_line(x, 0, x, W, fill='#eee')
                self.canvas.create_line(0, x, W, x, fill='#eee')
            if k == 0:
                continue
            m = k * self.grid_m
            # +Y is screen-left, so right of center (k>0) is negative Y
            if 0 <= x <= W:
                self.canvas.create_text(x, c + 9, text=f'{-m:g}',
                                        fill='#aaa', font=('TkDefaultFont', 7))
            # +X is screen-up, so below center (k>0) is negative X
            y = c + off
            if 0 <= y <= W:
                self.canvas.create_text(c + 12, y, text=f'{-m:g}',
                                        fill='#aaa', font=('TkDefaultFont', 7))

        # center axes + orientation hints
        self.canvas.create_line(c, 0, c, W, fill='#bbb')
        self.canvas.create_line(0, c, W, c, fill='#bbb')
        self.canvas.create_text(W / 2, 12, text='↑ +X (robot forward)', fill='#888')
        self.canvas.create_text(80, W / 2, text='+Y (robot left) ←', fill='#888')
        self.canvas.create_text(W - 40, W - 12,
                                text=f'grid = {self.grid_m:g} m', fill='#888')

    def _on_press(self, e):
        self._on_clear()  # drop any previous curve + its start marker
        self._pixels = [(e.x, e.y)]
        self.canvas.create_oval(e.x - 4, e.y - 4, e.x + 4, e.y + 4,
                                fill='green', outline='',
                                tags='stroke')  # start = robot

    def _on_motion(self, e):
        if not self._pixels:
            return
        x0, y0 = self._pixels[-1]
        self.canvas.create_line(x0, y0, e.x, e.y, fill='blue', width=2,
                                tags='stroke')
        self._pixels.append((e.x, e.y))

    def _on_clear(self):
        # 'stroke' tags every drawn item (start dot + line segments); the grid
        # is untagged and stays.
        self.canvas.delete('stroke')
        self._pixels = []

    def _pixels_to_world(self, robot):
        """Map the pixel stroke to odom-frame points in the ROBOT BODY frame:
        the stroke origin (first point) = the robot position, screen-up = +X
        (robot forward), screen-left = +Y (robot left). The body offsets are
        rotated by the robot's yaw into the odom frame, so the drawn 'forward'
        follows wherever the robot is currently facing."""
        rx, ry, ryaw = robot
        c, s = math.cos(ryaw), math.sin(ryaw)
        px0, py0 = self._pixels[0]
        out = []
        for px, py in self._pixels:
            fwd = (py0 - py) * self.m_per_px    # screen-up   -> +X body (forward)
            left = (px0 - px) * self.m_per_px   # screen-left -> +Y body (left)
            dx = c * fwd - s * left             # rotate body -> odom
            dy = s * fwd + c * left
            out.append((rx + dx, ry + dy))
        return out

    def _on_publish(self):
        robot = self.robot_pose()
        if robot is None:
            self.status.config(text='No /odom yet — is the sim running?',
                               fg='red')
            return
        if len(self._pixels) < 2:
            self.status.config(text='Draw a curve first (click and drag).',
                               fg='red')
            return
        wheel = self.wheel_pose()
        world = self._pixels_to_world(robot)
        n, n_relay = self.publish_curve(world, robot, wheel)
        if wheel is None:
            self.status.config(
                text=f'Published {n} poses on /vlm_path_truth, but no '
                     f'{self.wheel_odom_topic} yet — controller path skipped.',
                fg='orange')
        else:
            self.status.config(
                text=f'Published {n} poses (/vlm_path_truth) + {n_relay} '
                     f'({self.relay_topic}) at ({robot[0]:.2f}, {robot[1]:.2f}).',
                fg='green')

    def _tick_status(self, root):
        robot = self.robot_pose()
        base = (f'robot: ({robot[0]:.2f}, {robot[1]:.2f})'
                if robot else 'robot: waiting for /odom…')
        cur = self.status.cget('text')
        if cur.startswith('robot:') or cur == '':
            self.status.config(text=base, fg='black')
        root.after(300, lambda: self._tick_status(root))


def main():
    rclpy.init()
    node = PathDrawerNode()
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()
    try:
        node.run_gui()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
