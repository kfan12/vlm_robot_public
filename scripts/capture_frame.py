#!/usr/bin/env python3
"""Capture RGB (+depth) frames from the running sim, for offline VLM tests.

Two modes:

  SINGLE SHOT (default, --interval 0): wait for one RGB + one depth frame, save
    <outdir>/frame_rgb.png   — BGR image (VLM prompt tests, R2)
    <outdir>/frame_depth.npy — float32 depth in metres (floor-depth probe, R1)
  and print the depth valid-fraction, then exit.

  PERIODIC (--interval > 0): keep the latest frame and save one every INTERVAL
    seconds with a sequence number, until Ctrl-C. RGB only by default; add
    --depth to also dump depth .npy each tick. Used by the demo pane to build a
    dataset of real S-world frames during a run.

Run with the sim up (ROS sourced, system python — no venv needed):
    python3 scripts/capture_frame.py                 # single shot -> ./captures/
    python3 scripts/capture_frame.py --interval 1.0  # 1 frame/sec -> ./captures/seq_<ts>/
"""
import os
import sys
import argparse
from datetime import datetime

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'vlm_planner_py'))
from vlm_planner_py.image_utils import ros_image_to_cv2, ros_depth_to_cv2  # noqa: E402


def _depth_summary(depth):
    d = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(d)
    valid = finite & (d > 0.05) & (d < 15.0)
    if valid.any():
        return (f'depth valid {valid.mean():.2%} (finite {finite.mean():.2%}, '
                f'range [{d[valid].min():.2f}, {d[valid].max():.2f}] m)')
    return 'depth has NO valid samples in (0.05, 15) m — see R1.'


class FrameGrabber(Node):
    def __init__(self, outdir, interval, save_depth, max_frames):
        super().__init__('frame_grabber')
        self.outdir = outdir
        self.interval = interval
        self.save_depth = save_depth
        self.max_frames = max_frames
        self.rgb = None
        self.depth = None
        self.seq = 0
        self.create_subscription(Image, '/camera/front/image_raw', self._rgb_cb, 10)
        self.create_subscription(Image, '/camera/front/depth/image_raw', self._depth_cb, 10)

        if interval > 0.0:
            os.makedirs(self.outdir, exist_ok=True)
            self.create_timer(interval, self._tick)
            limit = f'{max_frames} frames max' if max_frames > 0 else 'unlimited'
            self.get_logger().info(
                f'Periodic capture every {interval:.2f}s -> {self.outdir} '
                f'(RGB{"+depth" if save_depth else ""}, {limit}). Ctrl-C to stop.')
        else:
            self.get_logger().info('Waiting for one RGB + one depth frame...')

    def _rgb_cb(self, msg):
        self.rgb = ros_image_to_cv2(msg)

    def _depth_cb(self, msg):
        self.depth = ros_depth_to_cv2(msg)

    # --- periodic mode ---
    def _tick(self):
        if self.rgb is None:
            self.get_logger().warn('No RGB frame yet; is the camera publishing?',
                                   throttle_duration_sec=5.0)
            return
        rgb_path = os.path.join(self.outdir, f'frame_{self.seq:05d}.png')
        cv2.imwrite(rgb_path, self.rgb)
        msg = f'saved {rgb_path}'
        if self.save_depth and self.depth is not None:
            depth_path = os.path.join(self.outdir, f'frame_{self.seq:05d}_depth.npy')
            np.save(depth_path, self.depth.astype(np.float32))
            msg += f' + depth'
        self.get_logger().info(msg)
        self.seq += 1
        if self.max_frames > 0 and self.seq >= self.max_frames:
            self.get_logger().info(
                f'Reached max {self.max_frames} frames -> {self.outdir}. Stopping.')
            rclpy.shutdown()

    # --- single-shot mode ---
    def have_pair(self):
        return self.rgb is not None and self.depth is not None

    def save_single(self):
        os.makedirs(self.outdir, exist_ok=True)
        rgb_path = os.path.join(self.outdir, 'frame_rgb.png')
        depth_path = os.path.join(self.outdir, 'frame_depth.npy')
        cv2.imwrite(rgb_path, self.rgb)
        np.save(depth_path, self.depth.astype(np.float32))
        self.get_logger().info(f'Saved {rgb_path}')
        self.get_logger().info(f'Saved {depth_path}')
        self.get_logger().info(_depth_summary(self.depth))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outdir', nargs='?', default='captures')
    ap.add_argument('--interval', type=float, default=0.0,
                    help='seconds between saves; 0 = single shot (default)')
    ap.add_argument('--depth', action='store_true',
                    help='in periodic mode, also save depth .npy each tick')
    ap.add_argument('--max-frames', type=int, default=50,
                    help='periodic mode: stop after this many frames (0 = unlimited)')
    ap.add_argument('--timeout', type=float, default=15.0,
                    help='single-shot only: give up after this many seconds')
    args = ap.parse_args()

    outdir = args.outdir
    # In periodic mode, isolate each run in its own timestamped subdir under the
    # base dir so repeated sim runs don't overwrite each other.
    if args.interval > 0.0:
        outdir = os.path.join(outdir, 'seq_' + datetime.now().strftime('%Y%m%d_%H%M%S'))

    rclpy.init()
    node = FrameGrabber(outdir, args.interval, args.depth, args.max_frames)
    try:
        if args.interval > 0.0:
            rclpy.spin(node)
        else:
            end = node.get_clock().now().nanoseconds * 1e-9 + args.timeout
            while rclpy.ok() and not node.have_pair():
                rclpy.spin_once(node, timeout_sec=0.2)
                if node.get_clock().now().nanoseconds * 1e-9 > end:
                    node.get_logger().error(
                        'Timed out. Is the sim running and publishing the camera topics?')
                    node.destroy_node()
                    rclpy.shutdown()
                    sys.exit(1)
            node.save_single()
    except KeyboardInterrupt:
        node.get_logger().info(f'Stopped. {node.seq} frames saved to {outdir}.')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
