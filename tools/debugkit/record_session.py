#!/usr/bin/env python3
"""Record a debugging session from the running sim into a timestamp-synced folder.

Subscribes to the signals selected via presets in topics.yaml and writes one
session folder under data/sessions/:

    data/sessions/<YYYYmmdd_HHMMSS>[_<label>]/
    ├── meta.yaml                  # what/when/how this session was recorded
    ├── signals/<name>.csv         # scalar signals, one row per message
    ├── paths/<name>/<t_ns>.csv    # one x,y polyline per Path message (+ index.csv)
    └── images/<name>/<t_ns>.jpg   # throttled camera/debug frames    (+ index.csv)

Every row/file carries two clocks:
    t       header.stamp when the message has one (sim time), else receive time
    t_recv  the recorder node clock at arrival
Run the recorder with use_sim_time:=true (the scripts/record_session.sh wrapper
does) so both clocks share the sim epoch and everything is cross-plottable.

Run with the sim up (ROS sourced, system python — no venv needed):
    scripts/record_session.sh --preset core --label baseline
    scripts/record_session.sh --preset core,paths,vlm --duration 120

Analyze afterwards with:  scripts/analyze_session.sh latest
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime

import yaml

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Imu, Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64, String

WS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'topics.yaml')
DEFAULT_OUT = os.path.join(WS_ROOT, 'data', 'sessions')


def quat_to_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


# ---------------------------------------------------------------------------
# Per-kind handlers. CSV kinds define msg_type / columns / row(); the special
# kinds (path, image, depth) write their own per-message files + an index.csv.
# ---------------------------------------------------------------------------

class CsvHandler:
    """Writes signals/<name>.csv with one row per message."""
    msg_type = None
    columns = []          # extractor columns, appended after t,t_recv
    has_header = True     # message carries header.stamp

    def __init__(self, node, name, spec, session_dir):
        self.node = node
        self.name = name
        self.min_interval = float(spec.get('min_interval', 0.0))
        self._last_t = None
        sig_dir = os.path.join(session_dir, 'signals')
        os.makedirs(sig_dir, exist_ok=True)
        self._file = open(os.path.join(sig_dir, name + '.csv'), 'w', newline='')
        self._csv = csv.writer(self._file)
        self._csv.writerow(['t', 't_recv'] + self.columns)
        self.count = 0

    def stamp(self, msg, t_recv):
        if self.has_header:
            s = msg.header.stamp
            t = s.sec + s.nanosec * 1e-9
            if t > 0.0:
                return t
        return t_recv

    def accept(self, t):
        if self.min_interval > 0.0 and self._last_t is not None \
                and (t - self._last_t) < self.min_interval:
            return False
        self._last_t = t
        return True

    def on_msg(self, msg):
        t_recv = self.node.get_clock().now().nanoseconds * 1e-9  # receive time
        t = self.stamp(msg, t_recv) # sim-time if available, else receive time
        if not self.accept(t):
            return
        self._csv.writerow([f'{t:.6f}', f'{t_recv:.6f}'] + self.row(msg))
        self._file.flush()
        self.count += 1

    def row(self, msg):
        raise NotImplementedError

    def close(self):
        self._file.close()


class OdometryHandler(CsvHandler):
    msg_type = Odometry
    columns = ['x', 'y', 'yaw', 'vx', 'vy', 'wz']

    def row(self, m):
        p = m.pose.pose.position
        yaw = quat_to_yaw(m.pose.pose.orientation)
        tw = m.twist.twist
        return [f'{p.x:.4f}', f'{p.y:.4f}', f'{yaw:.5f}',
                f'{tw.linear.x:.4f}', f'{tw.linear.y:.4f}', f'{tw.angular.z:.5f}']


class ImuHandler(CsvHandler):
    msg_type = Imu
    columns = ['yaw', 'wx', 'wy', 'wz', 'ax', 'ay', 'az']

    def row(self, m):
        yaw = quat_to_yaw(m.orientation)
        w, a = m.angular_velocity, m.linear_acceleration
        return [f'{yaw:.5f}', f'{w.x:.5f}', f'{w.y:.5f}', f'{w.z:.5f}',
                f'{a.x:.4f}', f'{a.y:.4f}', f'{a.z:.4f}']


class TwistHandler(CsvHandler):
    msg_type = Twist
    columns = ['v', 'wz']
    has_header = False

    def row(self, m):
        return [f'{m.linear.x:.4f}', f'{m.angular.z:.5f}']


class Float64Handler(CsvHandler):
    msg_type = Float64
    columns = ['value']
    has_header = False

    def row(self, m):
        return [f'{m.data:.4f}']


class StringHandler(CsvHandler):
    msg_type = String
    columns = ['value']
    has_header = False

    def row(self, m):
        return [m.data]


class TapHandler(CsvHandler):
    """Node-internal debug tap (JSON String, see vlm_planner_py/debugtap.py)
    -> taps/<name>.jsonl, one JSON object per line. The payload's own 't'
    (node sim time at flush) is kept as the primary clock; arbitrary keys are
    stored as-is, so new tapped values need no recorder change."""
    msg_type = String
    has_header = False

    def __init__(self, node, name, spec, session_dir):
        self.node = node
        self.name = name
        self.min_interval = float(spec.get('min_interval', 0.0))
        self._last_t = None
        tap_dir = os.path.join(session_dir, 'taps')
        os.makedirs(tap_dir, exist_ok=True)
        self._file = open(os.path.join(tap_dir, name + '.jsonl'), 'w')
        self.count = 0

    def on_msg(self, msg):
        t_recv = self.node.get_clock().now().nanoseconds * 1e-9
        try:
            payload = json.loads(msg.data)
            if not isinstance(payload, dict):
                payload = {'value': payload}
        except ValueError:
            payload = {'raw': msg.data}
        t = float(payload.get('t', t_recv))
        if not self.accept(t):
            return
        payload['t'] = t
        payload['t_recv'] = t_recv
        self._file.write(json.dumps(payload) + '\n')
        self._file.flush()
        self.count += 1


class PathHandler(CsvHandler):
    """Each Path message -> paths/<name>/<t_ns>.csv (x,y rows) + index.csv."""
    msg_type = Path
    columns = ['file', 'n_points', 'frame_id']

    def __init__(self, node, name, spec, session_dir):
        self.dir = os.path.join(session_dir, 'paths', name)
        os.makedirs(self.dir, exist_ok=True)
        self.node = node
        self.name = name
        self.min_interval = float(spec.get('min_interval', 0.0))
        self._last_t = None
        self._file = open(os.path.join(self.dir, 'index.csv'), 'w', newline='')
        self._csv = csv.writer(self._file)
        self._csv.writerow(['t', 't_recv'] + self.columns)
        self.count = 0

    def on_msg(self, msg):
        t_recv = self.node.get_clock().now().nanoseconds * 1e-9
        t = self.stamp(msg, t_recv)
        if not self.accept(t):
            return
        fname = f'{int(round(t * 1e9)):019d}.csv'
        with open(os.path.join(self.dir, fname), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['x', 'y'])
            for ps in msg.poses:
                w.writerow([f'{ps.pose.position.x:.4f}', f'{ps.pose.position.y:.4f}'])
        self._csv.writerow([f'{t:.6f}', f'{t_recv:.6f}',
                            fname, len(msg.poses), msg.header.frame_id])
        self._file.flush()
        self.count += 1


class ImageHandler(CsvHandler):
    """Each Image message -> images/<name>/<t_ns>.jpg + index.csv."""
    msg_type = Image
    columns = ['file', 'width', 'height', 'encoding']
    ext = '.jpg'

    def __init__(self, node, name, spec, session_dir):
        self.dir = os.path.join(session_dir, 'images', name)
        os.makedirs(self.dir, exist_ok=True)
        self.node = node
        self.name = name
        self.min_interval = float(spec.get('min_interval', 0.0))
        self._last_t = None
        self._file = open(os.path.join(self.dir, 'index.csv'), 'w', newline='')
        self._csv = csv.writer(self._file)
        self._csv.writerow(['t', 't_recv'] + self.columns)
        self.count = 0
        # cv_bridge-free decode, same helpers the planner uses (see CLAUDE.md).
        sys.path.insert(0, os.path.join(WS_ROOT, 'src', 'vlm_planner_py'))
        from vlm_planner_py import image_utils
        self._utils = image_utils
        import cv2
        self._cv2 = cv2

    def save(self, msg, path):
        self._cv2.imwrite(path, self._utils.ros_image_to_cv2(msg))

    def on_msg(self, msg):
        t_recv = self.node.get_clock().now().nanoseconds * 1e-9
        t = self.stamp(msg, t_recv)
        if not self.accept(t):
            return
        fname = f'{int(round(t * 1e9)):019d}{self.ext}'
        self.save(msg, os.path.join(self.dir, fname))
        self._csv.writerow([f'{t:.6f}', f'{t_recv:.6f}',
                            fname, msg.width, msg.height, msg.encoding])
        self._file.flush()
        self.count += 1


class DepthHandler(ImageHandler):
    """Depth Image -> float32 .npy in metres (heavier; throttle in topics.yaml)."""
    ext = '.npy'

    def save(self, msg, path):
        import numpy as np
        np.save(path, self._utils.ros_depth_to_cv2(msg).astype('float32'))


KIND_HANDLERS = {
    'odometry': OdometryHandler,
    'imu': ImuHandler,
    'twist': TwistHandler,
    'float64': Float64Handler,
    'string': StringHandler,
    'path': PathHandler,
    'image': ImageHandler,
    'depth': DepthHandler,
    'tap': TapHandler,
}


# ---------------------------------------------------------------------------
# Config / preset resolution
# ---------------------------------------------------------------------------

def resolve_selection(cfg, presets, extra_signals, exclude):
    """Expand preset names (recursively) + explicit signals into {name: spec}."""
    catalog = cfg.get('signals', {})
    preset_map = cfg.get('presets', {})
    selected, stack = [], list(presets)
    seen_presets = set()
    while stack:
        entry = stack.pop(0)
        if entry in preset_map:
            if entry not in seen_presets:
                seen_presets.add(entry)
                stack = list(preset_map[entry]) + stack
        elif entry in catalog:
            if entry not in selected:
                selected.append(entry)
        else:
            raise SystemExit(f"Unknown preset or signal '{entry}' "
                             f"(known presets: {sorted(preset_map)}; "
                             f"signals: {sorted(catalog)})")
    for s in extra_signals:
        if s not in catalog:
            raise SystemExit(f"Unknown signal '{s}' (known: {sorted(catalog)})")
        if s not in selected:
            selected.append(s)
    selected = [s for s in selected if s not in exclude]
    if not selected:
        raise SystemExit('Nothing selected to record.')
    return {name: catalog[name] for name in selected}


def ros2_param_set(node_name, param, yaml_val):
    """Best-effort live `ros2 param set`; returns True on success."""
    try:
        out = subprocess.run(
            ['ros2', 'param', 'set', node_name, param, yaml_val],
            capture_output=True, text=True, timeout=10.0)
        ok = 'successful' in (out.stdout + out.stderr).lower()
    except Exception:
        ok = False
    if not ok:
        print(f'WARNING: could not set {node_name} {param}={yaml_val} '
              '(is the node running?)', file=sys.stderr)
    return ok


def set_planner_lane_dump(value):
    """Point /vlm_planner's lane_cloud_dump_dir at `value` ('' to disable)."""
    return ros2_param_set('/vlm_planner', 'lane_cloud_dump_dir',
                          value if value else "''")


def toggle_debug_taps(specs, on):
    """Enable/disable the debug_tap param of every node whose tap we record.
    Tap topics follow the /debug/<node_name> convention (debugtap.py)."""
    done = set()
    for spec in specs.values():
        topic = spec['topic']
        if spec['kind'] == 'tap' and topic.startswith('/debug/') and topic not in done:
            done.add(topic)
            ros2_param_set('/' + topic[len('/debug/'):], 'debug_tap',
                           'true' if on else 'false')


def git_commit():
    try:
        return subprocess.check_output(
            ['git', '-C', WS_ROOT, 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


# ---------------------------------------------------------------------------

class SessionRecorder(Node):
    def __init__(self, session_dir, specs, duration):
        super().__init__('session_recorder')
        self.session_dir = session_dir
        self.handlers = {}
        for name, spec in specs.items():
            cls = KIND_HANDLERS[spec['kind']]
            h = cls(self, name, spec, session_dir)
            self.handlers[name] = h
            self.create_subscription(h.msg_type, spec['topic'], h.on_msg, 10)
            self.get_logger().info(
                f"recording {spec['topic']} ({spec['kind']}) -> {name}")
        if duration > 0:
            self.get_logger().info(f'auto-stop after {duration:.0f}s (wall clock)')
        self.create_timer(5.0, self._status)
        self.get_logger().info(f'session dir: {session_dir}  (Ctrl-C to stop)')

    def _status(self):
        counts = ', '.join(f'{n}:{h.count}' for n, h in self.handlers.items())
        self.get_logger().info(f'msgs  {counts}')

    def counts(self):
        return {n: h.count for n, h in self.handlers.items()}

    def close(self):
        for h in self.handlers.values():
            h.close()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--preset', default='core',
                    help='comma-separated preset names from topics.yaml (default: core)')
    ap.add_argument('--signals', default='',
                    help='comma-separated extra signal names to add')
    ap.add_argument('--exclude', default='',
                    help='comma-separated signal names to drop from the selection')
    ap.add_argument('--label', default='',
                    help='suffix for the session folder name, e.g. the test scenario')
    ap.add_argument('--duration', type=float, default=0.0,
                    help='stop automatically after N seconds (0 = run until Ctrl-C)')
    ap.add_argument('--out', default=DEFAULT_OUT,
                    help=f'sessions root (default: {DEFAULT_OUT})')
    ap.add_argument('--config', default=DEFAULT_CONFIG,
                    help='signal catalog yaml (default: tools/debugkit/topics.yaml)')
    ap.add_argument('--notes', default='', help='free-text note stored in meta.yaml')
    ap.add_argument('--lane-dump', action='store_true',
                    help='redirect the planner lane-cloud dump into this session '
                         '(<session>/lane_cloud/lane_cloud.jsonl; sets /vlm_planner '
                         'lane_cloud_dump_dir live, restores it on stop)')
    ap.add_argument('--wait-for-go', action='store_true',
                    help='print the selected signals and block on Enter before '
                         'subscribing/recording anything (Ctrl-C aborts cleanly, '
                         'no session folder created). For interactive/manual runs; '
                         'leave off for headless/automated callers (run_scenario.sh etc).')
    args, ros_args = ap.parse_known_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    presets = [p for p in args.preset.split(',') if p]
    extra = [s for s in args.signals.split(',') if s]
    exclude = {s for s in args.exclude.split(',') if s}
    specs = resolve_selection(cfg, presets, extra, exclude)

    if args.wait_for_go:
        print('Selected signals:')
        for n, s in specs.items():
            print(f"  {n:20s} {s['topic']} ({s['kind']})")
        try:
            input('Press Enter to start recording (Ctrl-C to abort)... ')
        except KeyboardInterrupt:
            print('\naborted — nothing recorded.')
            return

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    name = stamp + (f'_{args.label}' if args.label else '')
    session_dir = os.path.join(args.out, name)
    os.makedirs(session_dir, exist_ok=True)

    meta = {
        'label': args.label,
        'started': datetime.now().isoformat(timespec='seconds'),
        'presets': presets,
        'extra_signals': extra,
        'exclude': sorted(exclude),
        'signals': {n: {'topic': s['topic'], 'kind': s['kind']}
                    for n, s in specs.items()},
        'git_commit': git_commit(),
        'notes': args.notes,
        'argv': sys.argv[1:],
    }
    with open(os.path.join(session_dir, 'meta.yaml'), 'w') as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    if args.lane_dump:
        lane_dir = os.path.join(session_dir, 'lane_cloud')
        meta['lane_dump'] = set_planner_lane_dump(lane_dir)
    toggle_debug_taps(specs, True)

    rclpy.init(args=ros_args)
    node = SessionRecorder(session_dir, specs, args.duration)
    try:
        if args.duration > 0:
            # Wall-clock deadline via spin_once: sim-time independent, and
            # avoids rclpy.shutdown()-from-a-callback, which can deadlock.
            import time
            end = time.monotonic() + args.duration
            while rclpy.ok() and time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.2)
            node.get_logger().info('duration reached — stopping.')
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if args.lane_dump and meta.get('lane_dump'):
            set_planner_lane_dump('')   # stop the planner writing into the session
        toggle_debug_taps(specs, False)
        meta['ended'] = datetime.now().isoformat(timespec='seconds')
        meta['message_counts'] = node.counts()
        with open(os.path.join(session_dir, 'meta.yaml'), 'w') as f:
            yaml.safe_dump(meta, f, sort_keys=False)
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print(f'\nSession saved: {session_dir}')
        print(f'Analyze with:  scripts/analyze_session.sh {name}')


if __name__ == '__main__':
    main()
