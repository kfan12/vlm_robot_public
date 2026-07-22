"""Path construction helpers: turn sparse cone-centerline points into a dense,
evenly-spaced robot->cones path with per-pose headings.

Everything here works in the ROBOT frame (x forward, y left) and is ROS-free so
it can be unit-tested without a node. The caller (vlm_node) builds the
nav_msgs/Path from the (x, y, yaw) tuples and transforms it into odom.
"""

import math
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import rclpy

def smooth_points(points, window = 3):
    """
    Smooth a path using a moving average filter, 
    returns a new list of smoothed (x, y) tuples.
    """
    if len(points) < window:
        return points
    arr = np.array(points)
    smoothed = np.copy(arr)

    half_window = window // 2
    for i in range(half_window, len(points) - half_window):
        smoothed[i] = np.mean(arr[i - half_window:i + half_window + 1], axis=0)

    return [(float(p[0]), float(p[1])) for p in smoothed]

def fit_polyline(points, degree=3):
    """Stateless de-noise: fit y = f(x) as a low-order polynomial in base_link
    and re-sample it over the same x-range. Many noisy points -> a few poly
    params IS an averaging, so per-frame depth jitter (the source of the path
    oscillation) is smoothed out with ZERO temporal lag and no cross-frame state
    -- unlike an EMA, which would lag and trail the lane through the curve.

    Safe because within the near horizon the boundary is x-monotonic (x forward),
    so y as a function of x is single-valued. Falls back to the input unchanged
    when there are too few points or the x-span is degenerate (e.g. the 2-point
    creep path), so callers can apply it unconditionally.
    """
    if len(points) < 4:
        return points
    arr = np.asarray(points, dtype=float)
    xs, ys = arr[:, 0], arr[:, 1]
    x0, x1 = float(xs.min()), float(xs.max())
    if x1 - x0 < 1e-3:
        return points
    deg = min(int(degree), len(points) - 1)
    try:
        coeffs = np.polyfit(xs, ys, deg)
    except (np.linalg.LinAlgError, ValueError):
        return points
    xq = np.linspace(x0, x1, len(points))
    yq = np.polyval(coeffs, xq)
    return [(float(x), float(y)) for x, y in zip(xq, yq)]

def compute_headings(points):
    """
    Compute headings (yaw angles) for a list of (x, y) points.
    Returns a list of yaw angles in radians, where yaw = atan2(dy, dx) toward the next point.
    """
    headings = []
    for i in range(len(points) - 1):
        dx = points[i + 1][0] - points[i][0]
        dy = points[i + 1][1] - points[i][1]
        headings.append(math.atan2(dy, dx))
    # For the last point, repeat the previous heading
    headings.append(headings[-1] if headings else 0.0)
    return headings

def resample_uniform(points, spacing=0.15):
    """
    Resample a list of (x, y) points to have uniform arc length.
    Returns a new list of (x, y) tuples with approximately `spacing` between consecutive points.
    """
    if len(points) < 2:
        return points

    resampled = [points[0]]
    carry = 0.0  # leftover distance carried into the next segment
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        seg_length = math.hypot(x1 - x0, y1 - y0)
        if seg_length < 1e-9:
            continue
        ux, uy = (x1 - x0) / seg_length, (y1 - y0) / seg_length # unit vector along the segment
        d = spacing - carry  # start at the next spacing point along the segment
        while d <= seg_length:
            resampled.append((x0 + ux * d, y0 + uy * d))
            d += spacing
        carry = seg_length - (d - spacing)

    # Always include the true endpoint
    if math.hypot(resampled[-1][0] - points[-1][0], resampled[-1][1] - points[-1][1]) > 1e-6:
        resampled.append(points[-1])
    return resampled

def validate_path(points, min_dist = 0.1, max_dist = 10.0, max_lateral = 5.0):
    """
    Validate a path represented as a list of (x, y) points.
    Returns (is_valid, reason) where is_valid is True if the path is valid, and reason is a string explaining why it's invalid if not.
    """
    if len(points) < 2:
        # output the points for debugging
        print(f"validate_path: points={points}")
        return False, "Path has too few points"
    

    total_length = sum(math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
                        for i in range(len(points) - 1))
    
    if total_length < min_dist:
        return False, f"Path is too short ({total_length:.2f}m)"
    if total_length > max_dist:
        return False, f"Path is too long ({total_length:.2f}m)"
    
    for x, y in points:
        if abs(y) > max_lateral:
            return False, f"Point ({x:.2f}, {y:.2f}) is too far laterally (|y| > {max_lateral})"
        if x < -0.1:
            return False, f"Point ({x:.2f}, {y:.2f}) is behind the robot (x < 0)"
    
    return True, "Path is valid"

def build_path_msg_with_headings(points, frame_id, stamp):
    """
    Build a nav_msgs/Path message from a list of (x, y) points and their computed headings.
    Returns a Path message with PoseStamped entries.
    """
    path_msg = Path()
    path_msg.header.frame_id = frame_id
    path_msg.header.stamp = stamp 


    headings = compute_headings(points)

    try:
        import tf_transformations
        yaw_to_quat = lambda yaw: tf_transformations.quaternion_from_euler(0, 0, yaw)
    except ImportError:
        # Fallback: a yaw-only quaternion is just (0, 0, sin(yaw/2), cos(yaw/2)).
        yaw_to_quat = lambda yaw: (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))

    for (x, y), yaw in zip(points, headings):
        pose_stamped = PoseStamped()
        pose_stamped.header = path_msg.header
        pose_stamped.pose.position.x = x
        pose_stamped.pose.position.y = y
        pose_stamped.pose.position.z = 0.0
        q = yaw_to_quat(yaw)
        pose_stamped.pose.orientation.x = q[0]
        pose_stamped.pose.orientation.y = q[1]
        pose_stamped.pose.orientation.z = q[2]
        pose_stamped.pose.orientation.w = q[3]
        path_msg.poses.append(pose_stamped)


    return path_msg

  






"""
temporary code for path smoothing and resampling, to be integrated into the main path planning pipeline.
"""


def _resample_polyline(points, spacing):
    """Resample a polyline at fixed arc-length `spacing` via linear interpolation.

    points: list of (x, y). Returns a list of (x, y) with ~`spacing` between
    consecutive points; the original first and last points are preserved.
    """
    if len(points) < 2:
        return list(points)

    out = [points[0]]
    carry = 0.0  # leftover distance carried into the next segment
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg < 1e-9:
            continue
        ux, uy = (x1 - x0) / seg, (y1 - y0) / seg
        d = spacing - carry
        while d <= seg:
            out.append((x0 + ux * d, y0 + uy * d))
            d += spacing
        carry = seg - (d - spacing)

    # Always include the true endpoint (the farthest cone point).
    if math.hypot(out[-1][0] - points[-1][0], out[-1][1] - points[-1][1]) > 1e-6:
        out.append(points[-1])
    return out


def _headings(points):
    """Per-point tangent yaw = atan2(dy, dx) toward the next point.

    The last point copies the previous heading (no successor to point at)."""
    if not points:
        return []
    yaws = []
    for i in range(len(points) - 1):
        dx = points[i + 1][0] - points[i][0]
        dy = points[i + 1][1] - points[i][1]
        yaws.append(math.atan2(dy, dx))
    yaws.append(yaws[-1] if yaws else 0.0)
    return yaws


def build_path(cone_points, spacing=0.3, include_robot=True):
    """Formulate a dense robot->cones path in the robot frame.

    cone_points  : list of (x, y) centerline points in the robot frame
                   (x forward, y left), any order — sorted here by forward dist.
    spacing      : arc-length spacing of the resampled path [m].
    include_robot: prepend the robot origin (0, 0) so the path starts on the car
                   and the controller always has a nearby closest-point.

    Returns: list of (x, y, yaw) poses in the robot frame. The caller converts
    yaw -> quaternion, builds the Path, then transforms it into odom.
    """
    pts = sorted(cone_points, key=lambda p: p[0])  # near -> far by forward dist
    if include_robot:
        pts = [(0.0, 0.0)] + pts

    # Drop consecutive duplicates (e.g. robot origin == a cone at ~0).
    cleaned = [pts[0]]
    for p in pts[1:]:
        if math.hypot(p[0] - cleaned[-1][0], p[1] - cleaned[-1][1]) > 1e-6:
            cleaned.append(p)

    if len(cleaned) < 2:
        return [(cleaned[0][0], cleaned[0][1], 0.0)] if cleaned else []

    dense = _resample_polyline(cleaned, spacing)
    yaws = _headings(dense)
    return [(x, y, yaw) for (x, y), yaw in zip(dense, yaws)]


if __name__ == '__main__':
    # Quick self-test: two far cone points -> dense path starting at the robot.
    demo = build_path([(5.0, 0.5), (8.0, -0.3)], spacing=0.5)
    print(f'{len(demo)} poses, first={demo[0]}, last={demo[-1]}')
    # Expect: first ~ (0,0,~0.1rad), evenly spaced ~0.5 m apart, last ~ (8,-0.3)
