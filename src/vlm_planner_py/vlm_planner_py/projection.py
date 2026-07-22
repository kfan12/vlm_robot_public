import numpy as np
import math

def pixel_to_camera_frame(pixel_x, pixel_y, depth_m, fx, cx, cy):
    """Back-project (u,v)+depth to 3-D point in camera optical frame (meters)."""
    if depth_m <= 0.05 or depth_m > 15.0 or math.isnan(depth_m) or math.isinf(depth_m):
        return None

    x_cam = (pixel_x - cx) * depth_m / fx
    y_cam = (pixel_y - cy) * depth_m / fx  # square pixels assumed (fy = fx)
    z_cam = depth_m
    return (x_cam, y_cam, z_cam)

def camera_to_base_link(point_cam,camera_to_base_transform):
    """
    Convert a point from camera frame to base_link frame.
    camera_to_base_tf: a 4x4 numpy homogeneous transform matrix
    Returns (x, y, z) in base_link frame.
    """
    p = np.array([point_cam[0], point_cam[1], point_cam[2], 1.0])  # homogeneous coordinates
    transformed = camera_to_base_transform @ p  # matrix multiplication
    return (transformed[0], transformed[1], transformed[2])

def build_camera_to_base_transform(tf_buffer, camera_frame, base_link_frame,stamp):
    """
    Build a homogeneous transform matrix from camera frame to base_link frame using tf_buffer.
    Returns a 4x4 numpy array.
    """
    import tf2_ros
    import tf2_geometry_msgs
    from rclpy.duration import Duration

    try:
        t = tf_buffer.lookup_transform(base_link_frame, 
                                       camera_frame, 
                                       stamp, 
                                       timeout=Duration(seconds=0.1))
    except Exception as e:
        print(f"TF lookup failed: {e}")
        return None
    
    tx = t.transform.translation.x
    ty = t.transform.translation.y
    tz = t.transform.translation.z
    rx = t.transform.rotation.x
    ry = t.transform.rotation.y
    rz = t.transform.rotation.z
    rw = t.transform.rotation.w

    # Convert quaternion to rotation matrix
    R = np.array([  # 3x3 rotation matrix
        [1.0 - 2.0 * (ry * ry + rz * rz), 2.0 * (rx * ry - rz * rw), 2.0 * (rx * rz + ry * rw)],
        [2.0 * (rx * ry + rz * rw), 1.0 - 2.0 * (rx * rx + rz * rz), 2.0 * (ry * rz - rx * rw)],
        [2.0 * (rx * rz - ry * rw), 2.0 * (ry * rz + rx * rw), 1.0 - 2.0 * (rx * rx + ry * ry)],
    ])
    T = np.eye(4)  # 4x4 homogeneous transform
    T[:3, :3] = R
    T[0, 3] = tx
    T[1, 3] = ty
    T[2, 3] = tz
    
    return T

def project_pixel_to_base_link(
        pixel_list,
        depth_img_np,
        fx, cx, cy,
        camera_to_base_transform,
        img_h, img_w
):
    """Project pixel list to base_link (x, y) points using corrected intrinsics.

    LEGACY single-pixel projection: samples depth at exactly one pixel, so it is
    unreliable when the pixel falls in a gap (e.g. a centerline midpoint between
    two cones, where the depth ray hits the background). Kept for the legacy
    image-space centerline path; new code should use project_cones_to_base_link.
    """
    if camera_to_base_transform is None:
        return []

    points = []
    for (px, py) in pixel_list:
        px_c = int(max(0, min(img_w - 1, px)))
        py_c = int(max(0, min(img_h - 1, py)))
        depth = float(depth_img_np[py_c, px_c])
        pt_cam = pixel_to_camera_frame(px_c, py_c, depth, fx, cx, cy)
        if pt_cam is None:
            continue
        pt_base = camera_to_base_link(pt_cam, camera_to_base_transform)
        if pt_base[0] > 0.1 and pt_base[0] < 10.0 and abs(pt_base[1]) < 3.0:
            points.append((pt_base[0], pt_base[1]))
    return points


def project_pixels_to_base_link(
        pixel_list,
        depth_img_np,
        fx, cx, cy,
        camera_to_base_transform,
        img_h, img_w,
        depth_window=3,
        x_min=0.1, x_max=12.0, y_max=4.0,
        z_min=-0.3, z_max=0.3,
        return_pixels=False,
        with_z=False,
):
    """Project each pixel of a detection list to a metric (x, y) point in
    base_link. General-purpose despite the historic "cones" name it carried
    until 2026-07-10: used for cone centroids, line-mask pixels, red goal blobs
    and sign boards alike (the old name survives as an alias below).

    Unlike project_pixel_to_base_link, the depth is sampled as the MEDIAN over a
    small window centered on the pixel, so we read the object surface rather
    than a single (possibly-gap) pixel. Returns one (x, y) per pixel whose depth
    is valid and whose base_link position is in front of the robot and within
    lane bounds. The returned list is the raw set of positions (unordered, sides
    mixed) — the centerline builder decides left/right and ordering.

    HEIGHT GATE (z_min, z_max): only blobs whose projected base_link height lies in
    this band are kept. base_link sits at ~ground level, so cones (centre ~0.1 m,
    top ~0.2 m) pass while elevated same-coloured objects — the roadside traffic
    sign boards at ~0.55 m — are rejected. This is what stops the yellow sign
    boards registering as cones and the red STOP board registering as the goal.

    If return_pixels is True, each surviving entry is (x, y, px, py) — the metric
    point plus the source centroid pixel — so callers can draw a detection overlay.
    Otherwise entries are (x, y) as before.
    """
    if camera_to_base_transform is None:
        return []

    half = max(0, int(depth_window) // 2)
    points = []
    for (px, py) in pixel_list:
        px_c = int(max(0, min(img_w - 1, px)))
        py_c = int(max(0, min(img_h - 1, py)))

        y0, y1 = max(0, py_c - half), min(img_h, py_c + half + 1)
        x0, x1 = max(0, px_c - half), min(img_w, px_c + half + 1)
        win = np.asarray(depth_img_np[y0:y1, x0:x1], dtype=float).ravel()
        win = win[np.isfinite(win)]
        win = win[(win > 0.05) & (win < 15.0)]
        if win.size == 0:
            continue
        depth = float(np.median(win))

        pt_cam = pixel_to_camera_frame(px_c, py_c, depth, fx, cx, cy)
        if pt_cam is None:
            continue
        pt_base = camera_to_base_link(pt_cam, camera_to_base_transform)
        if (x_min < pt_base[0] < x_max and abs(pt_base[1]) < y_max
                and z_min < pt_base[2] < z_max):
            if return_pixels and with_z:
                points.append((pt_base[0], pt_base[1], pt_base[2], px_c, py_c))
            elif return_pixels:
                points.append((pt_base[0], pt_base[1], px_c, py_c))
            else:
                points.append((pt_base[0], pt_base[1]))
    return points


# Legacy alias (pre-2026-07-10 name): older notebooks/plan docs import this.
project_cones_to_base_link = project_pixels_to_base_link





