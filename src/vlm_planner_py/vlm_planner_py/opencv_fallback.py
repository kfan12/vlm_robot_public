import cv2
import numpy as np
import math
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from vlm_planner_py.image_utils import ros_image_to_cv2, ros_depth_to_cv2  

# Yellow HSV range for path detection (tune as needed according to the Gazebo material color)
#
# NOTE on the V (value/brightness) floors: under WSL software GL rendering the
# sign/cone materials render DARK -- the yellow sign board measures S~250 but V
# tops out at ~99 (median ~75). A V>=100 floor rejected the whole sign, giving
# blobs=0. The floor is kept low (40) because the world is otherwise grayscale
# (S~0 on road/sky), so only the saturated signs/cones can clear S>=100 -- a low
# V floor can't admit the gray background as a false positive.
YELLOW_LOWER_HSV = np.array([20, 100, 40])
YELLOW_UPPER_HSV = np.array([30, 255, 255])

# Red wraps around the HSV hue seam (0/180), so it needs two ranges.
RED_LOWER_HSV1 = np.array([0, 100, 40])
RED_UPPER_HSV1 = np.array([10, 255, 255])
RED_LOWER_HSV2 = np.array([170, 100, 40])
RED_UPPER_HSV2 = np.array([180, 255, 255])

def detect_yellow_cones(rgb_image, area_min=20.0, area_max=20000.0):
    """Detect yellow cones in the RGB image and return their pixel coordinates.

    area_min/area_max bound the contour area in px^2. Defaults are relaxed vs the
    original 50..5000 so that FAR (small) cones survive — curve tracking needs as
    many rows in view as possible (see docs/detection_curve_tracking_plan.md, F5).
    """
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)  # Convert to HSV color space for better color segmentation
    mask = cv2.inRange(hsv_image, YELLOW_LOWER_HSV, YELLOW_UPPER_HSV)  # Create a binary mask where yellow colors are white and the rest are black

    # Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Kernel for morphological operations
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours in the mask

    centroids = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < area_min or area > area_max: # Filter contours by area to remove noise and irrelevant objects
            continue
        M = cv2.moments(contour)  # Calculate moments of the contour to find the centroid
        if M['m00'] ==0:
            continue
        cX = int(M['m10'] / M['m00'])  # X coordinate of the centroid
        cY = int(M['m01'] / M['m00'])  # Y coordinate of the centroid
        centroids.append((cX, cY))
    return centroids

def detect_red_cones(rgb_image, area_min=20.0, area_max=20000.0):
    """Detect the red GOAL cone(s) and return their pixel centroids.

    Same pipeline as detect_yellow_cones but with the two-range red HSV mask.
    Used to creep toward / stop at the goal once the yellow lane runs out.
    """
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(cv2.inRange(hsv_image, RED_LOWER_HSV1, RED_UPPER_HSV1),
                           cv2.inRange(hsv_image, RED_LOWER_HSV2, RED_UPPER_HSV2))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < area_min or area > area_max:
            continue
        M = cv2.moments(contour)
        if M['m00'] == 0:
            continue
        centroids.append((int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])))
    return centroids


def detect_track_lines(rgb_image, val_min=150, sat_max=70, sample_stride=8,
                       area_min_px=50, roi_top_px=0):
    """Detect the LIGHT-GRAY boundary lines on the DARK road and return a
    subsampled list of (px, py) mask pixels along them.

    The markings are bright + near-colourless (light gray/white) while the road
    is dark, so the mask is "high value, low saturation": V >= val_min and
    S <= sat_max. Unlike the cone detector this does NOT centroid blobs — a line
    is a long thin region, so we return many pixels SAMPLED along the mask
    (every sample_stride-th mask pixel). Those pixels are projected to metric
    base_link by the caller (project_pixels_to_base_link) and turned into a
    centerline by centerline.centerline_from_line_cloud.

    A tiny-area open removes speckle; sample_stride caps the cloud size so the
    downstream clustering stays cheap. area_min_px was raised 15 -> 50
    (2026-07-10): a 15-49 px far-range blob survived the 15-px gate, projected
    to a stray mid-lane point cluster and kinked the centerline (see
    docs/knowledge_base.md; centerline._drop_stray_chains is the second guard).

    roi_top_px excludes all rows above it (the sky) BEFORE the color mask runs,
    so a bright/desaturated sky can never match the same "light gray" criteria
    as a real ground marking — unlike the downstream depth/height gates, this
    exclusion is geometric and doesn't depend on the sky returning invalid depth
    (2026-07-18: the procedural <sky> in track_lines.world.sdf was matching the
    mask and, near the horizon, occasionally surviving the depth+height gates
    too, since the sensor's far clip (15.0 m) coincides with the depth filter's
    upper bound). Rows are cropped, not just masked out, so the returned pixel
    coordinates below still line up with the caller's full-image intrinsics.
    """
    top = max(0, min(int(roi_top_px), rgb_image.shape[0]))
    cropped = rgb_image[top:, :]
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, int(val_min)])
    upper = np.array([180, int(sat_max), 255])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Drop tiny connected components (speckle) below area_min_px.
    if area_min_px > 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = np.zeros_like(mask)
        for lbl in range(1, num):
            if stats[lbl, cv2.CC_STAT_AREA] >= area_min_px:
                keep[labels == lbl] = 255
        mask = keep

    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return []
    step = max(1, int(sample_stride))
    return [(int(x), int(y) + top) for x, y in zip(xs[::step], ys[::step])]


def split_left_right_cones(centroids, image_width):
    """LEGACY (image-space). Split cones into left/right by ABSOLUTE image center.

    Not curve-aware: on a bend the lane shifts to one half of the frame and this
    mis-splits. Superseded by the heading-relative gate-walk in centerline.py.
    Kept for the legacy run_opencv_planner path and reference.
    """
    left_cones = []
    right_cones = []
    for (cX, cY) in centroids:
        if cX < image_width / 2:
            left_cones.append((cX, cY))  # Cone is on the left half of the image
        else:
            right_cones.append((cX, cY))  # Cone is on the right half of the image
    return left_cones, right_cones

def estimate_centerline_pixels(left_cones, right_cones,img_width,img_height,n_points=10):
    """LEGACY (image-space). Estimate centerline by averaging cone PIXELS and
    appending a hardcoded vanishing point.

    Two structural problems superseded by the metric-space stage
    (project_pixels_to_base_link + centerline.build_centerline_gatewalk):
      - pixel averaging + a single depth sample at the midpoint reads the gap
        between cones, not a real surface;
      - the (w/2, h/4) vanishing point biases the path straight on curves.
    Kept for the legacy run_opencv_planner path and reference.
    """
    centerline_points = []
    if not left_cones and not right_cones:
        return centerline_points  # Nothing detected at all -> cannot estimate.
    # NOTE: only ONE side being empty is fine — on a curve the lane shifts to one
    # half of the frame so every cone lands on the same side. The single-side
    # branches below estimate the centerline by offsetting half a lane width.


    # sort cones by their y-coordinate (distance from the robot, larger y means closer to the robot in image space)
    left_sorted = sorted(left_cones, key=lambda c: c[1], reverse=True)  # Sort left cones by y-coordinate
    right_sorted = sorted(right_cones, key=lambda c: c[1], reverse=True)  # Sort right cones by y-coordinate

    if left_sorted and right_sorted:
        n = min(n_points, len(left_sorted), len(right_sorted))  # Number of points to use for centerline estimation
        for i in range(n):
            left_cone = left_sorted[i]
            right_cone = right_sorted[i]
            center_x = (left_cone[0] + right_cone[0]) // 2  # Average x-coordinate of the left and right cones
            center_y = (left_cone[1] + right_cone[1]) // 2  # Average y-coordinate of the left and right cones
            centerline_points.append((center_x, center_y))
    elif left_sorted:
        # If only left cones are detected, assume a fixed lane width to estimate the centerline
        lane_width_pixels = 200  # This is an assumption and may need tuning based on the actual camera setup
        for (cX, cY) in left_sorted[:n_points]:
            center_x = cX + lane_width_pixels // 2  # Estimate centerline by adding half the lane width to the left cone's x-coordinate
            centerline_points.append((center_x, cY))
    elif right_sorted:
        # If only right cones are detected, assume a fixed lane width to estimate the centerline
        lane_width_pixels = 200  # This is an assumption and may need tuning based on the actual camera setup
        for (cX, cY) in right_sorted[:n_points]:
            center_x = cX - lane_width_pixels // 2  # Estimate centerline by subtracting half the lane width from the right cone's x-coordinate
            centerline_points.append((center_x, cY))
    # vanishing point estimation (optional)
    if centerline_points:
        centerline_points.append((img_width // 2, img_height // 4)) 

    return centerline_points





def run_opencv_planner(image_msg, frame_id, stamp):
    bgr = ros_image_to_cv2(image_msg)  # Convert ROS image message to OpenCV BGR format
    h,w = bgr.shape[:2]

    cones = detect_yellow_cones(bgr)  # Detect yellow cones in the image
    if not cones:
        return None  # No cones detected, cannot plan a path
    
    left_cones, right_cones = split_left_right_cones(cones, w)  # Split cones into left and right groups
    centerline_pixels = estimate_centerline_pixels(left_cones, right_cones,w,h)

    if not centerline_pixels:
        return None  # Cannot estimate centerline, cannot plan a path
    
    path = Path()
    path.header.stamp = stamp.to_msg()  # Use the timestamp from the image message
    path.header.frame_id = frame_id  # Use the provided frame ID

    for (cX, cY) in centerline_pixels:
        pose = PoseStamped()
        pose.header = path.header  # Use the same header for each pose in the path
        # Simple placeholder: map cY to x distance (cY=h → 0m, cY=0 → 2m)
        depth_estimate = (h - cY) / h * 5.0  # Estimate depth based on vertical pixel position
        lateral_estimate = (cX - w / 2) / (w / 2) * 2.0  # Estimate lateral position based on horizontal pixel position
        pose.pose.position.x = depth_estimate  # Forward distance from the robot
        pose.pose.position.y = -lateral_estimate  # Lateral offset from the centerline
        pose.pose.orientation.w = 1.0  # Facing forward (no rotation)
        path.poses.append(pose)
    return path