import cv2
import numpy as np
import sys
sys.path.insert(0, 'src/vlm_planner_py')
from vlm_planner_py.opencv_fallback import detect_yellow_cones, split_left_right_cones, estimate_centerline_pixels

# Create a synthetic test image with yellow rectangles (simulated cones)
img = np.zeros((360, 640, 3), dtype=np.uint8)
img[:] = (80, 80, 80)  # gray background

# Draw yellow cones (left and right)
for y_pos in [100, 200, 280]:
    cv2.rectangle(img, (100, y_pos), (130, y_pos+60), (0, 220, 220), -1)  # left cone (BGR yellow)
    cv2.rectangle(img, (510, y_pos), (540, y_pos+60), (0, 220, 220), -1)  # right cone

cones = detect_yellow_cones(img)
print(f"Detected {len(cones)} cones: {cones}")

left, right = split_left_right_cones(cones, 640)
print(f"Left: {left}, Right: {right}")


# Visualize
for (u, v) in cones:
    cv2.circle(img, (u, v), 10, (0, 0, 255), -1)
cv2.imwrite('/tmp/cone_detection_test.png', img)
print("Saved to /tmp/cone_detection_test.png")