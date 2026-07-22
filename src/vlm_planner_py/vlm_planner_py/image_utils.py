import cv2
import numpy as np
from cv_bridge import CvBridge

_bridge = CvBridge()

def ros_image_to_cv2(ros_img):
    # Convert ROS Image message to OpenCV image (numpy array)
    return _bridge.imgmsg_to_cv2(ros_img, desired_encoding='bgr8')
def ros_depth_to_cv2(ros_depth):
    # Convert ROS Image message to OpenCV depth image in meters (numpy array)
    return _bridge.imgmsg_to_cv2(ros_depth, desired_encoding='32FC1')

def cv2_to_ros_image(cv_img, encoding='bgr8', header=None):
    # Convert an OpenCV image (numpy array) back to a ROS Image message.
    msg = _bridge.cv2_to_imgmsg(cv_img, encoding=encoding)
    if header is not None:
        msg.header = header
    return msg
