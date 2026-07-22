"""cv_bridge-free sensor_msgs/Image <-> numpy converters.

cv_bridge's compiled boost module crashes at runtime under NumPy 2.x
(`_ARRAY_API not found` in cvtColor2), and the VLM node must run inside the torch
venv (NumPy 2). These numpy-only converters keep both nodes independent of that
conflict; they also work fine under system NumPy 1.x, so the planner uses them too.
Shared by vlm_node.py (planner) and vlm_sign_node.py (VLM). See CLAUDE.md.
"""
import numpy as np
from sensor_msgs.msg import Image


def img_msg_to_bgr(msg):
    """Decode a sensor_msgs/Image to a BGR numpy array. Handles the common camera
    encodings + row padding (msg.step)."""
    enc = msg.encoding.lower() # check encoding case-insensitively
    ch = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1}.get(enc) #select number of channels based on encoding
    if ch is None:
        raise ValueError(f'unsupported image encoding {msg.encoding!r}')
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    arr = buf.reshape(msg.height, msg.step)[:, :msg.width * ch].reshape(
        msg.height, msg.width, ch)  # reshape to (height, width, channels) and crop to width
    if enc == 'rgb8':
        arr = arr[:, :, ::-1]                 # RGB -> BGR
    elif enc == 'rgba8':
        arr = arr[:, :, 2::-1]                # RGBA -> BGR
    elif enc == 'bgra8':
        arr = arr[:, :, :3]                   # BGRA -> BGR
    elif enc == 'mono8':
        arr = np.repeat(arr, 3, axis=2)
    return np.ascontiguousarray(arr)


def bgr_to_img_msg(bgr, header=None):
    """Pack a BGR numpy array into a sensor_msgs/Image."""
    msg = Image()
    if header is not None:
        msg.header = header
    msg.height, msg.width = bgr.shape[:2]
    msg.encoding = 'bgr8'
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(bgr, dtype=np.uint8).tobytes()
    return msg


def img_msg_to_depth(msg):
    """Decode a depth sensor_msgs/Image to a float32 (metres) numpy array.
    ros_gz_bridge publishes depth as 32FC1 (metres) — a plain reshape, equivalent
    to cv_bridge's passthrough; 16UC1 (mm) handled as a fallback."""
    enc = msg.encoding.lower()
    if enc in ('32fc1', '32f'):
        d = np.frombuffer(msg.data, dtype=np.float32).reshape(
            msg.height, msg.step // 4)[:, :msg.width]
        return np.ascontiguousarray(d)
    if enc in ('16uc1', 'mono16'):
        d = np.frombuffer(msg.data, dtype=np.uint16).reshape(
            msg.height, msg.step // 2)[:, :msg.width]
        return np.ascontiguousarray(d.astype(np.float32) / 1000.0)  # mm -> m
    raise ValueError(f'unsupported depth encoding {msg.encoding!r}')
