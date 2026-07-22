"""VLM sign node — Qwen sign reading, decoupled from the path planner.

Phase 1c/1d. Subscribes to the front camera, classifies the nearest road sign with
Qwen2.5-VL-3B on its OWN timer, and publishes the result:
  /vlm/sign        std_msgs/String  — JSON {"sign": <label>, "stamp": <capture sec>}
                   where label is one of {right,left,winding,stop,none,unparsed} and
                   stamp is the classified FRAME's image-header time. Inference takes
                   ~2 s, so by arrival the car has moved ~1.6 m — the stamp lets the
                   planner attribute the label to the board distance AT CAPTURE
                   (sign_latch.SignLabelLatch). The planner also accepts a plain
                   label string (legacy, attributed to the newest frame).
  /vlm/sign_image  sensor_msgs/Image — the camera frame with the label drawn on it

READ-ONLY: it never publishes a path. The path planner (vlm_node.py) runs separately
at its own rate; in later phases it consumes /vlm/sign. This node must run under the
torch venv, so launch it with the venv python directly (NOT `ros2 run`/a launch file,
whose console-script shebang is the system python -> no torch):

    source ~/venvs/vlm_robot/bin/activate
    python3 -m vlm_planner_py.vlm_sign_node --ros-args -p vlm_query_rate_hz:=0.3

Toggle the query live (no restart):
    ros2 param set /vlm_sign vlm_query_enabled false|true

cv_bridge-free (img_convert) so the venv's NumPy 2 is fine (see CLAUDE.md).
"""
import json
import time

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2

from vlm_planner_py.img_convert import img_msg_to_bgr, bgr_to_img_msg
from vlm_planner_py.vlm_prompts import SIGN_PROMPT
from vlm_planner_py.json_parser import parse_sign_class


class VlmSignNode(Node):
    def __init__(self):
        super().__init__('vlm_sign')

        self.image_topic = self.declare_parameter('image_topic', '/camera/front/image_raw').value
        self.sign_topic = self.declare_parameter('sign_topic', '/vlm/sign').value
        self.sign_image_topic = self.declare_parameter('sign_image_topic', '/vlm/sign_image').value
        self.publish_debug_image = bool(self.declare_parameter('publish_debug_image', True).value)
        self.vlm_model_name = self.declare_parameter(
            'vlm_model_name', 'Qwen/Qwen2.5-VL-3B-Instruct').value
        self.vlm_max_new_tokens = int(self.declare_parameter('vlm_max_new_tokens', 64).value)
        # Min confidence to accept a read. Qwen returns a near-constant 0.95 in-world
        # (Phase 1b), so this is off by default; real gating is geometry + temporal
        # consistency in later phases.
        self.vlm_conf_min = float(self.declare_parameter('vlm_confidence_min', 0.0).value)
        self.vlm_query_rate_hz = float(self.declare_parameter('vlm_query_rate_hz', 0.3).value)
        # Live ON/OFF switch, read fresh each tick: ros2 param set /vlm_sign vlm_query_enabled false
        self.declare_parameter('vlm_query_enabled', True)
        # ROI crop hand-off from the planner (vlm_node publishes the latched board's
        # pixel box as JSON on sign_roi_topic). When a FRESH ROI is available, Qwen
        # classifies the crop instead of the full frame — this is what makes it read
        # THE latched board rather than the closest board-shaped object (the
        # featureless BACK of a passed sign made full-frame reads return 'none' on
        # the last straight; see test/lanedump/stop_probe). Falls back to the full
        # frame when no fresh ROI (no board latched yet).
        self.use_sign_roi = bool(self.declare_parameter('use_sign_roi', True).value)
        self.sign_roi_topic = self.declare_parameter('sign_roi_topic', '/vlm/sign_roi').value
        self.roi_max_age_sec = float(self.declare_parameter('roi_max_age_sec', 0.8).value)
        # Crops narrower than this get a 2x cubic upscale before inference (more
        # vision tokens on the board; validated in the stop_probe ROI runs).
        self.roi_upscale_below_px = int(self.declare_parameter('roi_upscale_below_px', 220).value)

        self.sign_pub = self.create_publisher(String, self.sign_topic, 10)
        self.sign_image_pub = self.create_publisher(Image, self.sign_image_topic, 10)
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self._image_cb, 10)
        self.roi_sub = self.create_subscription(
            String, self.sign_roi_topic, self._roi_cb, 10)

        self.latest_image = None
        self.latest_roi = None   # dict {x0,y0,x1,y1,stamp} from the planner
        self._qwen = None
        self._last_raw = ''
        self.latest_sign = None

        # VLM query on its own callback group; MultiThreadedExecutor in main() lets the
        # image subscription keep updating latest_image during the ~2 s inference, and
        # MutuallyExclusive prevents a tick re-entering while one is still running.
        self.vlm_cb_group = MutuallyExclusiveCallbackGroup()
        self.vlm_timer = self.create_timer(1.0 / self.vlm_query_rate_hz, self._sign_tick,
                                           callback_group=self.vlm_cb_group)

        state = ('%.2f Hz' % self.vlm_query_rate_hz
                 if bool(self.get_parameter('vlm_query_enabled').value)
                 else 'DISABLED (vlm_query_enabled=false)')
        self.get_logger().info(
            f'VLM sign node up: reading {self.image_topic} -> {self.sign_topic} '
            f'(+ {self.sign_image_topic}) @ {state}.')

    def _image_cb(self, msg):
        self.latest_image = msg

    def _roi_cb(self, msg):
        try:
            d = json.loads(msg.data)
            if all(k in d for k in ('x0', 'y0', 'x1', 'y1', 'stamp')):
                self.latest_roi = d
        except (ValueError, TypeError):
            pass

    def _crop_for_inference(self, bgr, img_stamp_sec):
        """Return (image-for-Qwen, roi_used). Crops to the planner's board ROI when
        it is fresh relative to THIS frame (both stamps are sim time), upscaling
        small crops; otherwise the full frame."""
        roi = self.latest_roi
        if not (self.use_sign_roi and roi):
            return bgr, None
        if abs(img_stamp_sec - roi['stamp']) > self.roi_max_age_sec:
            return bgr, None
        h, w = bgr.shape[:2]
        x0, y0 = max(0, int(roi['x0'])), max(0, int(roi['y0']))
        x1, y1 = min(w, int(roi['x1'])), min(h, int(roi['y1']))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return bgr, None
        crop = bgr[y0:y1, x0:x1]
        if crop.shape[1] < self.roi_upscale_below_px:
            crop = cv2.resize(crop, None, fx=2.0, fy=2.0,
                              interpolation=cv2.INTER_CUBIC)
        return crop, (x0, y0, x1, y1)

    def _query_enabled(self):
        """Live switch, read fresh so `ros2 param set` toggles it at runtime."""
        return bool(self.get_parameter('vlm_query_enabled').value)

    def _run_vlm(self, prompt, bgr):
        """Run Qwen on the given BGR image -> {'action','confidence'} or None (fail
        safe on unknown/low-confidence/error). Lazy-loads the model on first call."""
        if bgr is None:
            return None
        from vlm_planner_py.model_runner import (load_qwen_model, run_qwen_inference,
                                                 cv2_to_pil)
        if self._qwen is None:
            self.get_logger().info(f'Loading VLM ({self.vlm_model_name}) on first tick '
                                   '(this can take a while)...')
            self._qwen = load_qwen_model(self.vlm_model_name)
        model, processor = self._qwen
        try:
            pil = cv2_to_pil(bgr)
            raw, _elapsed = run_qwen_inference(model, processor, pil, prompt,
                                               max_new_tokens=self.vlm_max_new_tokens)
        except Exception as e:  # inference must never crash the node
            self.get_logger().warn(f'VLM read failed: {e}', throttle_duration_sec=5.0)
            self._last_raw = f'ERROR: {e}'
            return None
        self._last_raw = raw
        result = parse_sign_class(raw)
        if result is None:
            return None
        conf = result.get('confidence')
        if self.vlm_conf_min > 0.0 and conf is not None and conf < self.vlm_conf_min:
            return None
        return result

    def _sign_tick(self):
        if not self._query_enabled():
            self.get_logger().info('VLM query disabled (vlm_query_enabled=false), skipping.',
                                   throttle_duration_sec=10.0)
            return
        if self.latest_image is None:
            self.get_logger().info('VLM tick: no image yet, skipping.',
                                   throttle_duration_sec=5.0)
            return
        # Snapshot the frame ONCE: the ~2 s inference must classify, stamp and
        # overlay the SAME image (latest_image keeps updating underneath us).
        img_msg = self.latest_image
        stamp_sec = img_msg.header.stamp.sec + img_msg.header.stamp.nanosec * 1e-9
        try:
            bgr = img_msg_to_bgr(img_msg)
        except Exception as e:
            self.get_logger().warn(f'Image decode failed: {e}', throttle_duration_sec=5.0)
            return
        infer_img, roi_used = self._crop_for_inference(bgr, stamp_sec)
        t0 = time.time()
        result = self._run_vlm(SIGN_PROMPT, infer_img)
        dt = time.time() - t0
        self.latest_sign = result
        action = result['action'] if result else 'unparsed'
        conf = result.get('confidence') if result else None
        cstr = f'{conf:.2f}' if isinstance(conf, float) else 'n/a'
        self.get_logger().info(
            f'[vlm_sign] action={action} confidence={cstr} latency={dt:.2f}s '
            f'frame_t={stamp_sec:.2f} roi={roi_used or "full"} raw={self._last_raw!r}')
        self.sign_pub.publish(String(data=json.dumps(
            {'sign': action, 'stamp': stamp_sec})))
        if self.publish_debug_image:
            try:
                self._publish_overlay(bgr, action, cstr, dt,
                                      header=img_msg.header, roi=roi_used)
            except Exception as e:
                self.get_logger().warn(f'Failed to publish sign overlay: {e}',
                                       throttle_duration_sec=5.0)

    def _publish_overlay(self, bgr, action, cstr, latency, header=None, roi=None):
        """Draw the label (and the inference ROI, when one was used) on the frame and
        publish on the sign-image topic (its own topic so it never collides with the
        planner's /vlm/debug_image overlay)."""
        dbg = bgr.copy()
        if roi is not None:
            cv2.rectangle(dbg, (roi[0], roi[1]), (roi[2], roi[3]), (255, 255, 0), 2)
        txt = f'{action}  conf={cstr}  {latency:.1f}s  [{"roi" if roi else "full"}]'
        cv2.rectangle(dbg, (0, 0), (dbg.shape[1], 28), (0, 0, 0), -1)
        color = (0, 0, 255) if action in ('unparsed', 'none') else (0, 255, 0)
        cv2.putText(dbg, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        self.sign_image_pub.publish(bgr_to_img_msg(dbg, header=header))


def main(args=None):
    rclpy.init(args=args)
    node = VlmSignNode()
    # MultiThreadedExecutor: the VLM timer (own group) runs the ~2 s inference on a
    # separate thread while the image subscription keeps latest_image fresh.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
