import rclpy
from typing import cast
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Float64
import math
import os
import json
import cv2
import numpy as np
from transforms3d.euler import euler2quat, quat2euler
import tf2_ros
from vlm_planner_py.projection import (project_pixel_to_base_link, project_pixels_to_base_link, build_camera_to_base_transform) # type: ignore[import-untyped]
from vlm_planner_py.centerline import build_centerline_gatewalk, centerline_from_line_cloud, path_folds_back # type: ignore[import-untyped]
from vlm_planner_py.debugtap import DebugTap # type: ignore[import-untyped]
from vlm_planner_py.img_convert import img_msg_to_bgr, bgr_to_img_msg, img_msg_to_depth
from vlm_planner_py.maneuver import ManeuverStateMachine
from vlm_planner_py.sign_latch import SignLabelLatch


class VlmPlannerNode(Node):
    def __init__(self):
        super().__init__('vlm_planner')

        # Parameters
        self.mode = self.declare_parameter('mode', 'fake_path').value
        self._rate_hz = cast(float, self.declare_parameter('planner_rate_hz', 1.0).value)
        self.img_w = self.declare_parameter('image_width', 640).value
        self.img_h = self.declare_parameter('image_height', 480).value
        self.output_frame = self.declare_parameter('output_frame', 'odom').value
        self.max_path_dist = cast(float, self.declare_parameter('max_path_distance_m', 8.0).value)
        self.path_topic = cast(str, self.declare_parameter('path_topic', '/vlm_path_odom').value)
        self.odom_topic = cast(str, self.declare_parameter('odometry_topic', '/odom_ekf').value)
        self.reuse_last_path = cast(bool, self.declare_parameter('reuse_last_path_if_failed', True).value)
        self.max_reuse_time = cast(float, self.declare_parameter('max_reuse_time_sec', 1.5).value)

        # Curve-aware (metric-space) detection params — see
        # docs/detection_curve_tracking_plan.md.
        self.cone_area_min_px = cast(float, self.declare_parameter('cone_area_min_px', 20.0).value)
        self.cone_area_max_px = cast(float, self.declare_parameter('cone_area_max_px', 20000.0).value)
        self.depth_window_px = cast(int, self.declare_parameter('depth_window_px', 3).value)
        # Height gate (base_link z, metres). base_link is ~ground level, so cones
        # (centre ~0.1 m, top ~0.2 m) pass while the elevated roadside sign boards
        # (~0.55 m) are rejected — stops the yellow signs registering as cones and
        # the red STOP board as the goal. See project_pixels_to_base_link.
        self.cone_height_min_m = cast(float, self.declare_parameter('cone_height_min_m', -0.3).value)
        self.cone_height_max_m = cast(float, self.declare_parameter('cone_height_max_m', 0.3).value)
        # Lane markings are paint on the ground (z~0), not 3D objects, so they get a
        # tighter z-gate than the cones: this rejects spurious line points thrown to
        # elevated z by depth noise, while still dropping the sign boards (z>=0.4).
        self.line_z_min_m = cast(float, self.declare_parameter('line_z_min_m', -0.15).value)
        self.line_z_max_m = cast(float, self.declare_parameter('line_z_max_m', 0.15).value)
        self.half_lane_m = cast(float, self.declare_parameter('half_lane_m', 0.5).value)

        # Track marking type. True => gray boundary LINES (continuous; use
        # detect_track_lines + centerline_from_line_cloud). False => legacy yellow
        # CONES (detect_yellow_cones + build_centerline_gatewalk). half_lane_m
        # defaults to 0.5 to match the line world's LANE_W=1.0.
        self.line_detection = cast(bool, self.declare_parameter('line_detection', True).value)
        self.line_val_min = cast(int, self.declare_parameter('line_val_min', 150).value)
        self.line_sat_max = cast(int, self.declare_parameter('line_sat_max', 70).value)
        self.line_sample_stride = cast(int, self.declare_parameter('line_sample_stride', 8).value)
        # Min connected-component area [px] for a line-mask blob. Small far-range
        # blobs (15-49 px) project to stray mid-lane point clusters that kink the
        # centerline; 50 kills them at the pixel level (2026-07-10).
        self.line_area_min_px = cast(int, self.declare_parameter('line_area_min_px', 50).value)
        # Rows above this are the sky and are cropped out before the color mask
        # runs (2026-07-18): the procedural <sky> in track_lines.world.sdf can be
        # bright + desaturated enough to match the "light gray marking" mask, and
        # near the horizon can slip past the depth/height gates too (the 15 m far
        # clip coincides with the depth filter's upper bound). Default 150 px is
        # MEASURED, not derived from camera geometry: the trig estimate from hfov/
        # pitch put the horizon at row ~124-134, but a captured frame (scripts/
        # capture_frame.py + a per-row HSV/depth sweep) showed the mask actually
        # firing solidly through row 142 and clearing at 144, with ground depth
        # not 100% valid until row 148 -- haze pushes the visually sky-like band a
        # bit past the pure-geometry horizon. 150 clears both signals with a small
        # margin. Retune (recapture + resweep) if the camera mount, resolution, or
        # sky/fog settings change.
        self.line_roi_top_px = cast(int, self.declare_parameter('line_roi_top_px', 150).value)
        # Forward horizon for the LINE centerline [m]: far markings project from
        # noisy long-range depth and drag the centerline out of the lane, so clip
        # the line cloud to the reliable near range before building the centerline.
        self.line_d_max_m = cast(float, self.declare_parameter('line_d_max_m', 3.0).value)
        # Offline centerline debugging: when non-empty, every planning frame appends
        # its base_link lane cloud (the input to centerline_from_line_cloud) as one
        # JSON line to <dir>/lane_cloud.jsonl. Replay + plot with
        # test/debug_centerline.py. Empty (default) => no dump, zero overhead.
        self.lane_cloud_dump_dir = cast(str, self.declare_parameter('lane_cloud_dump_dir', '').value)
        self._lane_dump_idx = 0
        self._dump_lane_base = []
        # Keep the dump dir live-settable: the debugkit recorder redirects it
        # into the session (`record_session.sh --lane-dump`) via `ros2 param
        # set` mid-run, which the init-time cast above alone would ignore.
        self.add_on_set_parameters_callback(self._on_params_set)
        # Path smoothing: stateless poly fit (degree>0) de-noises per-frame jitter
        # with no lag; degree=0 falls back to the moving average of smooth_window.
        self.path_fit_degree = cast(int, self.declare_parameter('path_fit_degree', 3).value)
        self.smooth_window = cast(int, self.declare_parameter('smooth_window', 5).value)
        # Uniform arc-length spacing of the published path [m]. Smaller = finer
        # curve detail + more poses (steadier headings, bigger msg); larger =
        # coarser + fewer poses. The MPC re-resamples its own horizon on top.
        self.path_spacing_m = cast(float, self.declare_parameter('path_spacing_m', 0.2).value)
        self.gate_d_min_m = cast(float, self.declare_parameter('gate_d_min_m', 0.3).value)
        self.gate_d_max_m = cast(float, self.declare_parameter('gate_d_max_m', 3.0).value)
        self.max_centerline_pts = cast(int, self.declare_parameter('max_centerline_pts', 20).value)
        # Max accepted centerline arc length [m]; the S course runs ~11 m, so the
        # old validate_path default of 10 m rejected a valid full-course path.
        self.path_max_len_m = cast(float, self.declare_parameter('path_max_len_m', 15.0).value)

        # Creep-to-goal behavior: when no yellow lane is usable, keep moving
        # slowly toward the red goal cone and stop once it is within reach.
        self.creep_len_m = cast(float, self.declare_parameter('creep_len_m', 1.0).value)
        self.goal_stop_dist_m = cast(float, self.declare_parameter('goal_stop_dist_m', 0.8).value)

        # Debug FPV overlay: republish the front camera image with the detected
        # cones/goal drawn on it (raw detections vs. height-accepted), so the
        # detection can be watched in RViz as an Image display.
        self.publish_debug_image = cast(bool, self.declare_parameter('publish_debug_image', True).value)
        self.debug_image_topic = cast(str, self.declare_parameter('debug_image_topic', '/vlm/debug_image').value)

        # Cross-node VLM gate (optional). VLM sign reading lives in a SEPARATE node
        # (vlm_sign_node.py) publishing the label on sign_topic. When
        # wait_for_first_sign is True, the planner withholds ALL path output until
        # the first sign message arrives (i.e. the VLM is up and has produced its
        # first inference), so the car does not drive before the VLM is ready.
        # Default False -> planner runs standalone (old behaviour) with no VLM node.
        self.wait_for_first_sign = cast(bool, self.declare_parameter('wait_for_first_sign', False).value)
        self.sign_topic = cast(str, self.declare_parameter('sign_topic', '/vlm/sign').value)

        # Phase 2 — sign localization in odom. The elevated sign boards (~0.55 m)
        # are dropped by the lane/goal height gate, so detect them (yellow = turn/
        # winding, red = stop) and project with a HIGH z-gate that keeps only the
        # boards. The nearest is latched as sign_odom; its distance feeds the banner.
        # If board depth proves unreliable (thin elevated surface — risk R-D), fall
        # back to ray-casting the board pixel to the known sign height plane.
        self.sign_z_min_m = cast(float, self.declare_parameter('sign_z_min_m', 0.4).value)
        self.sign_z_max_m = cast(float, self.declare_parameter('sign_z_max_m', 0.8).value)
        self.sign_area_min_px = cast(float, self.declare_parameter('sign_area_min_px', 30.0).value)
        self.sign_area_max_px = cast(float, self.declare_parameter('sign_area_max_px', 100000.0).value)
        # Sign arbitration reach [m]: the VLM label is only armed as the candidate
        # maneuver (pending_sign) when the localized board is within this range ahead
        # (0 < dist < sign_reach_dist_m). Beyond it (or unknown / behind), pending_sign
        # falls back to 'none'. See _arbitrate_pending_sign.
        self.sign_reach_dist_m = cast(float, self.declare_parameter('sign_reach_dist_m', 3.0).value)
        # Sign COMMIT distance [m]: the armed maneuver is COMMITTED (turn begins) on the
        # first frame the localized board comes within this range AHEAD -- NOT when it
        # passes behind. Must be < sign_reach_dist_m so the label is armed (validated
        # ahead) before the commit fires. Larger => the maneuver starts earlier / farther
        # before the feature.
        self.sign_commit_dist_m = cast(float, self.declare_parameter('sign_commit_dist_m', 1.5).value)
        # STOP commits LATE (halt at the line), unlike turns/winding whose
        # road geometry starts several metres BEFORE their board (loop-8
        # autotune: the U-turn 'left' latched at 4.6 m but its curve begins
        # ~5 m out — commit at 3.5 m was never reached in 'straight' weave).
        self.sign_commit_stop_dist_m = cast(float, self.declare_parameter('sign_commit_stop_dist_m', 1.5).value)
        # WINDING commits EARLY: its S-geometry begins ~3-4 m before the
        # board (loops 2 and 10: at commit 1.5 the car reaches the first S
        # bend still in 'straight' at cruise and diverges within 2 s).
        self.sign_commit_winding_dist_m = cast(float, self.declare_parameter('sign_commit_winding_dist_m', 4.0).value)
        # LOCK distance [m]: once the tracked board comes within this range
        # with a latched label, the decision is FROZEN — no further VLM read
        # may change it, and no nearer blob may steal the board identity. On
        # the final approach the board escapes the camera view (edge clip /
        # crop-out), so anything "seen" from here on is the board's remains
        # or the NEXT sign; the maneuver must ride on the decision made while
        # the board was fully visible. Must be > the commit distances so the
        # locked label is what commits.
        self.sign_label_lock_dist_m = cast(float, self.declare_parameter('sign_label_lock_dist_m', 3.0).value)
        # Label READ acceptance band [m] on the latched-board distance AT THE READ'S
        # CAPTURE TIME (sign_latch.SignLabelLatch). Ceiling: far reads are unreliable
        # (small board -> past misreads) and never enter the latch. Floor: below it
        # the pitched-down camera crops the board out of the frame top (top from
        # ~3.3 m board distance, the STOP letters from ~2.7 m), so reads there are
        # 'none' or — with the next sign in view — the WRONG board's symbol; the
        # latched label is frozen instead. 'none' never erases a latched label.
        self.sign_label_max_dist_m = cast(float, self.declare_parameter('sign_label_max_dist_m', 5.0).value)
        self.sign_label_min_dist_m = cast(float, self.declare_parameter('sign_label_min_dist_m', 3.3).value)
        # ROI hand-off to the VLM sign node: publish the latched board's pixel box so
        # Qwen classifies a CROP around the board instead of the full frame. Root
        # cause (stop_probe, 2026-07-04): full-frame Qwen picks the CLOSEST board —
        # on the last straight that is the featureless BACK of the passed left sign,
        # so it answers 'none' at 5.5-2.8 m (prompt steering did NOT fix it; the ROI
        # crop reads 'stop' 5/5 at all distances with the unchanged prompt).
        self.sign_roi_topic = cast(str, self.declare_parameter('sign_roi_topic', '/vlm/sign_roi').value)
        self.sign_board_width_m = cast(float, self.declare_parameter('sign_board_width_m', 0.45).value)

        # Maneuver state machine (see maneuver.py / KB §2.19). ENTRY into a turn/winding/
        # stop fires when the sign board passes BEHIND the robot (using the label that was
        # validated ahead); EXIT back to 'straight' fires when the planned lane path is
        # straight again, with engage-then-release hysteresis so the straight APPROACH
        # before the turn cannot end it early. straight_release_dist_m = the metres of
        # continuously-straight road needed to release (debounces a WINDING road, which is
        # only briefly straight at each inflection); straight_min_len_m guards against
        # judging too short a path.
        self.straight_heading_tol_rad = cast(float, self.declare_parameter('straight_heading_tol_rad', 0.20).value)
        self.maneuver_engage_tol_rad = cast(float, self.declare_parameter('maneuver_engage_tol_rad', 0.35).value)
        self.straight_release_dist_m = cast(float, self.declare_parameter('straight_release_dist_m', 2.0).value)
        self.straight_min_len_m = cast(float, self.declare_parameter('straight_min_len_m', 1.0).value)

        # Phase 5 — speed setpoint into the MPC. The maneuver FSM state maps to a target
        # speed [m/s] published on target_speed_topic (Float64) EVERY plan tick; the MPC
        # subscribes and overrides its target speed at runtime (it reads params only at
        # startup). Turn *direction* stays in the path; only *speed* is set here. On a
        # dead planner the MPC falls back to its own default (never accelerates).
        self.speed_straight = cast(float, self.declare_parameter('speed_straight', 0.8).value)
        self.speed_turn = cast(float, self.declare_parameter('speed_turn', 0.5).value)
        self.speed_winding = cast(float, self.declare_parameter('speed_winding', 0.4).value)
        self.speed_stop = cast(float, self.declare_parameter('speed_stop', 0.0).value)
        self.target_speed_topic = cast(str, self.declare_parameter('target_speed_topic', '/maneuver/target_speed').value)
        # Raw maneuver state topic (String): consumed by the MPC for its
        # maneuver-dependent heading lookahead (and handy for debugging).
        self.maneuver_state_topic = cast(str, self.declare_parameter('maneuver_state_topic', '/maneuver/state').value)

        # state
        self.latest_image = None
        self.latest_depth = None
        self.latest_cam_info = None
        self.latest_odom = None
        # Set True on the first sign message from the VLM node (releases the gate).
        self._sign_seen = False
        # Latest sign label from /vlm/sign, drawn onto the detection overlay so the
        # detection image and the VLM read share one view. None until first message.
        self.latest_sign_label = None
        # Phase 2: latched sign position in odom + distance to it + its image pixel.
        self.sign_odom = None
        self.latest_sign_dist = None
        self._sign_px = None
        # Last successfully planned path (odom frame) + the time it was produced,
        # used to bridge brief detection failures (reuse_last_path).
        self.last_good_path = None
        self.last_good_time = -1.0e9
        # Latched red-goal position in the odom frame (set the first time the goal
        # cone is seen during creep); used to stop by odometry distance even after
        # the cone leaves the camera view at close range.
        self.goal_odom = None
        # Maneuver coordination. pending_sign = the validated candidate maneuver while
        # its board is localized close ahead (set by _arbitrate_pending_sign); the state
        # machine commits it when the board comes within sign_commit_dist_m ahead.
        # _sign_commit_event is that commit edge for the current tick, and _sign_committed
        # guards it to fire ONCE per latched board (reset when the board passes behind).
        self.pending_sign = 'none'
        self._sign_commit_event = False
        self._sign_committed = False
        self.maneuver_sm = ManeuverStateMachine(
            engage_tol_rad=self.maneuver_engage_tol_rad,
            straight_tol_rad=self.straight_heading_tol_rad,
            release_dist_m=self.straight_release_dist_m,
            min_len_m=self.straight_min_len_m)
        # Robot odom position at the last is_lane on_path call, to measure how far the
        # robot has travelled on straight road for the maneuver-release debounce.
        self._last_lane_pos = None
        # Temporal left/right boundary memory for centerline_from_line_cloud:
        # {'left': y, 'right': y} last-seen near-end laterals, carried across
        # frames so a single-boundary frame is sided consistently instead of by
        # the noise sign of its lateral (the path flip-flop source). Cleared
        # whenever the lane is lost (creep) — stale sides are worse than none.
        self._lane_side_memory = {}
        # Deadline until which a garbage frame (fold-back centerline) may HOLD
        # the previous path (publish nothing; the MPC keeps tracking its last
        # path) instead of falling to creep.
        self._lane_hold_until = -1.0e9

        # Distance-gated label latch: attributes each VLM read to the board that was
        # latched when the read's FRAME was captured, accepts it only inside the
        # readable band, newest accepted read wins, 'none' never erases, and the
        # latch clears when the board passes behind. Replaces the old sign_locked
        # flag: pending_sign is stable inside reach because the latch is frozen
        # below sign_label_min_dist_m (> sign_reach_dist_m arm threshold).
        self.sign_latch = SignLabelLatch(min_dist_m=self.sign_label_min_dist_m,
                                         max_dist_m=self.sign_label_max_dist_m)

        # Internal-signal tap for the debugkit recorder (docs/testing_pipeline.md):
        # per-tick centerline internals + tick outcome -> /debug/vlm_planner as JSON.
        # Off by default; enable live with `ros2 param set /vlm_planner debug_tap true`.
        self.tap = DebugTap(self)

        # Timer for planning loop (planner_rate_hz). VLM runs in a SEPARATE node now.
        period = 1.0 / self._rate_hz
        self.timer = self.create_timer(period, self.plan_once)

        self.get_logger().info(
            f'VLM Planner Node initialized in {self.mode} mode, publishing to '
            f'{self.path_topic} at {self._rate_hz} Hz.'
            f'{" Gating path until first sign on %s." % self.sign_topic if self.wait_for_first_sign else ""}')

        # TF buffer and listener for camera-to-base transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.camera_frame = self.declare_parameter('camera_frame', 'front_camera_optical_frame').value
        self.base_link_frame = self.declare_parameter('robot_frame', 'base_link').value


            # Subscriptions
        self.image_sub = self.create_subscription(
            Image, '/camera/front/image_raw', self.image_callback, 10)     # camera image for lane/goal detection (opencv mode)
        self.depth_sub = self.create_subscription(
            Image, '/camera/front/depth/image_raw', self.depth_callback, 10)   # depth image for lane/goal detection (opencv mode)
        self.cam_info_sub = self.create_subscription(
            CameraInfo, '/camera/front/camera_info', self.cam_info_callback, 10)   # camera info for lane/goal detection (opencv mode)
        self.odom_sub = self.create_subscription( 
            Odometry, self.odom_topic, self.odom_callback, 10)     # odometry for robot pose and goal distance (all modes)
        self.sign_sub = self.create_subscription(
            String, self.sign_topic, self._sign_callback, 10)   # sign label from the VLM node (only used for the optional startup gate)
        # MPC control-debug readouts for the FPV overlay: the slew-limited target
        # speed the solver actually receives (not the raw maneuver setpoint) and
        # the raw QP outputs. Stored with a receive stamp so the overlay can show
        # 'n/a' instead of stale numbers when the MPC is down.
        self._mpc_dbg = {}          # name -> (value, recv_time_sec)
        self.mpc_vref_sub = self.create_subscription(
            Float64, '/mpc/v_ref_ramped', lambda m: self._mpc_dbg_cb('vref', m), 10)
        self.mpc_accel_sub = self.create_subscription(
            Float64, '/mpc/cmd_accel', lambda m: self._mpc_dbg_cb('accel', m), 10)
        self.mpc_steer_sub = self.create_subscription(
            Float64, '/mpc/cmd_steer', lambda m: self._mpc_dbg_cb('steer', m), 10)

        # Publisher
        self.path_pub = self.create_publisher(Path, self.path_topic, 10)
        self.debug_image_pub = self.create_publisher(Image, self.debug_image_topic, 10)
        # Phase 5: maneuver-derived speed setpoint for the MPC (published every tick).
        self.speed_pub = self.create_publisher(Float64, self.target_speed_topic, 10)
        # Latched-board pixel ROI for the VLM sign node (JSON String, see above).
        self.sign_roi_pub = self.create_publisher(String, self.sign_roi_topic, 10)
        # Raw maneuver FSM state, published every tick with the speed setpoint. The
        # MPC maps it to its maneuver-dependent heading lookahead.
        self.state_pub = self.create_publisher(String, self.maneuver_state_topic, 10)
        # Arbitrated candidate maneuver (see _arbitrate_pending_sign): the latched
        # sign label while its board is within sign_reach_dist_m ahead, else 'none'.
        # Telemetry only -- no node subscribes to this; it exists so the record/
        # analyze pipeline (tools/debugkit) can see the arm step between the raw
        # VLM read (/vlm/sign) and the FSM's committed state (/maneuver/state).
        self.pending_sign_pub = self.create_publisher(String, '/vlm/pending_sign', 10)



    def image_callback(self, msg):
        self.latest_image = msg

    def depth_callback(self, msg):
        self.latest_depth = msg

    def cam_info_callback(self, msg):
        self.latest_cam_info = msg

    def odom_callback(self, msg):
        self.latest_odom = msg

    def _sign_callback(self, msg):
        # New format: JSON {"sign": <label>, "stamp": <capture sec>}. Legacy plain
        # label strings still parse (stamp None -> latch attributes to the newest
        # recorded frame).
        label, t_capture = msg.data, None
        try:
            d = json.loads(msg.data)
            if isinstance(d, dict) and 'sign' in d:
                label = d['sign']
                t_capture = d.get('stamp')
        except (ValueError, TypeError):
            pass
        # Raw latest label for the FPV overlay banner (may be 'none'/'unparsed').
        self.latest_sign_label = label
        # Offer the read to the distance-gated latch (see sign_latch.py): only a
        # read whose CAPTURE frame had the board inside the readable band updates
        # the latched label used by the arbitration.
        accepted, reason = self.sign_latch.on_label(label, t_capture)
        self.get_logger().info(
            f'[sign] VLM read {label!r} (frame_t='
            f'{"n/a" if t_capture is None else f"{t_capture:.2f}"}) -> '
            f'{"LATCHED" if accepted else "ignored"}: {reason} | '
            f'latched={self.sign_latch.label!r}')
        if not self._sign_seen:
            self._sign_seen = True
            self.get_logger().info(f'First sign received on {self.sign_topic} '
                                   f'(={label!r}); path gate released.')

    def _update_sign_odom(self, sign_proj, p, cos_y, sin_y, frame_t=None,
                          fx=None, img_wh=None):
        """Latch the nearest elevated sign board in odom (mirrors goal_odom) and set
        latest_sign_dist for the banner. Re-latches to the nearest board ahead each
        time one is seen and current distance is < dist_reach_sign; clears the latch once it passes behind the robot.

        frame_t = the processed image's header stamp [s]; recorded with the
        resulting board distance into the label latch so an arriving VLM read
        (captured ~2 s ago) is judged against the distance AT ITS CAPTURE TIME.

        sign_proj entries are (x, y, z, px, py) in base_link."""
        self._sign_px = None
        self._sign_view_clipped = False   # set per-frame by the ROI edge-clip guard
        # While the label is LOCKED (board within sign_label_lock_dist_m with
        # a latched label) the board identity is frozen too: no nearer blob
        # may replace sign_odom (on the final approach any new blob is the
        # NEXT sign or this board's clipped remains), and the ROI stays
        # unpublished so the sign node cannot be steered onto that blob.
        if sign_proj and not self.sign_latch.locked:
            sx, sy, sz, spx, spy = min(sign_proj, key=lambda s: math.hypot(s[0], s[1]))
            # get new sign_odom
            new_sign_odom = (p.x + cos_y * sx - sin_y * sy,
                             p.y + sin_y * sx + cos_y * sy)
            new_sign_px = (spx, spy)
            dx, dy = new_sign_odom[0] - p.x, new_sign_odom[1] - p.y
            new_sign_dist = math.hypot(dx, dy)
        
            if self.latest_sign_dist is None or new_sign_dist < self.latest_sign_dist:     # update if the new sign is closer than the previous one
                self.sign_odom = new_sign_odom
                self._sign_px = new_sign_px
                self.latest_sign_dist = new_sign_dist

            # Publish the board's pixel ROI for the VLM sign node on EVERY frame
            # the nearest board is detected (not just on a closer re-latch — a
            # static or equal-distance frame must keep the ROI fresh, or the sign
            # node's freshness gate falls back to the full frame). Crop box
            # centered on the blob, sized from the expected board width in px
            # (fx * board_w / forward). The crop makes Qwen see THE board — not
            # the closer featureless back of an already-passed sign (the 'none'
            # failure on the last straight, see test/lanedump/stop_probe).
            if fx is not None and img_wh is not None and frame_t is not None:
                w_px = fx * self.sign_board_width_m / max(sx, 0.3)
                half = int(max(0.8 * w_px, 48))
                iw, ih = img_wh
                # Edge-clip guard: when the box would extend past the image
                # border, part of the board is OUT OF FRAME and the visible
                # half of the symbol reads as the wrong arrow (loop-4 autotune:
                # right board sliced by the right border at 3.5 m -> Qwen read
                # 'left' -> wrong maneuver committed). Suppress the ROI and
                # mark the frame so the label latch ignores reads captured on
                # it — the fully-visible farther reads decide instead.
                if (int(spx) - half < 0 or int(spx) + half > iw
                        or int(spy) - half < 0):
                    self._sign_view_clipped = True
                else:
                    roi = {'x0': max(0, int(spx) - half),
                           'y0': max(0, int(spy) - half),
                           'x1': min(iw, int(spx) + half),
                           'y1': min(ih, int(spy) + half),
                           'stamp': frame_t}
                    self.sign_roi_pub.publish(String(data=json.dumps(roi)))

        commit = False

        if self.sign_odom is not None:
            dx, dy = self.sign_odom[0] - p.x, self.sign_odom[1] - p.y
            if cos_y * dx + sin_y * dy < 0:   # latched sign now behind -> drop it
                self.sign_odom = None
                self.latest_sign_dist = None
                self._sign_committed = False  # re-arm the commit edge for the next board
                # New board identity: clears the latched label and invalidates any
                # in-flight VLM read of the old board (stale-board rejection).
                self.sign_latch.clear_board()
            else:
                self.latest_sign_dist = math.hypot(dx, dy)
                # LOCK the decision once the board is close enough that its
                # view is about to degrade (edge clip / crop-out): the label
                # read while it was fully visible rides through to the commit
                # untouched, even if the board vanishes from the camera.
                if (not self.sign_latch.locked
                        and self.sign_latch.label is not None
                        and self.latest_sign_dist < self.sign_label_lock_dist_m):
                    self.sign_latch.lock()
                    self.get_logger().info(
                        f"[sign] label '{self.sign_latch.label}' LOCKED at "
                        f"{self.latest_sign_dist:.2f}m (board tracking frozen "
                        f"until it passes)")
                # Maneuver ENTRY trigger: assert EVERY tick while the board is within
                # sign_commit_dist_m ahead until the FSM actually consumes it (label
                # armed). A one-shot edge here races the ~2 s VLM inference lag: the
                # label can arrive after the range is entered, and the lost edge
                # silently drops the whole maneuver. _sign_committed is set where the
                # FSM reports the commit landed (_arbitrate_pending_sign).
                # Per-LABEL commit range: 'stop' halts AT the line (late);
                # 'winding' engages before its S starts (early); left/right
                # use the default short distance (the corner works there —
                # loop 10).
                commit_dist = {
                    'stop': self.sign_commit_stop_dist_m,
                    'winding': self.sign_commit_winding_dist_m,
                }.get(self.sign_latch.label or '', self.sign_commit_dist_m)
                if (self.latest_sign_dist < commit_dist
                        and not self._sign_committed):
                    commit = True

        # Commit edge consumed by the maneuver FSM this tick (see _arbitrate_pending_sign).
        self._sign_commit_event = commit

        # Record this frame (capture stamp -> board distance) for the label latch,
        # AFTER the latch/behind update so the distance reflects this frame.
        # An edge-clipped frame records no distance: a read captured on it is
        # of a half-visible symbol and must not enter the latch.
        if frame_t is not None:
            self.sign_latch.record(
                frame_t,
                None if self._sign_view_clipped else self.latest_sign_dist)

    def _arbitrate_pending_sign(self):
        """Fuse the distance-gated LATCHED label (sign_latch, fed from /vlm/sign) with
        the geometrically localized board distance (latest_sign_dist) into pending_sign
        -- the candidate maneuver to ARM while its board is genuinely close ahead.

        Rule: pending_sign = the latched label while its board is close ahead
        (0 < dist < sign_reach_dist_m); 'none' when no board is localized in reach.
        The latch itself only changes while the board is in the readable band
        (sign_label_min_dist_m..max), which lies ABOVE the reach threshold — so
        inside reach the label is frozen and pending_sign cannot flicker (the old
        sign_locked flag is obsolete). A read taken early (board 3.3-5 m out) is
        preserved here even though the VLM may only return 'none' by the time the
        board is in reach (the pitched camera crops it — the stop-sign dropout).
        The maneuver FSM commits the candidate on the sign_commit_dist_m edge."""
        label = self.sign_latch.label
        dist = self.latest_sign_dist
        if (label is not None
                and dist is not None and 0.0 < dist < self.sign_reach_dist_m):
            self.pending_sign = label
        elif dist is None or dist >= self.sign_reach_dist_m or dist <= 0.0:
            self.pending_sign = 'none'
        self.pending_sign_pub.publish(String(data=self.pending_sign))
        # Drive the state machine ENTRY: arm the candidate + commit while the board
        # is inside commit range. Only a CONSUMED commit (label was armed) retires
        # the trigger for this board — see _update_sign_odom.
        if self.maneuver_sm.on_sign(self.pending_sign, self._sign_commit_event):
            self._sign_committed = True

    def _maneuver_target_speed(self):
        """Map the committed maneuver state to a target speed [m/s]. left/right -> the
        turn speed; winding -> the (slower) winding speed; stop -> 0; straight/default ->
        the straight cruising speed."""
        return {
            'left': self.speed_turn,
            'right': self.speed_turn,
            'winding': self.speed_winding,
            'stop': self.speed_stop,
        }.get(self.maneuver_sm.state, self.speed_straight)

    def _publish_target_speed(self):
        """Publish the maneuver-derived speed setpoint AND the raw maneuver state for
        the MPC. Called EVERY plan tick (in all modes/branches) so the MPC's freshness
        checks always see live values at planner rate; if the planner dies the MPC
        falls back to its own defaults and never accelerates (fail-safe invariant).
        The state feeds the MPC's maneuver-dependent heading lookahead
        (heading_lookahead_straight/turn/winding_m in mpc_params.yaml)."""
        self.speed_pub.publish(Float64(data=float(self._maneuver_target_speed())))
        self.state_pub.publish(String(data=self.maneuver_sm.state))

    def plan_once(self):
        # Optional cross-node startup gate: withhold the DRIVING outputs (path +
        # speed setpoint) until the VLM node has produced its first sign, so the
        # car does not drive before the VLM is up. Perception, the sign latch and
        # the FPV debug overlay still run while gated, so RViz shows the camera
        # view during the ~1 min Qwen model load (the opencv branch returns after
        # the overlay, before any path is planned). Default off
        # (wait_for_first_sign=false) -> standalone planner.
        gated = self.wait_for_first_sign and not self._sign_seen
        if gated:
            self.get_logger().info(f'Waiting for first sign on {self.sign_topic} '
                                   'before publishing a path...',
                                   throttle_duration_sec=3.0)
        else:
            # Phase 5: publish the maneuver speed setpoint every tick (before the
            # mode branches, so early returns / fake_path still keep the MPC's
            # setpoint fresh). Uses the maneuver state committed on the previous
            # tick (<=100 ms lag, harmless for a speed target).
            self._publish_target_speed()

        if self.mode == 'fake_path':
            if gated:
                return
            path = self._make_fake_path()
            self.path_pub.publish(path)
            return

        if self.latest_image is None:
            self.get_logger().warn('No image received yet, skipping planning.',
                                   throttle_duration_sec=5.0)
            return

        if self.mode == 'opencv':
           if self.latest_depth is None or self.latest_cam_info is None:
               self.get_logger().warn('No depth or camera info received yet, skipping planning.')
               return

           img_msg = self.latest_image                  # snapshot: stamp must match the processed frame
           bgr = img_msg_to_bgr(img_msg)                # cv_bridge-free (works under venv NumPy 2)
           depth_np = img_msg_to_depth(self.latest_depth)  # cv_bridge-free
           h,w = bgr.shape[:2]

           from vlm_planner_py.opencv_fallback import detect_yellow_cones, detect_red_cones, detect_track_lines # type: ignore[import-untyped]

           # Lane markings: gray boundary LINES (default) or legacy yellow CONES.
           # Both yield a pixel list that is projected to metric base_link below;
           # only the source detector and the centerline builder differ.
           if self.line_detection:
               lane_px = detect_track_lines(bgr, val_min=self.line_val_min,
                                            sat_max=self.line_sat_max,
                                            sample_stride=self.line_sample_stride,
                                            area_min_px=self.line_area_min_px,
                                            roi_top_px=self.line_roi_top_px)
           else:
               lane_px = detect_yellow_cones(bgr, area_min=self.cone_area_min_px,
                                             area_max=self.cone_area_max_px)

           # The camera->base transform and corrected intrinsics are needed whether
           # we follow the yellow lane or look for the red goal, so compute once.
           stamp = self.get_clock().now()
           camera_to_base = build_camera_to_base_transform(self.tf_buffer,
                                                           self.camera_frame,
                                                           self.base_link_frame,
                                                           stamp.to_msg())
           if camera_to_base is None:
               self.get_logger().warn('Could not get camera to base transform, skipping planning.')
               return

           # Gazebo Fortress publishes a CameraInfo whose K is for a stale
           # default resolution (320x240) while the image actually renders at
           # the configured size — so cx/fx are wrong for these pixels. The
           # published cx/fx RATIO is resolution-independent, though, so rescale
           # fx to the real image width instead of trusting K outright (and
           # without re-hardcoding the URDF hfov here).
           ci = self.latest_cam_info
           fx_pub, cx_pub = ci.k[0], ci.k[2]  # published focal length and principal point x (pixels)
           if fx_pub > 0.0 and cx_pub > 0.0:
               self.get_logger().info('Rescaling published camera_info K from '
                                      f'fx={fx_pub:.1f}, cx={cx_pub:.1f} to match actual image width {w}px.',
                                      throttle_duration_sec=5.0)
               fx_use = fx_pub * (w / (2.0 * cx_pub))  # e.g. 277 * 640/320 = 554
           else:
               hfov = 1.047  # rad — fallback to URDF <horizontal_fov> if K empty
               fx_use = (w / 2.0) / math.tan(hfov / 2.0) #
           cx_use, cy_use = w / 2.0, h / 2.0

           # Robot pose — needed to latch the red goal in odom and to place any
           # path in odom.
           if self.latest_odom is None:
               self.get_logger().warn('No odometry available, skipping planning.')
               return
           p = self.latest_odom.pose.pose.position
           o = self.latest_odom.pose.pose.orientation
           yaw = quat2euler((o.w, o.x, o.y, o.z))[2]
           cos_y, sin_y = math.cos(yaw), math.sin(yaw)

           # --- HIGHEST PRIORITY: stop before the red goal cone, even if the
           #     yellow lane is still in view. Detect + latch the goal in odom,
           #     then stop on odometry distance (robust to the cone leaving the
           #     camera view as we close in). This check runs BEFORE lane following
           #     so the goal always wins over finishing the yellow path.
           red_cones = detect_red_cones(bgr, area_min=self.cone_area_min_px,
                                        area_max=self.cone_area_max_px)

           # Project both detections to metric base_link WITH the height gate, so
           # elevated sign boards are dropped before they can be treated as cones
           # or as the goal. return_pixels=True keeps the source centroids for the
           # debug overlay; planning below uses the (x, y) pairs.
           red_proj = (project_pixels_to_base_link(red_cones, depth_np,
                                                   fx_use, cx_use, cy_use,
                                                   camera_to_base, h, w,
                                                   depth_window=self.depth_window_px,
                                                   z_min=self.cone_height_min_m,
                                                   z_max=self.cone_height_max_m,
                                                   return_pixels=True)
                       if red_cones else [])
           lane_proj = (project_pixels_to_base_link(lane_px, depth_np,
                                                    fx_use, cx_use, cy_use,
                                                    camera_to_base, h, w,
                                                    depth_window=self.depth_window_px,
                                                    z_min=self.line_z_min_m,
                                                    z_max=self.line_z_max_m,
                                                    return_pixels=True)
                        if lane_px else [])
           red_base = [(x, y) for (x, y, _px, _py) in red_proj]
           lane_base = [(x, y) for (x, y, _px, _py) in lane_proj]

           # --- Phase 2 sign localization: detect the elevated sign boards (yellow
           # = turn/winding, red = stop). Project ALL candidates with a WIDE z so
           # the per-stage debug shows where each one lands, then keep only those at
           # board height (the real z-gate) for the latch. The ground goal cone /
           # lane markings land near z~0 and are dropped by the gate.
           sign_yellow = detect_yellow_cones(bgr, area_min=self.sign_area_min_px,
                                             area_max=self.sign_area_max_px)
           sign_red = detect_red_cones(bgr, area_min=self.sign_area_min_px,
                                       area_max=self.sign_area_max_px)
           sign_px = sign_yellow + sign_red
           sign_all = (project_pixels_to_base_link(sign_px, depth_np,
                                                   fx_use, cx_use, cy_use,
                                                   camera_to_base, h, w,
                                                   depth_window=self.depth_window_px,
                                                   z_min=-2.0, z_max=3.0,
                                                   return_pixels=True, with_z=True)
                       if sign_px else [])
           sign_proj = [s for s in sign_all
                        if self.sign_z_min_m < s[2] < self.sign_z_max_m]
           # Per-stage debug so a missing banner distance can be diagnosed: how many
           # raw blobs, how many got valid depth (projected), how many sit at board
           # height (in the z-gate), and the (x,y,z) each candidate landed at.
           self.get_logger().info(
               f'[sign] blobs y={len(sign_yellow)} r={len(sign_red)} | '
               f'projected(valid depth)={len(sign_all)} | '
               f'in z-gate[{self.sign_z_min_m:.2f},{self.sign_z_max_m:.2f}]={len(sign_proj)} | '
               f'cand(x,y,z)={[(round(x,2), round(y,2), round(z,2)) for (x, y, z, _, _) in sign_all]}',
               throttle_duration_sec=1.0)
           frame_t = (img_msg.header.stamp.sec
                      + img_msg.header.stamp.nanosec * 1e-9)
           self._update_sign_odom(sign_proj, p, cos_y, sin_y, frame_t=frame_t,
                                  fx=fx_use, img_wh=(w, h))
           # Fuse the VLM label with the localized board distance into pending_sign and
           # drive the maneuver state-machine ENTRY (arm + commit on the pass edge).
           self._arbitrate_pending_sign()

           # FPV overlay for RViz: raw detections (hollow) vs. height-accepted
           # (filled). A yellow sign board shows as a hollow-yellow mark but never
           # a filled-green cone, so the filter's effect is visible.
           if self.publish_debug_image:
               self._publish_debug_overlay(bgr, lane_px, lane_proj,
                                           red_cones, red_proj)

           # Startup gate (see top of plan_once): perception + overlay ran, but
           # no path may be planned/published until the VLM's first read.
           if gated:
               self.tap.put(outcome='gated_waiting_vlm')
               self.tap.flush()
               return

           if red_base:
               # Latch / refresh the goal position in odom (the cone is static).
               nearest_red = min(red_base, key=lambda c: math.hypot(c[0], c[1]))
               self.goal_odom = (p.x + cos_y * nearest_red[0] - sin_y * nearest_red[1],
                                 p.y + sin_y * nearest_red[0] + cos_y * nearest_red[1])

           goal_dist = None
           if self.goal_odom is not None:
               gx, gy = self.goal_odom[0] - p.x, self.goal_odom[1] - p.y
               goal_dist = math.hypot(gx, gy)
               if goal_dist <= self.goal_stop_dist_m:
                   self.get_logger().info(f'Red goal cone close ({goal_dist:.2f} m); stopping '
                                          '(overrides lane following).',
                                          throttle_duration_sec=2.0)
                   # Publish a halt path ENDING at the robot so the MPC's
                   # "reached end of path" logic brakes actively (zero command)
                   # instead of coasting on the stale creep path for path_timeout.
                   self._publish_halt(p)
                   self.tap.put(outcome='halt_goal', goal_dist=round(goal_dist, 3))
                   self.tap.flush()
                   return

           if lane_base:
               # --- Normal lane following: build a centerline from the
               # height-accepted lane markings (already projected to metric
               # base_link above, with the sign boards filtered out).
               if self.line_detection:
                   # Continuous gray lines: cluster into boundary chains and
                   # offset half a lane inward (constant-width => exact centre).
                   # Stash the cloud; the actual dump happens in _publish_base_path
                   # once the smoothed/published path is also available, so one
                   # record holds cloud + raw centerline + published path.
                   if self.lane_cloud_dump_dir:
                       self._dump_lane_base = lane_base
                   # debug= is only requested when the tap is on (None => zero
                   # overhead inside centerline_from_line_cloud).
                   cl_dbg = {} if self.tap.enabled else None
                   base_pts = centerline_from_line_cloud(lane_base,
                                                         half_lane=self.half_lane_m,
                                                         d_max=self.line_d_max_m,
                                                         side_memory=self._lane_side_memory,
                                                         debug=cl_dbg)
                   if cl_dbg is not None:
                       sides = [c['side'] for c in cl_dbg.get('classified', [])]
                       self.tap.put(
                           n_cloud=len(lane_base),
                           n_gridded=len(cl_dbg.get('gridded', [])),
                           n_chains_raw=len(cl_dbg.get('chains_raw', [])),
                           n_chains=len(cl_dbg.get('chains_resampled', [])),
                           two_boundaries=(len(set(sides)) == 2),
                           side_thresh=cl_dbg.get('thresh'),
                           side_mem_left=self._lane_side_memory.get('left'),
                           side_mem_right=self._lane_side_memory.get('right'),
                           n_centerline=len(base_pts),
                           centerline=[[round(x, 3), round(y, 3)]
                                       for (x, y) in base_pts])
               else:
                   # Discrete cones: heading-relative gate-walk.
                   base_pts = build_centerline_gatewalk(lane_base,
                                                        half_lane=self.half_lane_m,
                                                        d_min=self.gate_d_min_m,
                                                        d_max=self.gate_d_max_m,
                                                        max_pts=self.max_centerline_pts)
               # print(f"Centerline base points: {base_pts}")
               now_s = self.get_clock().now().nanoseconds * 1e-9
               # A centerline that turns >90 deg between segments is geometric
               # garbage (chains hopped between boundaries / x-monotonic
               # assumption broke at the U-turn apex) — publishing it slams the
               # controller side to side. Reject the frame instead.
               fold_back = len(base_pts) >= 2 and path_folds_back(base_pts)
               self.tap.put(fold_back=fold_back)
               if len(base_pts) >= 2 and not fold_back:
                   # is_lane=True -> this real centerline feeds the maneuver EXIT
                   # (straight-again) detector; the creep/halt paths below do not.
                   if self._publish_base_path(base_pts, is_lane=True):
                       self._lane_hold_until = now_s + self.max_reuse_time
                       self.tap.put(outcome='published')
                       self.tap.flush()
                       return
               # Markings seen but unusable this frame. Within the reuse window,
               # HOLD: publish nothing and let the MPC keep tracking the last
               # good path — far better than replacing it with a creep line.
               if self.reuse_last_path and now_s < self._lane_hold_until:
                   self.get_logger().warn('Unusable centerline (fold-back); holding last path.',
                                          throttle_duration_sec=1.0)
                   self.tap.put(outcome='hold_last_path')
                   self.tap.flush()
                   return
               # Hold window expired -> fall through to creep.
               self.get_logger().warn('Lane markings seen but no usable centerline; creeping.',
                                      throttle_duration_sec=2.0)

           # --- No usable yellow lane: creep slowly toward the red goal (if seen)
           #     or straight ahead until it appears. Drop the boundary side
           #     memory: after a detection gap the robot may have moved/yawed
           #     enough that the stored laterals would mis-side the next lone
           #     boundary.
           self._lane_side_memory.clear()
           if self.goal_odom is not None:
               # Creep toward the latched goal, expressed back in base_link.
               bx = cos_y * gx + sin_y * gy
               by = -sin_y * gx + cos_y * gy
               creep_pts = [(0.0, 0.0), (bx, by)]
               self.get_logger().warn(f'No yellow lane; creeping toward red goal ({goal_dist:.2f} m).',
                                      throttle_duration_sec=1.0)
               self.tap.put(outcome='creep_goal')
           else:
               # Goal never seen yet: creep ahead at the current heading — but
               # while a turn/winding maneuver is COMMITTED, arc gently toward
               # its side instead of straight. At a sharp corner apex the
               # single visible boundary chain collapses below the usable
               # minimum, this fallback takes over, and creeping STRAIGHT
               # aims off-track (loop-6 autotune: watchdog lane_lost mid
               # right-corner). The maneuver direction is ground truth the
               # sign committed; the arc only biases the blind fallback, the
               # visible-lane path stays the plain centerline.
               state = self.maneuver_sm.state
               turn = {'left': 1.0, 'right': -1.0}.get(state, 0.0)
               if turn:
                   n = 5
                   r = self.creep_len_m / 0.9   # ~52 deg of arc over the creep
                   creep_pts = [(r * math.sin(0.9 * i / (n - 1)),
                                 turn * r * (1.0 - math.cos(0.9 * i / (n - 1))))
                                for i in range(n)]
                   self.get_logger().warn(
                       f'No lane; creeping in committed {state} arc.',
                       throttle_duration_sec=1.0)
                   self.tap.put(outcome=f'creep_{state}')
               else:
                   creep_pts = [(0.0, 0.0), (self.creep_len_m, 0.0)]
                   self.get_logger().warn('No yellow lane and no red goal seen yet; creeping straight.',
                                          throttle_duration_sec=1.0)
                   self.tap.put(outcome='creep_straight')

           self._publish_base_path(creep_pts)
           self.tap.flush()
           return

    def _mpc_dbg_cb(self, name, msg):
        self._mpc_dbg[name] = (float(msg.data),
                               self.get_clock().now().nanoseconds * 1e-9)

    def _mpc_dbg_get(self, name, max_age_s=1.0):
        """Latest MPC debug value, or None if never received / stale."""
        v = self._mpc_dbg.get(name)
        if v is None:
            return None
        now_s = self.get_clock().now().nanoseconds * 1e-9
        return v[0] if (now_s - v[1]) <= max_age_s else None

    def _publish_debug_overlay(self, bgr, yellow_px, yellow_proj, red_px, red_proj):
        """Republish the front camera image with the cone/goal detections drawn on
        it. Raw detections (incl. elevated sign boards) are hollow circles;
        height-accepted ones are filled. So a sign board shows as a hollow mark
        but never a filled cone — the height filter's effect is visible in the
        RViz FPV image. (The goal keeps a 'GOAL' tag; no range numbers.)"""

        # dictionary of mapping sign label to driving instructions.
        sign_label_instructions = {
            'stop': 'prepare to stop',
            'left': 'the car is turning left',
            'right': 'the car is turning right',
            'winding': 'the car is entering a winding road',
            'straight': 'the car is going straight',
            'none': 'line following'
        }

        dbg = bgr.copy()
        # Raw detections (may include the sign boards): hollow outlines.
        for (px, py) in yellow_px:
            cv2.circle(dbg, (int(px), int(py)), 8, (0, 255, 255), 2)   # yellow
        for (px, py) in red_px:
            cv2.circle(dbg, (int(px), int(py)), 8, (255, 0, 255), 2)   # magenta
        # Height-accepted detections: filled markers (no range label).
        for (x, y, px, py) in yellow_proj:
            cv2.circle(dbg, (int(px), int(py)), 6, (0, 200, 0), -1)    # green
        for (x, y, px, py) in red_proj:
            cv2.circle(dbg, (int(px), int(py)), 6, (0, 0, 255), -1)    # red
            cv2.putText(dbg, 'GOAL', (int(px) + 9, int(py) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        # Latest VLM sign label (from /vlm/sign) as a top banner, so the detection
        # image and the sign read share one publisher. Persists between VLM ticks;
        # absent until the first /vlm/sign (i.e. when no vlm_sign node is running).
        # Cyan ring on the nearest localized sign board (Phase 2).
        if self._sign_px is not None:
            cv2.circle(dbg, (int(self._sign_px[0]), int(self._sign_px[1])),
                       12, (255, 255, 0), 2)  # cyan
        if self.latest_sign_label:
            # Two-line banner. Line 1: the maneuver the car is committed to (state
            # machine) + its instruction. Line 2: diagnostics — the raw VLM read, the
            # armed/pending candidate sign, and the localized board distance.
            lbl = self.maneuver_sm.state
            color = (0, 0, 255) if lbl in ('straight', 'none', 'unparsed') else (0, 255, 0)
            line1 = f'VLM: {sign_label_instructions.get(lbl, "unknown instruction")}'
            dist_s = ('n/a' if self.latest_sign_dist is None
                      else f'{self.latest_sign_dist:.1f}m')
            latched = self.sign_latch.label or '-'
            line2 = (f'Sign={self.latest_sign_label}  latched={latched}  '
                     f'confirmed ={self.pending_sign}  d={dist_s}')
            cv2.rectangle(dbg, (0, 0), (dbg.shape[1], 92), (0, 0, 0), -1)
            cv2.putText(dbg, line1, (10, 36),  # line 1 baseline ~36 px
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            cv2.putText(dbg, line2, (10, 74),  # line 2 baseline ~74 px
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

        # Control readout (ALWAYS shown, independent of the sign banner). Line 1:
        # ref = the MPC's RAMPED (slew-limited) target speed — what the solver is
        # actually chasing this tick, not the raw maneuver setpoint (falls back to
        # the raw setpoint, labelled 'tgt', when the MPC readout is absent/stale);
        # act = measured odom speed. Line 2: the MPC's raw QP commands.
        vref = self._mpc_dbg_get('vref')
        act_v = None
        if self.latest_odom is not None:
            tw = self.latest_odom.twist.twist.linear
            act_v = math.hypot(tw.x, tw.y)
        act_s = 'n/a' if act_v is None else f'{act_v:.2f}'
        if vref is not None:
            spd_line = f'spd ref={vref:.2f} act={act_s} m/s  [{self.maneuver_sm.state}]'
        else:
            spd_line = (f'spd tgt={self._maneuver_target_speed():.2f} act={act_s} '
                        f'm/s  [{self.maneuver_sm.state}]')
        accel = self._mpc_dbg_get('accel')
        steer = self._mpc_dbg_get('steer')
        cmd_line = ('cmd accel=' + ('n/a' if accel is None else f'{accel:+.2f}m/s2')
                    + '  steer=' + ('n/a' if steer is None else f'{steer:+.3f}rad'))
        h_img = dbg.shape[0]
        cv2.rectangle(dbg, (0, h_img - 64), (dbg.shape[1], h_img), (0, 0, 0), -1)
        cv2.putText(dbg, spd_line, (10, h_img - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(dbg, cmd_line, (10, h_img - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        try:
            header = self.latest_image.header if self.latest_image is not None else None
            self.debug_image_pub.publish(bgr_to_img_msg(dbg, header=header))  # cv_bridge-free
        except Exception as e:  # drawing/encoding must never crash the planner
            self.get_logger().warn(f'Failed to publish debug overlay: {e}',
                                   throttle_duration_sec=5.0)

    def _publish_halt(self, p):
        """Publish a single-pose path AT the robot's current odom position. The MPC
        treats the path's last point as the goal and brakes (zero command) once the
        robot is within goal_tolerance of it — so this actively stops the car rather
        than letting it coast on the last forward-pointing path."""
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = cast(str, self.output_frame)
        ps = PoseStamped()
        ps.header = path.header
        ps.pose.position.x = p.x
        ps.pose.position.y = p.y
        ps.pose.orientation.w = 1.0
        path.poses.append(ps)
        self.path_pub.publish(path)

    def _publish_base_path(self, base_pts, is_lane=False):
        """Smooth/resample/validate a base_link (x, y) path, transform it into the
        odom frame with the current robot pose, and publish it. Shared by both the
        yellow-lane centerline and the red-goal creep path. Returns True if a path
        was published.

        is_lane=True marks this as a REAL lane centerline, so its (smoothed, resampled)
        geometry -- exactly what becomes /vlm_path_odom -- feeds the maneuver EXIT
        (straight-again) detector. Creep/halt paths leave is_lane=False: they are
        straight by construction and must NOT be allowed to release a maneuver."""
        if self.latest_odom is None:
            self.get_logger().warn('No odometry available to anchor path, skipping planning.')
            return False
        p = self.latest_odom.pose.pose.position
        o = self.latest_odom.pose.pose.orientation
        yaw = quat2euler((o.w, o.x, o.y, o.z))[2]

        from vlm_planner_py.path_smoother import (smooth_points, fit_polyline, resample_uniform, validate_path, build_path_msg_with_headings) # type: ignore[import-untyped]

        # base_pts already starts at the robot origin (0,0), so no prepend.
        # Stateless polynomial fit (path_fit_degree > 0) de-noises per-frame depth
        # jitter with no temporal lag; degree 0 -> moving average.
        if self.path_fit_degree > 0:
            smooth_pts = fit_polyline(base_pts, degree=self.path_fit_degree)
        else:
            smooth_pts = smooth_points(base_pts, window=self.smooth_window)
        resampled_pts = resample_uniform(smooth_pts, spacing=self.path_spacing_m)
        # print(f"Resampled points: {resampled_pts}")
        # Validate in base_link: validate_path's checks (x<0 = behind robot,
        # |y| lateral offset) are only meaningful in the robot frame.
        valid, msg = validate_path(resampled_pts, max_dist=self.path_max_len_m)
        if not valid:
            self.get_logger().warn(f'Path validation failed: {msg}')
            return False

        # Offline debug: dump cloud + raw centerline (base_pts) + published path
        # (resampled_pts) together, so test/debug_centerline.py can localize any
        # near-field bias to the centerline vs the smoother. is_lane only (creep/halt
        # paths have no lane cloud).
        if self.lane_cloud_dump_dir and is_lane:
            self._dump_lane_cloud(getattr(self, '_dump_lane_base', []),
                                  centerline_raw=base_pts, published=resampled_pts)

        # Maneuver EXIT: only the real lane centerline (is_lane) may release a
        # turn/winding maneuver back to 'straight'. step_dist = robot travel since
        # the last lane frame, so the release debounce
        # is in metres of straight road (a frame count would false-release at a winding
        # inflection). A large gap (lane lost for a while -> odom jump) is clamped so it cannot
        # count as a long straight stretch.
        if is_lane:
            step_dist = 0.0
            if self._last_lane_pos is not None:
                step_dist = math.hypot(p.x - self._last_lane_pos[0],
                                       p.y - self._last_lane_pos[1])
                step_dist = min(step_dist, self.path_spacing_m)
            self._last_lane_pos = (p.x, p.y)
            self.maneuver_sm.on_path(resampled_pts, step_dist)

        # Transform base_link -> odom using the robot pose, then publish.
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        odom_pts = [(p.x + cos_y * x - sin_y * y,
                     p.y + sin_y * x + cos_y * y) for (x, y) in resampled_pts]
        path_msg = build_path_msg_with_headings(odom_pts,
                                                frame_id=cast(str, self.output_frame),
                                                stamp=self.get_clock().now().to_msg())
        if not path_msg:
            return False
        self.path_pub.publish(path_msg)
        return True

    def _on_params_set(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name == 'lane_cloud_dump_dir':
                self.lane_cloud_dump_dir = str(p.value or '')
                self.get_logger().info(
                    f"lane_cloud_dump_dir -> {self.lane_cloud_dump_dir!r}")
        return SetParametersResult(successful=True)

    def _dump_lane_cloud(self, lane_base, centerline_raw=None, published=None):
        """Append one planning frame to <lane_cloud_dump_dir>/lane_cloud.jsonl, one
        JSON object per line:
            {frame, half_lane, d_max,
             points:        [[x,y],...]  base_link lane cloud (input to centerline),
             centerline_raw:[[x,y],...]  centerline_from_line_cloud output (pre-smooth),
             published:     [[x,y],...]  smoothed+resampled path = what becomes /vlm_path_odom}
        Recording both centerline_raw and published lets test/debug_centerline.py
        show, in ONE frame, whether a near-field bias is in the centerline itself
        (detection/offset) or introduced by the smoother (moving-average corner-cut).
        Opt-in via the lane_cloud_dump_dir param; a no-op when that is empty."""
        try:
            os.makedirs(self.lane_cloud_dump_dir, exist_ok=True)
            path = os.path.join(self.lane_cloud_dump_dir, 'lane_cloud.jsonl')
            rec = {'frame': self._lane_dump_idx,
                   # sim-time stamp so the dump can be cross-referenced with a
                   # debugkit session (signals/*.csv share this clock).
                   't': self.get_clock().now().nanoseconds * 1e-9,
                   'half_lane': self.half_lane_m,
                   'd_max': self.line_d_max_m,
                   'points': [[float(x), float(y)] for (x, y) in lane_base]}
            if centerline_raw is not None:
                rec['centerline_raw'] = [[float(x), float(y)] for (x, y) in centerline_raw]
            if published is not None:
                rec['published'] = [[float(x), float(y)] for (x, y) in published]
            with open(path, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            self._lane_dump_idx += 1
        except OSError as e:
            self.get_logger().warn(
                f'lane cloud dump failed: {e}', throttle_duration_sec=5.0)

    def _make_fake_path(self):
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.output_frame

        # Anchor the path at the robot's current pose/heading (from odom_ekf).
        # Until the first odom arrives, fall back to the origin facing +x.
        if self.latest_odom is None:
            self.get_logger().warn(
                f'No odometry on {self.odom_topic} yet, anchoring fake path at origin.',
                throttle_duration_sec=5.0)
            x0, y0, yaw = 0.0, 0.0, 0.0
        else:
            p = self.latest_odom.pose.pose.position
            x0, y0 = p.x, p.y
            o = self.latest_odom.pose.pose.orientation
            yaw = quat2euler((o.w, o.x, o.y, o.z))[2]

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        # Heading quaternion shared by every pose (path runs straight ahead).
        qw, _, _, qz = euler2quat(0.0, 0.0, yaw)

        num_points = 20
        poses: list[PoseStamped] = []
        for i in range(num_points + 1):
            d = i * (self.max_path_dist / num_points)
            pose = PoseStamped()
            pose.header = path.header
            # Straight line ahead in the robot's local frame, rotated into odom.
            pose.pose.position.x = x0 + d * cos_yaw
            pose.pose.position.y = y0 + d * sin_yaw
            pose.pose.position.z = 0.0
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            poses.append(pose)
        path.poses = poses

        return path


def main(args=None):
    rclpy.init(args=args)
    node = VlmPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
