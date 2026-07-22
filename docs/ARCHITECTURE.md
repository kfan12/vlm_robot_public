# Architecture

How the demo works, node by node. Three layers — perception, control,
simulation — plus an optional VLM sign pipeline that modulates speed.

```
                    ┌───────────────────────────────────────────────┐
                    │ GAZEBO FORTRESS  (track_lines world)          │
                    │  AckermannSteering · RGBD camera · IMU ·      │
                    │  odometry plugins · sign models               │
                    └───────┬───────────────────────────▲───────────┘
              ros_gz_bridge │ /camera/front/*  /odom     │ /cmd_vel
                            │ /imu  /odom_truth  /clock  │
        ┌───────────────────▼────────────┐               │
        │ PERCEPTION  vlm_node (opencv)  │               │
        │  depth-projected lane points   │               │
        │  → gate-walk centerline        │               │
        │  → smoothed path in odom frame │               │
        └───────┬───────────────┬────────┘               │
   /vlm_path_odom (Path)   /maneuver/target_speed        │
        ┌───────▼───────────────▼────────┐        ┌──────┴──────────┐
        │ CONTROL  mpc_tracker_node      │        │ LOCALIZATION    │
        │  linearized bicycle model      │◄───────┤ ekf_node        │
        │  OSQP QP over horizon          │ /odom_ekf  (/odom + /imu)│
        │  → v, steering → /cmd_vel      │        └─────────────────┘
        └────────────────────────────────┘
```

## Topic map

| Topic | Type | Producer → Consumer |
|---|---|---|
| `/camera/front/image_raw`, `/camera/front/depth/image_raw`, `/camera/front/camera_info` | Image / CameraInfo | Gazebo (bridged) → planner, sign node, capture |
| `/odom` | Odometry | Gazebo wheel odometry (drifts) → EKF |
| `/imu` | Imu | Gazebo (noisy by design) → EKF |
| `/odom_ekf` | Odometry | EKF → planner (frame transform) + MPC (state) |
| `/odom_truth` | Odometry | Gazebo ground truth → RViz comparison, TF |
| `/vlm_path_odom` | Path | planner → MPC (reference path, odom frame) |
| `/cmd_vel` | Twist | MPC (or teleop) → Gazebo Ackermann plugin |
| `/vlm/debug_image` | Image | planner → RViz (FPV overlay: detections, path) |
| `/vlm/sign`, `/vlm/sign_image`, `/vlm/sign_roi` | String / Image / String(JSON) | sign pipeline (VLM_SIGN=1 only) |
| `/maneuver/target_speed` | Float64 | planner's maneuver FSM → MPC speed setpoint |

## Perception — `vlm_planner_py/vlm_node` (mode `opencv`)

Runs at 10 Hz on the system Python (no GPU):

1. Threshold the lane markings in the RGB frame; project each detected pixel
   to a 3-D point with the aligned depth image (per-blob median depth), into
   `base_link`, then into `odom`.
2. Height gates reject non-lane objects (`line_z_min/max_m` keeps paint at
   z≈0; sign boards at z≈0.55 m are excluded from lane detection).
3. A **gate-walk centerline** pairs left/right markings and walks forward
   gate by gate (up to `line_d_max_m` = 2 m ahead — far depth is too noisy).
4. The centerline is smoothed (moving average, `smooth_window`) and resampled
   at `path_spacing_m`, then published as `/vlm_path_odom`.
5. Fold-back frames (centerline doubling back >90°) are rejected; the MPC
   keeps tracking the last good path for up to `max_reuse_time_sec`.

All knobs live in `src/vlm_planner_py/config/vlm_params.yaml` (heavily
commented — the tuning history is in the comments). Edit + relaunch the pane;
no rebuild needed.

## Control — `mpc_tracker_cpp/mpc_tracker_node`

- State: pose + speed from `/odom_ekf`; reference: nearest segment of
  `/vlm_path_odom`.
- Model: kinematic bicycle, linearized around the **current** speed each tick.
- A quadratic program over the prediction horizon (tracking error + control
  effort + **Δu control-rate smoothing**, including the `(u₀ − u_prev)²`
  term) is solved with **OSQP** (via osqp-eigen). The heading reference uses
  a **lookahead bearing** rather than the local tangent — the two together
  suppress reference-noise limit cycles (~11.5× less steering noise on a
  straight vs. the naive setup).
- If the QP ever fails, a proportional-on-errors fallback takes over for that
  tick.
- The target speed defaults to cruise and is overridden at runtime by
  `/maneuver/target_speed` (slew-rate-limited before entering the QP).

Tuning file: `src/mpc_tracker_cpp/config/mpc_params.yaml` (edit + relaunch,
no rebuild).

## Localization

`robot_localization`'s `ekf_node` fuses drifting wheel odometry (`/odom`)
with the noisy IMU (`/imu`) → `/odom_ekf` (30 Hz). The `odom → base_link` TF
comes from the ground-truth odometry plugin so RViz shows the true pose;
the EKF estimate is used by the planner and MPC as the *believed* state.
Config: `src/robotcar_localization/config/ekf.yaml`.

## Optional sign pipeline (`VLM_SIGN=1`)

```
camera ─► vlm_sign_node (Qwen2.5-VL-3B, 4-bit, in the torch venv)
              │  reads the sign label every ~2 s, cropped to the ROI
              ▼  /vlm/sign ("left"/"right"/"winding"/"stop"/"none")
        vlm_node sign latch ── localizes the board in odom, attributes each
              │                read to the right board, freezes the label
              │                once the board is close (label lock)
              ▼
        maneuver FSM (maneuver.py) ── commits left/right/winding/stop at
              │                        per-label distances; releases when the
              │                        road is straight again (hysteresis)
              ▼
        /maneuver/target_speed ──► MPC   (1.2 m/s straight · 0.6 turn/winding
                                          · 0.0 stop — direction always comes
                                          from the path, only speed is set)
```

The planner also hands the sign node a pixel ROI (`/vlm/sign_roi`) so Qwen
reads the *right* board instead of the nearest board-shaped object. All
distances/bands are in `vlm_params.yaml` with their rationale.

## tmux pane ↔ component map

| Pane | Component | Starts at |
|---|---|---|
| 1 | Gazebo + bridge + robot spawn (`gazebo_spawn_robot.launch.py`) | 5 s |
| 2 | EKF localization | 20 s |
| 3 | MPC controller (skipped when `MANUAL_DRIVE=1`) | 25 s |
| 4 | Lane planner (`vlm_node`, opencv mode) | 30 s |
| 5 | Qwen sign node (only when `VLM_SIGN=1`; otherwise free shell) | 0 s |
| 6 | RViz | 20 s |
| 7 | debugkit recording (only when `RECORD_SESSION=1`; otherwise free shell) | 60 s |
| 8 | Free shell (keyboard teleop when `MANUAL_DRIVE=1`) | 40 s |

The stagger matters: the bridge must be up before the spawn, and the planner
starts last so everything it needs is already publishing.
