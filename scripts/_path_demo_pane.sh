#!/bin/bash
# ---------------------------------------------------------------------------
# _path_demo_pane.sh  N
# Helper invoked by start_path_demo.sh inside each tmux pane. Sources
# ROS 2 + the local workspace (system python, NO venv), waits its staggered
# delay, then runs one process of the interactive path-drawing demo.
# Kept separate so each tmux pane command stays a single simple invocation.
# ---------------------------------------------------------------------------
# Keep the stray ~/.local numpy 2.2.6 (user site-packages, installed for the
# Jupyter notebook work) out of the system-python ROS nodes: apt's transforms3d
# / cv2 are numpy-1-only and crash at import under it (kills cone_markers,
# odom_path_rezero, fpv_truth_overlay -> blank FPV overlay + no ground-truth
# odom in RViz, 2026-07-17). Venv panes are unaffected (venv excludes user site).
export PYTHONNOUSERSITE=1

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"

case "${1:-}" in
  1) echo "[1] Gazebo (track_lines) + bridge + path_marker_node + spawn_robot"
     # Clean slate (pane 1 ONLY — a pkill in every pane would kill the live
     # server on any later single-pane restart): a leftover ign server from an
     # earlier run (e.g. the urban course demo) makes the GUI/spawn attach to
     # the WRONG world.
     pkill -9 -f "ign gazebo" 2>/dev/null
     sleep 5
     ros2 launch robotcar_gazebo gazebo_spawn_robot.launch.py world:=track_lines.world.sdf spawn_x:=-1.5 spawn_y:=0.0  ;;

  2) echo "[2] EKF localization -> /odom_ekf"; sleep 20
     ros2 launch robotcar_localization localization.launch.py ;;
#   3) echo "[3] pure_pursuit controller"; sleep 20
#      ros2 run mpc_tracker_cpp pure_pursuit_node ;;

  3) if [ "${MANUAL_DRIVE:-0}" = "1" ]; then
       echo "[3] controller DISABLED (MANUAL_DRIVE=1) -- drive with teleop in pane 8"
     else
       echo "[3] mpc controller"; sleep 25
       # Tune MPC by editing the SOURCE yaml below + relaunching — no rebuild.
       ros2 launch mpc_tracker_cpp mpc_tracker.launch.py \
         params_file:="$WS/src/mpc_tracker_cpp/config/mpc_params.yaml";
     fi ;;

  4) # Path PLANNER (system python). Same launch file in both cases; when VLM_SIGN=1
     # it is GATED on the separate VLM sign node in pane 5 (withholds the path until
     # the first /vlm/sign, so the car does not drive before the VLM is up).
     GATE=false; [ "${VLM_SIGN:-1}" = "1" ] && GATE=true
     echo "[4] VLM planner (wait_for_first_sign=$GATE)"; sleep 30
     # Tune the planner by editing the SOURCE yaml below + relaunching — no rebuild.
     ros2 launch vlm_planner_py vlm_planner.launch.py \
       params_file:="$WS/src/vlm_planner_py/config/vlm_params.yaml" \
       wait_for_first_sign:=$GATE ;;

  5) if [ "${VLM_SIGN:-1}" = "1" ]; then
       # VLM SIGN reader (Qwen) in the venv -> /vlm/sign (String) + /vlm/sign_image
       # (RViz "Sign Label (VLM)" display). Read-only; the planner (pane 4) consumes
       # /vlm/sign. Toggle the query live:  ros2 param set /vlm_sign vlm_query_enabled false
       #   VLM_QUERY=0 launches with the query OFF.
       # Launch with the VENV python via `-m` (ros2 run's console-script shebang is
       # the SYSTEM python -> no torch). cv_bridge-free, so the venv NumPy 2 is fine.
       VLM_QUERY="${VLM_QUERY:-true}"; [ "$VLM_QUERY" = "0" ] && VLM_QUERY=false
       echo "[5] VLM sign node (query=$VLM_QUERY, Qwen in venv) -> /vlm/sign"; sleep 0
       source "$HOME/venvs/vlm_robot/bin/activate"
       # Tune the reader (vlm_query_rate_hz, ...) by editing the SOURCE yaml
       # below + relaunching this pane — no rebuild.
       python3 -m vlm_planner_py.vlm_sign_node --ros-args \
         --params-file "$WS/src/vlm_planner_py/config/vlm_params.yaml" \
         -p vlm_query_enabled:="$VLM_QUERY"
     else
       echo "[5] free terminal -- run manual commands here"
     fi ;;

  6) echo "[6] RViz"; sleep 20
     ros2 launch robot_bringup rviz.launch.py ;;

  7) if [ "${RECORD_SESSION:-0}" = "1" ]; then
       # debugkit recording -- the "full" preset (core+paths+camera+vlm+debug, see
       # tools/debugkit/topics.yaml); runs until this pane/session is killed.
       # Override the label with DEMO_LABEL=<name> before launching.
       echo "[7] record_session (preset=full) -- waiting for go"; sleep 60
       "$WS/scripts/record_session.sh" --preset full --label "${DEMO_LABEL:-demo1}" --wait-for-go
     else
       echo "[7] record_session DISABLED (RECORD_SESSION=0) -- free terminal"; sleep 60
     fi ;;

  8) if [ "${MANUAL_DRIVE:-0}" = "1" ]; then
       echo "[8] manual teleop -- keep this pane focused; w/s drive, a/d steer, space stop"; sleep 40
       ros2 run robotcar_utils_py teleop_keyboard
     else
       echo "[8] free terminal -- run manual commands here"
     fi ;;
  *) ;;
esac

# Keep the pane open after the process exits so errors stay visible.
exec bash
