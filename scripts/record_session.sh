#!/bin/bash
# record_session.sh — start a debugkit recording session (see docs/testing_pipeline.md).
# Sources ROS 2 + the workspace (system python, NO venv) and runs the recorder
# with use_sim_time so all timestamps share the sim clock epoch.
#
#   scripts/record_session.sh --preset core --label baseline
#   scripts/record_session.sh --preset core,paths,vlm --duration 120
#   scripts/record_session.sh --help
set -e
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
# A stray ~/.local numpy2 (pulled in by pandas, for unrelated Jupyter tooling)
# shadows apt's numpy1 on system python's sys.path, crashing apt's python3-opencv
# (numpy1 ABI) whenever record_session.py's image/depth handlers `import cv2`.
# PYTHONNOUSERSITE skips user site-packages for this process only.
export PYTHONNOUSERSITE=1
exec python3 "$WS/tools/debugkit/record_session.py" "$@" \
  --ros-args -p use_sim_time:=true
