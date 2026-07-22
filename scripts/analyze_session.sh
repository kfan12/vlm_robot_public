#!/bin/bash
# analyze_session.sh — run the debugkit analyzer on a recorded session.
# No ROS needed (pure numpy/matplotlib); runs on the system python.
#
#   scripts/analyze_session.sh latest
#   scripts/analyze_session.sh 20260705_141230_baseline
#   scripts/analyze_session.sh latest --only drift,oscillation
set -e
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# A stray ~/.local numpy2 (pulled in by pandas, for unrelated Jupyter tooling)
# shadows apt's numpy1 on system python's sys.path, crashing apt's matplotlib
# (numpy1 ABI) on import. See the matching note in record_session.sh.
export PYTHONNOUSERSITE=1
exec python3 "$WS/tools/debugkit/analyze_session.py" "$@"
