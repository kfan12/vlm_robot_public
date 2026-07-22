#!/bin/bash
# ---------------------------------------------------------------------------
# start_path_demo.sh
# Launch the interactive path-drawing demo (the current default) in a tmux
# session laid out as a 4x2 grid of 8 panes, shown in a NEW terminal window.
# The terminal this script is run from is left untouched: the tmux session is
# built detached, then a fresh WezTerm (or Windows Terminal) window is spawned
# that simply attaches to it.
#
# Each pane runs scripts/_path_demo_pane.sh N, which sources ROS 2 Humble + the
# local install/ workspace, waits its staggered delay, then starts one piece
# of the demo. Order: Gazebo -> spawn -> EKF -> controller -> drawer/RViz.
#
# Panes (columns left->right, top/bottom):
#   1. Gazebo + Bridge + Spawn (robotcar_gazebo gazebo_spawn_robot.launch.py)
#   2. EKF localization        (robotcar_localization localization.launch.py)
#   3. Controller (mpc)        (mpc_tracker_cpp mpc_tracker.launch.py) -- skipped if MANUAL_DRIVE=1
#   4. VLM planner node        (vlm_planner_py vlm_planner.launch.py)
#   5. VLM sign node           (vlm_planner_py vlm_sign_node, Qwen in venv) -- or bag record if VLM_SIGN=0
#   6. RViz                    (robot_bringup rviz.launch.py)
#   7. debugkit recording      (scripts/record_session.sh --preset full)
#   8. Free terminal           (manual commands) -- runs keyboard teleop if MANUAL_DRIVE=1
#
# NOTE: panes run with the SYSTEM python (ROS sourced), NOT the venv — so they
#       deliberately do not activate ~/venvs/vlmpc. capture_frame.py only uses
#       rclpy/cv_bridge/cv2 (all in the system ROS env); it does NOT load the VLM.
#
# Usage:  ./scripts/start_path_demo.sh
#         MANUAL_DRIVE=1 ./scripts/start_path_demo.sh   # controller OFF (pane 3),
#                                                       # keyboard teleop ON (pane 8)
#         RECORD_SESSION=0 ./scripts/start_path_demo.sh # pane 7 idle instead of recording
#         ./scripts/start_path_demo.sh --attach-window [session]
#                                      # internal: runs inside the spawned window
#
# Env vars (MANUAL_DRIVE, VLM_SIGN, VLM_QUERY, DEMO_LABEL, RECORD_SESSION) are
# read by _path_demo_pane.sh inside the tmux server, so they are pushed into the
# session environment below.
# Lifecycle: the demo lives and dies with its window — closing the spawned
# terminal window (or detaching) kills the tmux session and all its
# processes. Manual teardown:  tmux kill-session -t vlm_robot_demo
# ---------------------------------------------------------------------------
set -euo pipefail

if ! command -v tmux >/dev/null 2>&1; then
    echo "❌ tmux is not installed. Install it with:"
    echo "     sudo apt update && sudo apt install -y tmux"
    exit 1
fi

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PANE="$WS/scripts/_path_demo_pane.sh"
if [ ! -f "$WS/install/setup.bash" ]; then
    echo "⚠️  $WS/install/setup.bash not found — run 'colcon build' first."
fi

SESSION="vlm_robot_demo"
SELF="$WS/scripts/start_path_demo.sh"

# ---------------------------------------------------------------------------
# Window mode:  start_path_demo.sh --attach-window [session]
# This is what runs INSIDE the freshly spawned terminal window. It waits for
# the session to exist (tolerates any startup race), attaches, and if the
# attach fails it KEEPS THE WINDOW OPEN with the error visible instead of
# silently disappearing.
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--attach-window" ]; then
    SESSION="${2:-$SESSION}"
    # The demo lives and dies with this window: closing the window (HUP) or
    # detaching (normal exit) kills the whole tmux session, so every pane's
    # process (Gazebo, EKF, MPC, ...) is shut down with it.
    trap 'tmux kill-session -t "$SESSION" 2>/dev/null || true' EXIT HUP TERM
    for _ in $(seq 1 40); do
        tmux has-session -t "$SESSION" 2>/dev/null && break
        sleep 0.5
    done
    if tmux attach -t "$SESSION"; then
        exit 0   # EXIT trap kills the session -> window closes, demo is done
    fi
    trap - EXIT HUP TERM   # attach FAILED: keep the session for debugging
    echo
    echo "❌ Could not attach to tmux session '$SESSION'. Known sessions:"
    tmux ls 2>&1 || true
    echo "   Retry manually with:  tmux attach -t $SESSION"
    exec bash    # hold the window open so the error stays readable
fi

# Spawn a NEW terminal window that runs the --attach-window mode above; the
# current terminal stays as-is. On WSL we go through the Windows side; on
# native Linux, gnome-terminal/xterm. Windows Terminal (wt.exe) is preferred:
# `wezterm.exe start` invoked from inside WSL silently fails to run the
# command (verified 2026-07-06 — window flashes or nothing opens), so WezTerm
# is only a fallback. The trailing sleep makes sure the spawn survives this
# script exiting right after it.
open_attach_window() {
    local win=(bash "$SELF" --attach-window "$SESSION")
    if [ -n "${WSL_DISTRO_NAME:-}" ]; then
        if command -v wt.exe >/dev/null 2>&1; then
            (wt.exe wsl.exe -d "$WSL_DISTRO_NAME" -e "${win[@]}" >/dev/null 2>&1 &)
        elif command -v wezterm.exe >/dev/null 2>&1; then
            (wezterm.exe start -- wsl.exe -d "$WSL_DISTRO_NAME" -e "${win[@]}" >/dev/null 2>&1 &)
        else
            echo "⚠️  Neither wt.exe nor wezterm.exe found on PATH."
            echo "    Attach manually with:  tmux attach -t $SESSION"
            return 0
        fi
        sleep 2   # give the Windows-side spawn time to take off before we exit
        return 0
    fi
    if command -v gnome-terminal >/dev/null 2>&1; then
        gnome-terminal -- "${win[@]}" >/dev/null 2>&1 && return 0
    fi
    if command -v xterm >/dev/null 2>&1; then
        (setsid xterm -e "${win[@]}" >/dev/null 2>&1 &)
        sleep 1
        return 0
    fi
    echo "⚠️  No terminal emulator found to open a new window."
    echo "    Attach manually with:  tmux attach -t $SESSION"
    return 0
}

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ℹ️  tmux session '$SESSION' already exists — opening a window on it."
    echo "    (kill it first with:  tmux kill-session -t $SESSION)"
    open_attach_window
    exit 0
fi

echo "🚀 Building the demo tmux session ($SESSION, 4x2 pane grid)..."

# Build the grid with placeholder panes first (tmux's built-in "tiled" layout
# doesn't give an exact 4-column x 2-row grid for 8 panes — it approximates a
# square). Splitting column-by-column then row-by-row keeps every split a
# clean halving of an existing pane, so there's never a "no space for new
# pane" failure. Column order matches the old Terminator layout.
COL1=$(tmux new-session -d -s "$SESSION" -x 220 -y 50 -P -F '#{pane_id}')
COL3=$(tmux split-window -h -t "$COL1" -P -F '#{pane_id}')
COL2=$(tmux split-window -h -t "$COL1" -P -F '#{pane_id}')
COL4=$(tmux split-window -h -t "$COL3" -P -F '#{pane_id}')
ROW2_1=$(tmux split-window -v -t "$COL1" -P -F '#{pane_id}')
ROW2_2=$(tmux split-window -v -t "$COL2" -P -F '#{pane_id}')
ROW2_3=$(tmux split-window -v -t "$COL3" -P -F '#{pane_id}')
ROW2_4=$(tmux split-window -v -t "$COL4" -P -F '#{pane_id}')

# Forward the demo's toggles into the tmux session so the pane commands (which
# run inside the tmux server, not this shell) still see them.
for var in MANUAL_DRIVE VLM_SIGN VLM_QUERY DEMO_LABEL RECORD_SESSION; do
    if [ -n "${!var:-}" ]; then
        tmux set-environment -t "$SESSION" "$var" "${!var}"
    fi
done

declare -A PANE_ID=(
  [1]="$COL1"    [2]="$ROW2_1"
  [3]="$COL2"    [4]="$ROW2_2"
  [5]="$COL3"    [6]="$ROW2_3"
  [7]="$COL4"    [8]="$ROW2_4"
)
TITLES=(
  "" "1 gazebo" "2 ekf" "3 controller (mpc)" "4 planner"
  "5 vlm_sign" "6 rviz" "7 record_session (full)" "8 free"
)

tmux set-option -t "$SESSION" pane-border-status top
for n in 1 2 3 4 5 6 7 8; do
    tmux respawn-pane -k -t "${PANE_ID[$n]}" "bash $PANE $n"
    tmux select-pane -t "${PANE_ID[$n]}" -T "${TITLES[$n]}"
done

tmux select-pane -t "$COL1"

echo "🪟 Opening the demo in a new terminal window (this terminal is untouched)."
open_attach_window
echo "    Closing that window shuts the whole demo down."
echo "    Manual teardown:  tmux kill-session -t $SESSION"
