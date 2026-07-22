#!/bin/bash
# ---------------------------------------------------------------------------
# install_deps.sh — one-shot dependency installer for the VLM-MPC robot demo.
#
# Target: fresh Ubuntu 22.04 (native or WSL2). Installs:
#   * ROS 2 Humble (desktop) + the ROS packages the demo uses
#   * Gazebo Fortress (Ignition) + the ros_gz bridge
#   * tmux, Eigen, OpenCV/transforms3d python deps
#   * OSQP + osqp-eigen built from source into /usr/local
#     (libosqp-dev does not exist in Ubuntu 22.04 apt)
#
# Idempotent: safe to re-run; already-installed pieces are skipped.
# Does NOT install the optional VLM extra (GPU/torch/Qwen) — see
# scripts/setup/setup_vlm_venv.sh and docs/INSTALL.md for that.
#
# Usage:  ./scripts/setup/install_deps.sh
# ---------------------------------------------------------------------------
set -euo pipefail

if [ "$(id -u)" = "0" ]; then
    SUDO=""
else
    SUDO="sudo"
fi

. /etc/os-release
if [ "${VERSION_ID:-}" != "22.04" ]; then
    echo "⚠️  This script targets Ubuntu 22.04 (found: ${PRETTY_NAME:-unknown})."
    echo "    ROS 2 Humble + Gazebo Fortress are tied to 22.04 — continuing anyway in 5 s (Ctrl-C to abort)."
    sleep 5
fi

echo "==> Base tools"
$SUDO apt-get update
$SUDO apt-get install -y build-essential cmake pkg-config git curl wget gnupg lsb-release ca-certificates

# ---------------------------------------------------------------------------
echo "==> ROS 2 Humble apt repository"
if [ ! -f /usr/share/keyrings/ros-archive-keyring.gpg ]; then
    $SUDO curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg
fi
if [ ! -f /etc/apt/sources.list.d/ros2.list ]; then
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" \
        | $SUDO tee /etc/apt/sources.list.d/ros2.list > /dev/null
    $SUDO apt-get update
fi

echo "==> ROS 2 Humble + project ROS packages"
$SUDO apt-get install -y \
    ros-humble-desktop ros-dev-tools \
    ros-humble-xacro \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher \
    ros-humble-rviz2 \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-ackermann-msgs \
    ros-humble-robot-localization

# ---------------------------------------------------------------------------
echo "==> Gazebo Fortress (Ignition) + ros_gz bridge"
if [ ! -f /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg ]; then
    $SUDO curl -sSL https://packages.osrfoundation.org/gazebo.gpg \
        -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
fi
if [ ! -f /etc/apt/sources.list.d/gazebo-stable.list ]; then
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
        | $SUDO tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
    $SUDO apt-get update
fi
$SUDO apt-get install -y \
    ignition-fortress \
    ros-humble-ros-gz \
    ros-humble-ros-gz-bridge \
    ros-humble-ros-gz-sim \
    ros-humble-ros-gz-image

# ---------------------------------------------------------------------------
echo "==> Demo runtime deps (tmux, Eigen, python libs)"
$SUDO apt-get install -y \
    tmux \
    libeigen3-dev \
    python3-opencv \
    python3-numpy \
    python3-transforms3d \
    python3-tk

# ---------------------------------------------------------------------------
# OSQP + osqp-eigen from source (needed by mpc_tracker_cpp).
# mpc_tracker_cpp/CMakeLists.txt already points OsqpEigen_DIR at /usr/local.
# ---------------------------------------------------------------------------
if [ -f /usr/local/lib/cmake/OsqpEigen/OsqpEigenConfig.cmake ]; then
    echo "==> OSQP + osqp-eigen already installed in /usr/local — skipping source build"
else
    BUILD_DIR="$(mktemp -d)"
    trap 'rm -rf "$BUILD_DIR"' EXIT

    echo "==> Building OSQP from source into /usr/local"
    git clone --recursive --depth 1 https://github.com/osqp/osqp "$BUILD_DIR/osqp"
    cmake -S "$BUILD_DIR/osqp" -B "$BUILD_DIR/osqp/build" -DCMAKE_INSTALL_PREFIX=/usr/local
    cmake --build "$BUILD_DIR/osqp/build" -j"$(nproc)"
    $SUDO cmake --install "$BUILD_DIR/osqp/build"

    echo "==> Building osqp-eigen from source into /usr/local"
    git clone --depth 1 https://github.com/robotology/osqp-eigen.git "$BUILD_DIR/osqp-eigen"
    cmake -S "$BUILD_DIR/osqp-eigen" -B "$BUILD_DIR/osqp-eigen/build" -DCMAKE_INSTALL_PREFIX=/usr/local
    cmake --build "$BUILD_DIR/osqp-eigen/build" -j"$(nproc)"
    $SUDO cmake --install "$BUILD_DIR/osqp-eigen/build"

    $SUDO ldconfig
fi

# ---------------------------------------------------------------------------
echo
echo "✅ All demo dependencies installed."
echo
echo "Next steps (see docs/INSTALL.md for details):"
echo "  1. Add to your ~/.bashrc (required on WSL2, harmless elsewhere):"
echo "       export LIBGL_ALWAYS_SOFTWARE=1"
echo "       source /opt/ros/humble/setup.bash"
echo "  2. Build the workspace:"
echo "       cd <this repo> && colcon build --symlink-install"
echo "  3. Run the demo:"
echo "       ./scripts/start_path_demo.sh"
