# Installation Guide

Everything needed to run the simulation demo, from a fresh machine to the car
driving itself. The fast path is the install script; the manual steps below it
do exactly the same things, for auditing or picking pieces individually.

**Target platform: Ubuntu 22.04** — either native, or **WSL2 on Windows 11**
(the setup this project was built on). ROS 2 Humble and Gazebo Fortress are
tied to 22.04; newer Ubuntu releases will not work with these versions.

---

## 0. WSL2 preparation (Windows only — skip on native Ubuntu)

```powershell
# In a Windows terminal:
wsl --install -d Ubuntu-22.04
```

Recommended `%UserProfile%\.wslconfig` (adjust to your hardware, then
`wsl --shutdown` to apply):

```ini
[wsl2]
memory=12GB
processors=8
swap=8GB
guiApplications=true
```

Gazebo and RViz windows are displayed through **WSLg** (built into Windows 11) —
no X server setup needed. The demo launcher opens its window via **Windows
Terminal** (`wt.exe`), which Windows 11 ships by default.

---

## 1. Fast path — the install script

```bash
git clone -b demo <REPO_URL> vlm_robot_demo   # any directory name works
cd vlm_robot_demo
./scripts/setup/install_deps.sh
```

The script is idempotent (safe to re-run) and installs: ROS 2 Humble desktop,
the ROS packages the demo uses, Gazebo Fortress + the `ros_gz` bridge, `tmux`,
Eigen, the Python runtime deps, and **OSQP + osqp-eigen built from source into
`/usr/local`** (there is no `libosqp-dev` package in Ubuntu 22.04).

Then go to [§3 Environment & build](#3-environment--build).

---

## 2. Manual installation (what the script does)

### 2.1 ROS 2 Humble

```bash
sudo apt update && sudo apt install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install -y ros-humble-desktop ros-dev-tools
```

Project ROS packages:

```bash
sudo apt install -y \
    ros-humble-xacro ros-humble-robot-state-publisher ros-humble-joint-state-publisher \
    ros-humble-rviz2 ros-humble-tf2-ros ros-humble-tf2-geometry-msgs \
    ros-humble-cv-bridge ros-humble-image-transport ros-humble-ackermann-msgs \
    ros-humble-robot-localization
```

### 2.2 Gazebo Fortress + ROS bridge

```bash
sudo curl -sSL https://packages.osrfoundation.org/gazebo.gpg \
    -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
    | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
sudo apt update
sudo apt install -y ignition-fortress \
    ros-humble-ros-gz ros-humble-ros-gz-bridge ros-humble-ros-gz-sim ros-humble-ros-gz-image
```

### 2.3 Demo runtime deps

```bash
sudo apt install -y tmux libeigen3-dev python3-opencv python3-numpy python3-transforms3d python3-tk
```

### 2.4 OSQP + osqp-eigen (from source)

The MPC links against [OSQP](https://osqp.org) and
[osqp-eigen](https://github.com/robotology/osqp-eigen); neither is packaged
for Ubuntu 22.04, so both are built into `/usr/local`:

```bash
git clone --recursive --depth 1 https://github.com/osqp/osqp
cmake -S osqp -B osqp/build -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build osqp/build -j$(nproc) && sudo cmake --install osqp/build

git clone --depth 1 https://github.com/robotology/osqp-eigen.git
cmake -S osqp-eigen -B osqp-eigen/build -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build osqp-eigen/build -j$(nproc) && sudo cmake --install osqp-eigen/build

sudo ldconfig
```

Verify: `/usr/local/lib/cmake/OsqpEigen/OsqpEigenConfig.cmake` exists.
(`mpc_tracker_cpp/CMakeLists.txt` already points `OsqpEigen_DIR` there —
colcon does not search `/usr/local` on its own.)

---

## 3. Environment & build

Add to `~/.bashrc`:

```bash
export LIBGL_ALWAYS_SOFTWARE=1        # REQUIRED on WSL2 (Gazebo's Ogre crashes on WSL GPU GL); harmless elsewhere
source /opt/ros/humble/setup.bash
```

Build the workspace (from the repo root):

```bash
colcon build --symlink-install
source install/setup.bash             # also worth adding to ~/.bashrc (absolute path)
```

Expected: 8 packages build with no errors (warnings are OK).

---

## 4. Run the demo

```bash
./scripts/start_path_demo.sh
```

- A new terminal window opens showing a 4×2 tmux grid; each pane starts one
  component on a staggered delay (Gazebo first, planner last at ~40 s).
- The Gazebo window shows the car on a marked track; RViz shows the planned
  path, odometry trails, and the FPV camera with detection overlay.
- After ~40 s the car drives off and follows the lane.
- **Shutdown:** close the demo window, or `tmux kill-session -t vlm_robot_demo`.

If anything fails to appear, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md) —
the two most common issues are a missing `LIBGL_ALWAYS_SOFTWARE=1` and an
unbuilt workspace.

---

## 5. Optional: the VLM sign reader (`VLM_SIGN=1`)

The default demo needs none of this. With the VLM extra, a **Qwen2.5-VL-3B**
model (4-bit quantized) reads the roadside signs live and the car reacts:
slows into turns and the winding section, stops at the STOP sign.

**Hardware:** NVIDIA GPU with **~6 GB VRAM** (tested on an RTX 3060 Laptop).
On WSL2, install only the **Windows** NVIDIA driver — never a Linux display
driver inside WSL. The CUDA runtime for PyTorch ships with the pip wheels.

```bash
./scripts/setup/setup_vlm_venv.sh
```

This creates `~/venvs/vlm_robot` (with `--system-site-packages` so ROS's
`rclpy` stays importable) and installs PyTorch (CUDA 12.1 wheels),
transformers 5.x, and bitsandbytes. Details and version constraints live in
`scripts/setup/requirements-vlm.txt`.

Run:

```bash
VLM_SIGN=1 ./scripts/start_path_demo.sh
```

First run downloads Qwen2.5-VL-3B-Instruct (~3 GB) into `~/.cache/huggingface`.

Known constraints (already handled by the code — listed so you don't "fix" them):

- **transformers must be 5.x** — the code uses the 5.x class names
  (`SmolVLMProcessor` etc.); `AutoProcessor`/`AutoModelForVision2Seq` were removed upstream.
- **The venv's NumPy 2 breaks ROS `cv_bridge`** (compiled against NumPy 1.x).
  The VLM nodes are deliberately cv_bridge-free (raw numpy image conversion);
  don't route their images through cv_bridge.
- The sign node is launched with the **venv python via `python3 -m`**, not
  `ros2 run` (whose console-script shebang is the system python → no torch).
  The demo script does this for you.
