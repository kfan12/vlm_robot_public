#!/bin/bash
# ---------------------------------------------------------------------------
# setup_vlm_venv.sh — OPTIONAL: python venv for the VLM sign-reading extra.
#
# The default demo does NOT need this (the lane planner is pure OpenCV on the
# system python). Run this only if you want VLM_SIGN=1 — the Qwen2.5-VL-3B
# node that reads the traffic signs live. Requirements:
#   * NVIDIA GPU with ~6 GB VRAM (tested: RTX 3060 Laptop)
#   * CUDA-capable driver on the host (on WSL2: the *Windows* NVIDIA driver;
#     do NOT install a Linux display driver inside WSL)
#
# Creates ~/venvs/vlm_robot with --system-site-packages (so rclpy is visible)
# and installs torch (CUDA 12.1 wheels) + transformers 5.x + bitsandbytes.
#
# Known caveats baked in below (see docs/INSTALL.md for the full story):
#   * Ubuntu 22.04's system scipy/pillow/jinja2/num2words are too old for
#     transformers 5.x + NumPy 2 — venv-local upgrades are forced.
#   * ROS cv_bridge is NOT usable from this venv (compiled for NumPy 1.x);
#     the VLM nodes are deliberately cv_bridge-free, so this is fine.
#
# Usage:  ./scripts/setup/setup_vlm_venv.sh
# ---------------------------------------------------------------------------
set -euo pipefail

VENV="$HOME/venvs/vlm_robot"
REQ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/requirements-vlm.txt"

sudo apt-get install -y python3-venv python3-pip

if [ ! -f "$VENV/bin/activate" ]; then
    echo "==> Creating venv $VENV (--system-site-packages)"
    mkdir -p "$(dirname "$VENV")"
    python3 -m venv --system-site-packages "$VENV"
else
    echo "==> venv $VENV already exists — updating packages in place"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

pip install --upgrade pip

echo "==> Installing PyTorch (CUDA 12.1 wheels)"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "==> Installing VLM stack"
pip install -r "$REQ"

echo
echo "✅ VLM venv ready."
python - <<'PY'
import torch
print(f"   torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   device: {torch.cuda.get_device_name(0)}")
else:
    print("   ⚠️  CUDA not available — the sign node will be far too slow on CPU.")
PY
echo
echo "Run the demo with the VLM sign reader:"
echo "  VLM_SIGN=1 ./scripts/start_path_demo.sh"
echo "(first run downloads Qwen2.5-VL-3B-Instruct, ~3 GB, into ~/.cache/huggingface)"
