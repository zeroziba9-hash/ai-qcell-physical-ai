#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/jazzy/setup.bash
python3 -m venv --system-site-packages "$HOME/qcell_ros_venv"
source "$HOME/qcell_ros_venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e /mnt/c/Users/user/ai-qcell --no-deps

python - <<'PY'
import torch
import rclpy
import qcell

print("TORCH", torch.__version__)
print("CUDA", torch.cuda.is_available())
print("DEVICE", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("RCLPY", rclpy.__file__)
print("QCELL", qcell.__file__)
PY
