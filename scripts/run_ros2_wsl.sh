#!/usr/bin/env bash
set -eo pipefail

mode=${1:-mock}
project_root=${2:-/mnt/c/Users/user/ai-qcell}
if [[ "$mode" != "mock" && "$mode" != "deep" ]]; then
  echo "mode must be mock or deep" >&2
  exit 2
fi

source /opt/ros/jazzy/setup.bash
if [[ "$mode" == "deep" ]]; then
  source "$HOME/qcell_ros_venv/bin/activate"
fi

workspace=$(mktemp -d /tmp/qcell_ros2_run_XXXXXX)
cp -r "$project_root/ros2_ws/src" "$workspace/src"
cd "$workspace"

if [[ "$mode" == "deep" ]]; then
  python -m colcon build --symlink-install --event-handlers console_direct+
else
  colcon build --symlink-install --event-handlers console_direct+
fi
source install/setup.bash
export PYTHONPATH="$project_root:${PYTHONPATH:-}"

if [[ "$mode" == "deep" ]]; then
  ros2 launch ai_qcell_ros qcell_pipeline.launch.py \
    model_path:="$project_root/models/deep_patchcore_bottle.pt" \
    dataset_root:="$project_root/data/mvtec-ad/bottle"
else
  ros2 launch ai_qcell_ros qcell_mock_pipeline.launch.py \
    dataset_root:="$project_root/data/mvtec-ad/bottle"
fi
