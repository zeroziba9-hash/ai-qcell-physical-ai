#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/jazzy/setup.bash
set -u
project_root=${1:-/mnt/c/Users/user/ai-qcell}
workspace=$(mktemp -d /tmp/qcell_ros2_XXXXXX)
cp -r "$project_root/ros2_ws/src" "$workspace/src"

cd "$workspace"
colcon build --symlink-install --event-handlers console_direct+
set +u
source install/setup.bash
set -u

echo "=== INTERFACES ==="
ros2 interface show ai_qcell_interfaces/msg/InspectionResult
ros2 interface show ai_qcell_interfaces/action/RejectProduct

echo "=== LIVE PIPELINE ==="
set +e
timeout 12s ros2 launch ai_qcell_ros qcell_mock_pipeline.launch.py \
  dataset_root:="$project_root/data/mvtec-ad/bottle" \
  publish_period:=0.8
status=$?
set -e
if [[ $status -ne 0 && $status -ne 124 ]]; then
  exit "$status"
fi

echo "ROS2_RUNTIME_OK"
echo "BUILD_WORKSPACE=$workspace"
