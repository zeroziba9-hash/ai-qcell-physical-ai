#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/jazzy/setup.bash
source "$HOME/qcell_ros_venv/bin/activate"
project_root=${1:-/mnt/c/Users/user/ai-qcell}
workspace=$(mktemp -d /tmp/qcell_ros2_deep_XXXXXX)
cp -r "$project_root/ros2_ws/src" "$workspace/src"

cd "$workspace"
python -m colcon build --symlink-install --event-handlers console_direct+
source install/setup.bash
export PYTHONPATH="$project_root:${PYTHONPATH:-}"

set +e
log_file="$workspace/deep_pipeline.log"
timeout 18s ros2 launch ai_qcell_ros qcell_pipeline.launch.py \
  model_path:="$project_root/models/deep_patchcore_bottle.pt" \
  dataset_root:="$project_root/data/mvtec-ad/bottle" 2>&1 | tee "$log_file"
status=$?
set -e
if [[ $status -ne 0 && $status -ne 124 ]]; then
  exit "$status"

fi
grep -q "Deep PatchCore loaded" "$log_file"
grep -Eq "score=[0-9]" "$log_file"
grep -Eq "PASS|REJECT" "$log_file"
if grep -q "process has died" "$log_file"; then
  echo "A ROS2 node died during the deep runtime test" >&2
  exit 1
fi

echo "ROS2_DEEP_RUNTIME_OK"
echo "BUILD_WORKSPACE=$workspace"
