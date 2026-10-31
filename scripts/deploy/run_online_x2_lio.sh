#!/usr/bin/env bash
# Online diagnostic state only. No motor command path.
set -eo pipefail
run=${1:?private online localization directory}
session=${2:?new diagnostic session directory}
shift 2
base=/home/agi/lxh-supermimic
source /opt/ros/humble/setup.bash
source /home/agi/aimdk/install/setup.bash
source "$run/install/share/livox_ros_driver2/local_setup.bash"
source "$run/install/share/fast_lio/local_setup.bash"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$run/code/src:$base/python-cpu:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$run/install/lib:${LD_LIBRARY_PATH:-}"
exec flock -n -F "$base/online-pelvis.lock" /usr/bin/python3 "$run/code/scripts/online_x2_lio.py" \
  --binary "$run/install/lib/fast_lio/fastlio_mapping" \
  --urdf "$run/assets/robot-pc2.urdf" --profile "$run/assets/fk-only/profile.json" \
  --extrinsics "$run/extrinsics.json" --out "$session" "$@"
