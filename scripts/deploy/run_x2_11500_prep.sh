#!/usr/bin/env bash
# Preparation-only run: move to the 11500 startup pose and hold it.
# No policy handover request. Explicitly unvalidated manual supervision.
set -eo pipefail

release=${1:?release directory}
trial=${2:?new trial directory}
lio=${3:?private online localization directory}

base=/home/agi/lxh-supermimic
script=$(cd "$(dirname "$0")" && pwd)
source /opt/ros/humble/setup.bash
source /home/agi/aimdk/install/setup.bash
source "$lio/install/share/livox_ros_driver2/local_setup.bash"
source "$lio/install/share/fast_lio/local_setup.bash"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH_RELEASE="$trial/src:$base/python-cpu:$base/vendor-mjlab-math:${PYTHONPATH:-}"
export PYTHONPATH_LIO="$lio/code/src:$base/python-cpu:$base/vendor-mjlab-math:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$lio/install/lib:${LD_LIBRARY_PATH:-}"
export SM_ROBOT=x2_ultra SM_ROBOT_SPEC="$release/fk-only/profile.json"

mkdir "$trial"
mkdir "$trial/src"
cp -a "$release/source/src/." "$trial/src/"
cp "${PATCHED_CONTROLLER:-$trial/early_handover.py}" "$trial/src/supermimic/deploy/early_handover.py"
cd "$trial"

echo "== build contract =="
env PYTHONPATH="$PYTHONPATH_RELEASE" python3 "$script/build_x2_11500_trial.py" --release "$release" --out "$trial/contract" \
  --max-trial-seconds 30 --prep-seconds 10 > "$trial/contract.log" 2>&1
python3 - "$trial" <<'PY'
import json,sys,pathlib
t=pathlib.Path(sys.argv[1]); d=json.loads((t/'contract.log').read_text())
print(json.dumps({k:d[k] for k in ('plan_sha256','prep_seconds','max_tick_gap','max_target_step')},indent=2))
PY
plan=$trial/contract/plan.json
consent=$trial/contract/consent.json

echo "== start diagnostic state producer =="
nohup flock -n -F "$base/online-pelvis.lock" env PYTHONPATH="$PYTHONPATH_LIO" /usr/bin/python3 "$lio/code/scripts/online_x2_lio.py" \
  --binary "$lio/install/lib/fast_lio/fastlio_mapping" \
  --urdf "$lio/assets/robot-pc2.urdf" --profile "$lio/assets/fk-only/profile.json" \
  --extrinsics "$lio/extrinsics.json" --out "$trial/state" \
  --seconds 70 --state-clock imu-grid --freeze-startup-gc \
  --native-stamp-source lidar > "$trial/state.log" 2>&1 &
state_pid=$!
sleep 22

echo "== start hardware publisher =="
nohup env PYTHONPATH="$PYTHONPATH_RELEASE" python3 "$script/x2_hardware_publisher.py" --plan "$plan" --consent "$consent" \
  > "$trial/publisher.log" 2>&1 &
publisher_pid=$!
sleep 3

echo "== record joint states (read only) =="
nohup env PYTHONPATH="$PYTHONPATH_RELEASE" python3 "$script/record_x2_joint_states.py" --seconds 26 --startup "$trial/contract/startup.json" \
  --out "$trial/joint-recording.json" > "$trial/joints.log" 2>&1 &
monitor_pid=$!

echo "== start policy worker (preparation only) =="
nohup env PYTHONPATH="$PYTHONPATH_RELEASE" python3 "$script/x2_11500_worker.py" --plan "$plan" --consent "$consent" \
  > "$trial/worker.log" 2>&1 &
worker_pid=$!

python3 - <<'PY'
import time
import rclpy
from rclpy.node import Node
rclpy.init(); node = Node('trial_wait', enable_rosout=False, start_parameter_services=False)
end = time.monotonic() + 60.
while time.monotonic() < end and node.count_subscribers('/supermimic/x2/control_target') < 1:
    rclpy.spin_once(node, timeout_sec=.2)
print('publisher subscribed:', node.count_subscribers('/supermimic/x2/control_target'), flush=True)
node.destroy_node(); rclpy.try_shutdown()
PY

echo "== preparation and hold =="
sleep 26
wait "$monitor_pid" || true

echo "== stop publishing: hold last command in the HAL =="
kill -TERM "$publisher_pid" 2>/dev/null || true
wait "$publisher_pid" 2>/dev/null || true
kill -TERM "$worker_pid" 2>/dev/null || true
wait "$worker_pid" 2>/dev/null || true
kill -TERM "$state_pid" 2>/dev/null || true
wait "$state_pid" 2>/dev/null || true

echo "== joint recording =="
cat "$trial/joints.log"
echo "== worker log =="
cat "$trial/worker.log"
echo "== publisher log =="
cat "$trial/publisher.log"
