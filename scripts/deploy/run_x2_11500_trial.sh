#!/usr/bin/env bash
# Explicitly unvalidated manual-supervised 11500 trial. Run inside lxh only.
# Prepares the startup pose first, then requests the policy handover.
# No automatic host-loss protection: the operator holds the emergency stop.
set -eo pipefail

release=${1:?release directory}
trial=${2:?new trial directory}
lio=${3:?private online localization directory}
seconds=${4:-30}

base=/home/agi/lxh-supermimic
script=$(cd "$(dirname "$0")" && pwd)
source /opt/ros/humble/setup.bash
source /home/agi/aimdk/install/setup.bash
source "$lio/install/share/livox_ros_driver2/local_setup.bash"
source "$lio/install/share/fast_lio/local_setup.bash"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH_RELEASE="$release/source/src:$base/python-cpu:$base/vendor-mjlab-math:${PYTHONPATH:-}"
export PYTHONPATH_LIO="$lio/code/src:$base/python-cpu:$base/vendor-mjlab-math:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$lio/install/lib:${LD_LIBRARY_PATH:-}"
export SM_ROBOT=x2_ultra SM_ROBOT_SPEC="$release/fk-only/profile.json"
# The native sensor stack is pinned to cores 1-3 (perception on 4). Without a
# pin our processes contend there and lose 50 Hz grid ticks; cores 5-7 measured
# a 20 ms worst gap and zero stale packets.
TRIAL_CPUS=${TRIAL_CPUS:-5,6,7}
run=()

mkdir "$trial"
mkdir "$trial/src"
cp -a "$release/source/src/." "$trial/src/"
cp "${PATCHED_CONTROLLER:-$script/early_handover.py}" "$trial/src/supermimic/deploy/early_handover.py"
cd "$trial"

echo "== capture read-only measured pose =="
env PYTHONPATH="$PYTHONPATH_RELEASE" python3 "$script/capture_x2_entry_snapshot.py" \
  --candidate "$release/startup-candidate.json" --out "$trial/snapshot.json"

echo "== build contract =="
env PYTHONPATH="$PYTHONPATH_RELEASE" python3 "$script/build_x2_11500_trial.py" --release "$release" --out "$trial/contract" \
  --max-trial-seconds "$seconds" --prep-seconds 10 --snapshot "$trial/snapshot.json" > "$trial/contract.log" 2>&1
cat "$trial/contract.log"

plan=$trial/contract/plan.json
consent=$trial/contract/consent.json

echo "== start diagnostic state producer =="
nohup flock -n -F "$base/online-pelvis.lock" taskset -c "$TRIAL_CPUS" env PYTHONPATH="$PYTHONPATH_LIO" /usr/bin/python3 "$lio/code/scripts/online_x2_lio.py" \
  --binary "$lio/install/lib/fast_lio/fastlio_mapping" \
  --urdf "$lio/assets/robot-pc2.urdf" --profile "$lio/assets/fk-only/profile.json" \
  --extrinsics "$lio/extrinsics.json" --out "$trial/state" \
  --seconds $((seconds + 40)) --state-clock imu-grid --freeze-startup-gc \
  --native-stamp-source lidar > "$trial/state.log" 2>&1 &
state_pid=$!
sleep 22

echo "== start hardware publisher =="
nohup taskset -c "$TRIAL_CPUS" env PYTHONPATH="$PYTHONPATH_RELEASE" python3 "$script/x2_hardware_publisher.py" --plan "$plan" --consent "$consent" \
  > "$trial/publisher.log" 2>&1 &
publisher_pid=$!
sleep 3

echo "== trace targets =="
nohup env PYTHONPATH="$PYTHONPATH_RELEASE" python3 "$script/trace_x2_control_target.py" --seconds $((seconds + 10)) --out "$trial/target-trace.json" \
  > "$trial/monitor.log" 2>&1 &
monitor_pid=$!

echo "== start policy worker =="
nohup taskset -c "$TRIAL_CPUS" env PYTHONPATH="$PYTHONPATH_RELEASE" python3 "$script/x2_11500_worker.py" --plan "$plan" --consent "$consent" \
  > "$trial/worker.log" 2>&1 &
worker_pid=$!

cleanup() {
  kill -TERM "$worker_pid" "$publisher_pid" "$state_pid" "$monitor_pid" 2>/dev/null || true
  wait "$worker_pid" 2>/dev/null || true
  wait "$publisher_pid" 2>/dev/null || true
  wait "$state_pid" 2>/dev/null || true
}
trap cleanup EXIT

# The worker only publishes targets once it holds a fresh state packet.
python3 - <<'PY'
import time
import rclpy
from rclpy.node import Node
rclpy.init(); node = Node('trial_wait', enable_rosout=False, start_parameter_services=False)
topic = '/supermimic/x2/control_target'
end = time.monotonic() + 60.
while time.monotonic() < end and node.count_subscribers(topic) < 1:
    rclpy.spin_once(node, timeout_sec=.2)
print('publisher subscribed:', node.count_subscribers(topic), flush=True)
node.destroy_node(); rclpy.try_shutdown()
PY

echo "== preparation phase =="
sleep 7

echo "== request policy handover =="
python3 "$script/x2_11500_worker.py" --plan "$plan" --consent "$consent" --request-policy \
  > "$trial/request.log" 2>&1 || true
cat "$trial/request.log"

wait "$monitor_pid" || true
cat "$trial/monitor.log"

echo "== worker log =="
cat "$trial/worker.log"
echo "== publisher log =="
cat "$trial/publisher.log"
