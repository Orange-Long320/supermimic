#!/usr/bin/env bash
# Preparation-only run: move to the declared initial pose and hold it.
# State chain: 500 Hz IMU producer (no lidar/LIO). No policy handover request.
set -eo pipefail
release=${1:?release directory}
trial=${2:?new trial directory}
base=/home/agi/lxh-supermimic
script=/home/agi/lxh-supermimic/trials/11500_unvalidated_20261102_r1
max_seconds=${SM_MAX_SECONDS:-30}
prep_seconds=${SM_PREP_SECONDS:-6}
source /opt/ros/humble/setup.bash
source /home/agi/aimdk/install/setup.bash
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH_RELEASE="$trial/src:$base/python-cpu:$base/vendor-mjlab-math:${PYTHONPATH:-}"
export SM_ROBOT=x2_ultra SM_ROBOT_SPEC="$release/fk-only/profile.json"
mkdir -p "$trial"
[ -d "$trial/src" ] || { mkdir -p "$trial/src"; cp -a "$release/source/src/." "$trial/src/"; cp "$script/early_handover.py" "$trial/src/supermimic/deploy/early_handover.py"; }
cd "$trial"
plan=$trial/contract/plan.json
consent=$trial/contract/consent.json
[ -f "$plan" ] || { echo "plan missing" >&2; exit 1; }
/usr/bin/python3 - "$plan" "$max_seconds" "$prep_seconds" <<"PY"
import hashlib, sys
print("plan sha256", hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest()[:16],
      "max", sys.argv[2], "prep", sys.argv[3])
PY
echo "== start IMU state producer =="
nohup flock -n -F "$base/online-pelvis.lock" env PYTHONPATH="$PYTHONPATH_RELEASE" /usr/bin/python3 \
  "$base/publish_pelvis_packet.py" --profile "$release/fk-only/profile.json" \
  --model "$release/fk-only/fk-only.xml" --src "$base/producer-src/src" --out "$trial/state" --seconds 70 \
  --init-bias-seconds 3 --assume-initial-rest --imu-only-zupt \
  --subscribe-depth 3 --max-age 0.035 \
  > "$trial/state.log" 2>&1 &
state_pid=$!
sleep 8
tail -2 "$trial/state.log" || true
echo "== start hardware publisher =="
nohup env PYTHONPATH="$PYTHONPATH_RELEASE" python3 "$script/x2_hardware_publisher.py" \
  --plan "$plan" --consent "$consent" > "$trial/publisher.log" 2>&1 &
publisher_pid=$!
sleep 3
echo "== record joint states (read only) =="
nohup env PYTHONPATH="$PYTHONPATH_RELEASE" python3 "$script/record_joints.py" --seconds 26 \
  --startup "$trial/contract/startup.json" --out "$trial/joint-recording.json" > "$trial/joints.log" 2>&1 &
monitor_pid=$!
echo "== start worker (preparation only) =="
nohup env PYTHONPATH="$PYTHONPATH_RELEASE" python3 "$script/x2_11500_worker.py" \
  --plan "$plan" --consent "$consent" > "$trial/worker.log" 2>&1 &
worker_pid=$!
sleep "$prep_seconds"
echo "== preparation window done, holding =="
sleep 16
wait "$monitor_pid" || true
echo "== stop publishing: hold last command in the HAL =="
kill -TERM "$publisher_pid" 2>/dev/null || true
wait "$publisher_pid" 2>/dev/null || true
kill -TERM "$worker_pid" 2>/dev/null || true
wait "$worker_pid" 2>/dev/null || true
kill -TERM "$state_pid" 2>/dev/null || true
wait "$state_pid" 2>/dev/null || true
echo "== worker =="; tail -20 "$trial/worker.log" || true
echo "== publisher =="; tail -8 "$trial/publisher.log" || true
echo "== joints =="; tail -20 "$trial/joints.log" || true
