#!/usr/bin/env bash
# Continuous keyboard experiment, 500 Hz IMU state chain, manual supervision.
#
# The state source and recorder run continuously until the operator interrupts.
# Loss of keyboard input selects a stand reference; this is not a motor failsafe.
set -o pipefail
release=${1:?release directory}
trial=${2:?new trial directory}
base=/home/agi/lxh-supermimic
script=/home/agi/lxh-supermimic/trials/11500_unvalidated_20261102_r1
worker_script=${SM_WORKER_SCRIPT:-$script/x2_11500_worker.py}
recorder_script=${SM_RECORDER_SCRIPT:-$script/record_joints.py}
publisher_script=${SM_PUBLISHER_SCRIPT:-$script/x2_hardware_publisher.py}
prep_seconds=${SM_PREP_SECONDS:-20}
keyboard_port=${SM_KEYBOARD_PORT:-19181}
producer_src="$base/producer-keyboard-20261103/src"
source /opt/ros/humble/setup.bash
source /home/agi/aimdk/install/setup.bash
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH_RELEASE="$trial/src:$base/python-cpu:$base/vendor-mjlab-math:${PYTHONPATH:-}"
export SM_ROBOT=x2_ultra SM_ROBOT_SPEC="$release/fk-only/profile.json"

if [ ! -d "$trial/src" ]; then
  mkdir -p "$trial/src"
  cp -a "$release/source/src/." "$trial/src/"
  cp "$script/early_handover.py" "$trial/src/supermimic/deploy/early_handover.py"
fi
cd "$trial"
plan=$trial/contract/plan.json
consent=$trial/contract/consent.json
[ -f "$plan" ] || { echo "plan missing: $plan" >&2; exit 1; }

# A publisher left over from an earlier trial keeps the motor command topics
# occupied, so the new publisher exits with "Another motor command publisher
# exists" and the *old* one silently absorbs the new targets under the old
# contract. Refuse to start instead of running a trial nobody is driving.
stray=$(pgrep -f 'x2_hardware_publisher[^ ]*[.]py|publish_pelvis_(packet|imu_sensor)[^ ]*[.]py|x2_11500_worker[^ ]*[.]py' || true)
if [ -n "$stray" ]; then
  echo "stray trial process still running; refusing to start:" >&2
  pgrep -af 'x2_hardware_publisher[^ ]*[.]py|publish_pelvis_(packet|imu_sensor)[^ ]*[.]py|x2_11500_worker[^ ]*[.]py' >&2
  exit 1
fi

stamp() { printf '[%s] ' "$(date +%H:%M:%S)"; echo "$*"; }

# Install cleanup before calibration: failed readiness must leave no trial process.
state_pid= publisher_pid= worker_pid= monitor_pid=
cleanup() {
  for pid in "$worker_pid" "$publisher_pid" "$state_pid" "$monitor_pid"; do
    if [ -n "$pid" ]; then kill -TERM "$pid" 2>/dev/null || true; fi
  done
  for pid in "$worker_pid" "$publisher_pid" "$state_pid" "$monitor_pid"; do
    if [ -n "$pid" ]; then wait "$pid" 2>/dev/null || true; fi
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
stamp "start raw pelvis IMU state producer; keep robot physically stationary"
nohup flock -n -F "$base/online-pelvis.lock" env PYTHONPATH="$PYTHONPATH_RELEASE" /usr/bin/python3 \
  "$release/launcher/publish_pelvis_imu_sensor.py" --profile "$release/fk-only/profile.json" \
  --model "$release/fk-only/fk-only.xml" --urdf "$release/robot-pc2.urdf" \
  --out "$trial/state" --assume-initial-rest > "$trial/state.log" 2>&1 &
state_pid=$!
# No motor publisher is started until fresh, calibrated sensor state is available.
if ! /usr/bin/python3 - "$trial/state/ready.json" "$state_pid" <<'READYPY'
import json, os, sys, time
from pathlib import Path
path, pid = Path(sys.argv[1]), int(sys.argv[2])
deadline = time.monotonic()+25
while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        raise SystemExit('State producer exited; inspect state.log')
    try:
        health = json.loads(path.read_text())
        if health['packets'] >= 10 and 0 <= time.time()-health['wall'] < .25:
            print('Fresh calibrated sensor state available; accuracy remains unvalidated')
            break
    except (OSError, ValueError, KeyError):
        pass
    time.sleep(.05)
else:
    raise SystemExit('State initialization timeout; no motor publisher started')
READYPY
then
  tail -10 "$trial/state.log" >&2
  exit 1
fi

stamp "start hardware publisher"
nohup env PYTHONPATH="$PYTHONPATH_RELEASE" python3 "$publisher_script" \
  --plan "$plan" --consent "$consent" > "$trial/publisher.log" 2>&1 &
publisher_pid=$!
sleep 3

stamp "record joint states (read only)"
# The continuous recorder writes through a bounded queue rather than retaining
# the full trajectory in RAM.
nohup env PYTHONPATH="$producer_src:$PYTHONPATH_RELEASE" python3 "$recorder_script" \
  --startup "$trial/contract/startup.json" --out "$trial/joints" \
  > "$trial/joints.log" 2>&1 &
monitor_pid=$!

stamp "start policy worker"
nohup env PYTHONPATH="$PYTHONPATH_RELEASE" python3 "$worker_script" \
  --plan "$plan" --consent "$consent" --keyboard-port "$keyboard_port" > "$trial/worker.log" 2>&1 &
worker_pid=$!

# Anchor on the first command the worker actually issues instead of a fixed
# delay: worker setup and any calibration retry both shift it. The preparation
# path runs for prep_seconds from that instant, and requesting earlier is
# rejected by the contract.
# The worker requires exactly one subscriber on the target topic, so this probe
# must not subscribe there: it watches the joint command stream instead, which
# only carries traffic once the publisher has accepted a target.
stamp "wait for the first joint command"
if ! python3 - <<'WAITPY'
import time, rclpy
from rclpy.node import Node
from aimdk_msgs.msg import JointCommandArray
rclpy.init()
node = Node("command_wait", enable_rosout=False, start_parameter_services=False)
seen = []
node.create_subscription(JointCommandArray, "/aima/hal/joint/leg/command",
                         lambda m: seen.append(time.monotonic()), 10)
end = time.monotonic() + 60.
while not seen and time.monotonic() < end:
    rclpy.spin_once(node, timeout_sec=.05)
print("first command seen:", bool(seen), flush=True)
node.destroy_node(); rclpy.try_shutdown()
if not seen:
    raise SystemExit(1)
WAITPY
then
  stamp "no joint command observed; aborting before handover"
  exit 1
fi
stamp "wait ${prep_seconds}s for the preparation path"
sleep "$prep_seconds"

handover_acknowledged=false
for attempt in 1 2 3 4; do
  stamp "policy handover request, attempt $attempt"
  if env PYTHONPATH="$PYTHONPATH_RELEASE" python3 "$worker_script" \
    --plan "$plan" --consent "$consent" --request-policy > "$trial/request-$attempt.log" 2>&1; then
    request_ok=true
  else
    request_ok=false
  fi
  tail -1 "$trial/request-$attempt.log" || true
  if [ "$request_ok" = true ] && grep -Eq '^policy request acknowledged: (handover|active)$' "$trial/request-$attempt.log"; then
    stamp "handover acknowledged"
    handover_acknowledged=true
    break
  fi
  sleep 2
done

if [ "$handover_acknowledged" != true ]; then
  stamp "handover was not acknowledged; aborting"
  exit 1
fi

stamp "keyboard control active: loopback port $keyboard_port; runs until interrupted"
stamp "Ctrl-C stops this runner; the robot's resulting HAL state must be verified"
while kill -0 "$state_pid" 2>/dev/null && kill -0 "$publisher_pid" 2>/dev/null && \
      kill -0 "$worker_pid" 2>/dev/null && kill -0 "$monitor_pid" 2>/dev/null; do
  sleep 1
done
stamp "a trial process exited; ending the runner"
echo "== worker =="; grep -vE '^ +File|^ +self|^ +await|^ +return|^ +raise|Traceback' "$trial/worker.log" | tail -8
echo "== publisher =="; grep -vE '^ +File|^ +self|^ +await|^ +return|^ +raise|Traceback' "$trial/publisher.log" | tail -6
echo "== joints =="; tail -6 "$trial/joints.log"
