#!/usr/bin/env bash
# Preparation then policy handover, 500 Hz IMU state chain, manual supervision.
#
# Differs from run-trial-imu.sh in three ways that matter, all measured:
#  * the producer uses the verified arguments (shallow receive queues, declared
#    rest, IMU-only ZUPT) instead of the old deep-queue defaults;
#  * the handover request is sent only after the preparation path has finished,
#    so the policy takes over at the declared initial pose, and it is retried
#    until the worker acknowledges instead of firing once into the dark;
#  * the publisher is stopped before the bounded active window expires, so the
#    last command is held by the HAL instead of being replaced by the zero
#    stiffness fallback.
set -o pipefail
release=${1:?release directory}
trial=${2:?new trial directory}
base=/home/agi/lxh-supermimic
script=/home/agi/lxh-supermimic/trials/11500_unvalidated_20261102_r1
prep_seconds=${SM_PREP_SECONDS:-6}
active_seconds=${SM_ACTIVE_SECONDS:-12}
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
stray=$(pgrep -f 'x2_hardware_publisher.py|publish_pelvis_packet.py|x2_11500_worker.py' || true)
if [ -n "$stray" ]; then
  echo "stray trial process still running; refusing to start:" >&2
  pgrep -af 'x2_hardware_publisher.py|publish_pelvis_packet.py|x2_11500_worker.py' >&2
  exit 1
fi

stamp() { printf '[%s] ' "$(date +%H:%M:%S)"; echo "$*"; }

stamp "start IMU state producer"
nohup flock -n -F "$base/online-pelvis.lock" env PYTHONPATH="$PYTHONPATH_RELEASE" /usr/bin/python3 \
  "$base/publish_pelvis_packet.py" --profile "$release/fk-only/profile.json" \
  --model "$release/fk-only/fk-only.xml" --src "$base/producer-src/src" --out "$trial/state" --seconds $((prep_seconds + active_seconds + 40)) \
  --init-bias-seconds 3 --assume-initial-rest --imu-only-zupt --subscribe-depth 3 --max-age 0.035 \
  > "$trial/state.log" 2>&1 &
state_pid=$!
sleep 8
head -2 "$trial/state.log" || true

stamp "start hardware publisher"
nohup env PYTHONPATH="$PYTHONPATH_RELEASE" python3 "$script/x2_hardware_publisher.py" \
  --plan "$plan" --consent "$consent" > "$trial/publisher.log" 2>&1 &
publisher_pid=$!
sleep 3

stamp "record joint states (read only)"
# End the recording before cleanup so it writes its file: killed early it produced
# no trajectory at all, which left the last five runs without joint evidence.
nohup env PYTHONPATH="$PYTHONPATH_RELEASE" python3 "$script/record_joints.py" --seconds $((prep_seconds + active_seconds)) \
  --startup "$trial/contract/startup.json" --out "$trial/joint-recording.json" \
  > "$trial/joints.log" 2>&1 &
monitor_pid=$!

stamp "start policy worker"
nohup env PYTHONPATH="$PYTHONPATH_RELEASE" python3 "$script/x2_11500_worker.py" \
  --plan "$plan" --consent "$consent" > "$trial/worker.log" 2>&1 &
worker_pid=$!

cleanup() {
  # Stop the target stream first so the HAL keeps the final commanded pose.
  kill -TERM "$worker_pid" 2>/dev/null || true
  kill -TERM "$publisher_pid" 2>/dev/null || true
  kill -TERM "$state_pid" 2>/dev/null || true
  wait "$worker_pid" 2>/dev/null || true
  wait "$publisher_pid" 2>/dev/null || true
  wait "$state_pid" 2>/dev/null || true
  # Let the read-only recorder reach its own deadline and write the file.
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT

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
  if env PYTHONPATH="$PYTHONPATH_RELEASE" python3 "$script/x2_11500_worker.py" \
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

stamp "hold ${active_seconds}s under the policy"
sleep "$active_seconds"
stamp "end of bounded window, stopping the target stream"
echo "== worker =="; grep -vE '^ +File|^ +self|^ +await|^ +return|^ +raise|Traceback' "$trial/worker.log" | tail -8
echo "== publisher =="; grep -vE '^ +File|^ +self|^ +await|^ +return|^ +raise|Traceback' "$trial/publisher.log" | tail -6
echo "== joints =="; tail -6 "$trial/joints.log"
