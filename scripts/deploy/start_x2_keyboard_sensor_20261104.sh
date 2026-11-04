#!/bin/bash
# Explicitly unvalidated continuous keyboard experiment with unclipped targets.
#
# You start this; nothing here runs by itself.
#
# What matches official_sim_policy.py / official_sim_command.py:
#   * target = action_default + action_scale * action, sent as a position command
#   * stiffness/damping from the same fk-only/profile.json the simulator publishes
#   * no handover ramp, no per-tick step limit, no attitude or angular-speed guard
#   * no project-side joint position target range clipping
# What is hardware-only and NOT in the simulator:
#   * the preparation path that gets the robot to the reference stance
#   * leg preparation stiffness scaled 3.5x so the stance does not sag
#   * the stale-target fallback (the simulator's command publisher has the same)
set -e
source /opt/ros/humble/setup.bash
source /home/agi/aimdk/install/setup.bash

B=/home/agi/lxh-supermimic
REL=${SM_RELEASE:-$B/releases/11500_candidate_20261102_r3}
LIB=$B/trials/11500_unvalidated_20261102_r1
TRIAL=$B/trials/11500_prep_imu_20261102_r1/run-keyboard-$(date +%Y%m%d-%H%M%S)
PREP=${SM_PREP_SECONDS:-20}
KEYBOARD_PORT=${SM_KEYBOARD_PORT:-19181}

echo "== existing trial processes"
stale=$(pgrep -f 'publish_pelvis_(packet|imu_sensor)[^ ]*[.]py|x2_hardware_publisher[^ ]*[.]py|x2_11500_worker[^ ]*[.]py|run-policy[^ ]*[.]sh|record_(joints|x2_joint_stream)[^ ]*[.]py' || true)
if [ -n "$stale" ]; then
  echo "refusing: existing processes: $stale" >&2
  exit 1
fi

/usr/bin/python3 - "$KEYBOARD_PORT" <<'PY'
import socket, sys
port=int(sys.argv[1])
if not 1024<=port<=65535:raise SystemExit('Invalid keyboard port')
with socket.socket() as sock:
    sock.bind(('127.0.0.1',port))
PY

echo "This is an unvalidated real-robot control experiment, not a simulator."
echo "ALL JOINT TARGET RANGE CLAMPS ARE DISABLED. Targets may be far outside the device URDF joint ranges."
echo "Confirm feet are loaded, fall arrest is in place, and a person has checked and holds the emergency stop."
echo 'Also confirm the robot remains physically stationary throughout the initial 3-second IMU calibration.'
read -r -p 'Type YES to confirm these current conditions: ' confirmation
if [ "$confirmation" != YES ]; then echo 'Operator conditions not confirmed; aborting.' >&2; exit 1; fi

echo "== trial source: rebuilt from the release, new early_handover"
mkdir "$TRIAL"
mkdir "$TRIAL/src"
cp -a "$REL/source/src/." "$TRIAL/src/"
cp "$LIB/early_handover_keyboard_20261103.py" "$TRIAL/src/supermimic/deploy/early_handover.py"
md5sum "$TRIAL/src/supermimic/deploy/supervision.py" "$REL/source/src/supermimic/deploy/supervision.py"
grep -c outage_grace "$TRIAL/src/supermimic/deploy/supervision.py"
grep -c handover_max_tilt "$TRIAL/src/supermimic/deploy/early_handover.py"

cd "$B"
echo "== entry snapshot"
/usr/bin/python3 capture_x2_entry_snapshot.py \
  --candidate "$REL/startup-candidate.json" --out "$TRIAL/snapshot.json"

echo "== contract (simulator semantics)"
/usr/bin/python3 build_x2_11500_trial_keyboard_20261103.py \
  --release "$REL" --snapshot "$TRIAL/snapshot.json" \
  --prep-target startup --history-source executed --extend-margin 0.8 \
  --max-tilt 3.0 --max-tilt-delta 3.0 --max-angular-speed 100 \
  --max-tick-gap 0.6 --max-state-age 0.05 \
  --blend-seconds 0.001 --max-target-step 10 \
  --prep-leg-kp-scale 3.5 --unbounded --unclipped-all-joint-targets --operator-confirms-support-and-estop \
  --out "$TRIAL/contract" --max-trial-seconds 30 --prep-seconds "$PREP" > /dev/null
/usr/bin/python3 - "$TRIAL/contract/plan.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
for k in ('blend_seconds','max_tilt','max_tilt_delta','max_angular_speed','handover_max_tilt',
          'unbounded','history_source','prep_target_source','zero_stand_anchor_xy','unclipped_all_joint_targets'):
    print('  %-20s %s' % (k, p.get(k)))
print('  %-20s %s' % ('max_target_step', p['max_target_step'][0]))
n = p['joint_names']
i = n.index('left_ankle_pitch_joint')
print('  %-20s %s' % ('prep ankle kp', p['prep_kp'][i]))
print('  %-20s %s' % ('policy ankle kp', round(p['policy_kp'][i], 2)))
print('  %-20s %s' % ('max_trial_seconds', p['manual_supervision']['max_trial_seconds']))
PY

echo
echo "== operator: hold the emergency stop. No attitude guard, no ramp,"
echo "   no step limit: this is the simulator's law. The policy may move fast"
echo "   in the first second because the body starts ~19 deg forward."
echo "== stand anchor x/y feedback enabled (original reference position hold)"
echo "== all joint position targets are not range-clipped; downstream behavior is unverified"
echo "== preparation ${PREP}s, then keyboard policy until interrupted"
SM_PREP_SECONDS="$PREP" SM_KEYBOARD_PORT="$KEYBOARD_PORT" SM_WORKER_SCRIPT="$REL/launcher/x2_11500_worker.py" \
  SM_PUBLISHER_SCRIPT="$LIB/x2_hardware_publisher_unclipped_all_20261103.py" \
  SM_RECORDER_SCRIPT="$LIB/record_x2_joint_stream_20261103.py" \
  bash "$REL/launcher/run_x2_keyboard_sensor_20261104.sh" "$REL" "$TRIAL"
echo "== runner finished; verify the actual robot and HAL state."
