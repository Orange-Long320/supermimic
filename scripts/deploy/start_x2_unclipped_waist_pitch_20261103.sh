#!/bin/bash
# Explicitly unvalidated hardware experiment: waist pitch target not range-clipped.
#
# You start this; nothing here runs by itself.
#
# What matches official_sim_policy.py / official_sim_command.py:
#   * target = action_default + action_scale * action, sent as a position command
#   * stiffness/damping from the same fk-only/profile.json the simulator publishes
#   * no handover ramp, no per-tick step limit, no attitude or angular-speed guard
#   * other joint targets are range-clamped; waist pitch is not
# What is hardware-only and NOT in the simulator:
#   * the preparation path that gets the robot to the reference stance
#   * leg preparation stiffness scaled 3.5x so the stance does not sag
#   * the stale-target fallback (the simulator's command publisher has the same)
set -e
source /opt/ros/humble/setup.bash
source /home/agi/aimdk/install/setup.bash

B=/home/agi/lxh-supermimic
REL=$B/releases/11500_candidate_20261102_r3
LIB=$B/trials/11500_unvalidated_20261102_r1
TRIAL=$B/trials/11500_prep_imu_20261102_r1/run-unclipped-waist-$(date +%Y%m%d-%H%M%S)
PREP=${SM_PREP_SECONDS:-20}
ACTIVE=${SM_ACTIVE_SECONDS:-180}

echo "== existing trial processes"
stale=$(pgrep -f 'publish_pelvis_packet[.]py|x2_hardware_publisher[^ ]*[.]py|x2_11500_worker[^ ]*[.]py|run-policy[^ ]*[.]sh|record_joints[^ ]*[.]py' || true)
if [ -n "$stale" ]; then
  echo "refusing: existing processes: $stale" >&2
  exit 1
fi

echo "This is an unvalidated real-robot control experiment, not a simulator."
echo "WAIST PITCH TARGET RANGE CLAMP IS DISABLED. The target may be far outside the device URDF joint range."
echo "Confirm feet are loaded, fall arrest is in place, and a person has checked and holds the emergency stop."
read -r -p 'Type YES to confirm these current conditions: ' confirmation
if [ "$confirmation" != YES ]; then echo 'Operator conditions not confirmed; aborting.' >&2; exit 1; fi

echo "== trial source: rebuilt from the release, new early_handover"
mkdir "$TRIAL"
mkdir "$TRIAL/src"
cp -a "$REL/source/src/." "$TRIAL/src/"
cp "$LIB/early_handover_unclipped_waist_20261103.py" "$TRIAL/src/supermimic/deploy/early_handover.py"
md5sum "$TRIAL/src/supermimic/deploy/supervision.py" "$REL/source/src/supermimic/deploy/supervision.py"
grep -c outage_grace "$TRIAL/src/supermimic/deploy/supervision.py"
grep -c handover_max_tilt "$TRIAL/src/supermimic/deploy/early_handover.py"

cd "$B"
echo "== entry snapshot"
/usr/bin/python3 capture_x2_entry_snapshot.py \
  --candidate "$REL/startup-candidate.json" --out "$TRIAL/snapshot.json"

echo "== contract (simulator semantics)"
/usr/bin/python3 build_x2_11500_trial_unclipped_waist_20261103.py \
  --release "$REL" --snapshot "$TRIAL/snapshot.json" \
  --prep-target startup --history-source executed --extend-margin 0.8 \
  --max-tilt 3.0 --max-tilt-delta 3.0 --max-angular-speed 100 \
  --max-tick-gap 0.6 --max-state-age 0.05 \
  --blend-seconds 0.001 --max-target-step 10 \
  --prep-leg-kp-scale 3.5 --unbounded --unclipped-waist-pitch-target --operator-confirms-support-and-estop \
  --out "$TRIAL/contract" --max-trial-seconds 30 --prep-seconds "$PREP" > /dev/null
/usr/bin/python3 - "$TRIAL/contract/plan.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
for k in ('blend_seconds','max_tilt','max_tilt_delta','max_angular_speed','handover_max_tilt',
          'unbounded','history_source','prep_target_source','zero_stand_anchor_xy','unclipped_waist_pitch_target'):
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
echo "== waist pitch command target is not range-clipped; downstream behavior is unverified"
echo "== preparation ${PREP}s, then policy up to ${ACTIVE}s"
SM_PREP_SECONDS="$PREP" SM_ACTIVE_SECONDS="$ACTIVE" SM_WORKER_SCRIPT="$LIB/x2_11500_worker_unclipped_waist_20261103.py" \
  SM_PUBLISHER_SCRIPT="$LIB/x2_hardware_publisher_unclipped_waist_20261103.py" \
  SM_RECORDER_SCRIPT="$LIB/record_joints_checked_20261103.py" \
  bash "$B/trials/11500_prep_imu_20261102_r1/run-policy-unclipped-waist-20261103.sh" "$REL" "$TRIAL"
echo "== runner finished; verify the actual robot and HAL state."
