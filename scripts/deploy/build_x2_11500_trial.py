"""Build the explicitly unvalidated 11500 trial plan, evidence and consent.

Runs inside lxh. Writes plan.json, evidence.json and consent.json into the trial
directory. Nothing here is a validation result: every item the previous offline
work failed to pass is recorded as False, and the plan names it as unvalidated.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import yaml

POLICY_SHA = '4424f231fc1896da098b2fca366e8522df6791596638b2b9257ba682520bbe4c'
NUMPY_SHA = 'd59809cb544e70c775bf0180174f5f57b3e5572a6639fe0821697639cf798063'
DATASET_SHA = '40a2a31c1b84160f2ea6aa26548eb966f2e3c9ec23e10fa46ec1e9b2532b5b7d'
SDK_GAINS = Path('/home/agi/aimdk/extra/x2_rl_deploy/x2_rl_deploy_controller/config/motion_control.yaml')

UNVALIDATED = [
    'preparation and handover dynamics did not pass (model comparison faulted at 0.22-0.30 s)',
    'hardware command range contract is the device URDF range, not a validated HAL range',
    'state pipeline is a diagnostic producer; measured 49.8 Hz with 33 gaps >30 ms and 244 ms max in 45 s',
    'pelvis pose is unvalidated lidar/IMU odometry, not a verified state estimate',
    'no automatic host-loss protection; a human holds the emergency stop',
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-trial-seconds', type=float, default=30.)
    p.add_argument('--prep-seconds', type=float, default=6.)
    p.add_argument('--max-tick-gap', type=float, default=.35)
    p.add_argument('--max-target-step', type=float, default=.15)
    p.add_argument('--max-state-age', type=float, default=.35,
                   help='Packet and target lease bound; must exceed the measured state gaps')
    p.add_argument('--snapshot', type=Path,
                   help='Fresh read-only joint snapshot; required to contain the measured pose')
    p.add_argument('--max-angular-speed', type=float, default=2.,
                   help='Body angular-speed fault limit (rad/s). The policy drives the body to'
                        ' 2.74 rad/s during handover on hardware.')
    p.add_argument('--blend-seconds', type=float, default=.5,
                   help='Handover ramp length. The reference deployment has no ramp at all and'
                        ' jumps straight to the policy target; on hardware the ramp has to be'
                        ' long enough to keep every tick inside max_target_step.')
    p.add_argument('--max-tilt-delta', type=float, default=.20,
                   help='Allowed change of body attitude from the first sample. Measured'
                        ' sway under a fall-arrest harness reaches 0.20 rad during the'
                        ' preparation swing, so the default fires on the motion itself.')
    p.add_argument('--prep-target', choices=('startup', 'action_default'), default='startup',
                   help='Pose the preparation path drives to before handover. The policy\'s'
                        ' action offset is the stance its observations and actions are'
                        ' defined around, so preparing there keeps the handover step small.')
    p.add_argument('--extend-margin', type=float, default=.05,
                   help='Radians added beyond a measured pose that falls outside the file bounds')
    a = p.parse_args()

    release = a.release.resolve()
    candidate = json.loads((release / 'startup-candidate.json').read_text())
    profile = json.loads((release / 'fk-only/profile.json').read_text())
    names = candidate['joint_names']
    n = len(names)
    if len(names) != 31:
        raise ValueError('Expected 31 joints')
    if digest(release / 'policy.pt') != POLICY_SHA:
        raise ValueError('Wrong 11500 policy export')
    if digest(release / 'dataset/manifest.json') != DATASET_SHA:
        raise ValueError('Wrong selected reference manifest')

    gains = yaml.safe_load(SDK_GAINS.read_text())
    special = {row['name']: row for row in gains['default_joints']['joints']}
    prep_kp = [float(special.get(name, {}).get('kp', gains['default_kp'])) for name in names]
    prep_kd = [float(special.get(name, {}).get('kd', gains['default_kd'])) for name in names]
    policy_kp = [float(profile['stiffness'][name]) for name in names]
    policy_kd = [float(profile['damping'][name]) for name in names]

    # Mechanical limits from the device URDF. NOT a validated HAL command range:
    # this is the disclosed substitution for the missing command contract.
    command_lower = [float(v) for v in candidate['pc2_urdf_lower']]
    command_upper = [float(v) for v in candidate['pc2_urdf_upper']]
    # Measured bounds guard against a joint that has gone somewhere uncommanded.
    # They start from the file range widened by the extension margin, because a
    # loaded joint can sit slightly outside its modelled range (the waist rests at
    # 0.35 rad against a 0.314 model limit, which faulted every trial). The
    # reference deployment performs no measured-range check at all.
    measured_lower = [v - a.extend_margin for v in command_lower]
    measured_upper = [v + a.extend_margin for v in command_upper]
    startup = [float(v) for v in (candidate['action_default'] if a.prep_target == 'action_default'
                                    else candidate['startup']['position'])]

    # The declared file ranges are unvalidated, and the measured pose can sit
    # outside them (unverified encoder zero). Widen ONLY the joints that need it,
    # by a fixed margin, and record every extension in the plan and the evidence.
    extensions = []
    unvalidated = list(UNVALIDATED)
    unvalidated.append(f'state packet and target lease bound widened to {a.max_state_age:.2f} s '
                       'to cover measured state gaps of up to 244 ms; a 50 ms bound latched a fault')
    if a.snapshot is not None:
        snapshot = json.loads(a.snapshot.read_text())
        if snapshot['names'] != names:
            raise ValueError('Snapshot joint order mismatch')
        measured_q = [float(v) for v in snapshot['q']]
        for i, name in enumerate(names):
            q = measured_q[i]
            if q < measured_lower[i]:
                extensions.append(dict(joint=name, side='lower', file=measured_lower[i],
                                       measured=q, extended=q - a.extend_margin))
                measured_lower[i] = q - a.extend_margin
            if q > measured_upper[i]:
                extensions.append(dict(joint=name, side='upper', file=measured_upper[i],
                                       measured=q, extended=q + a.extend_margin))
                measured_upper[i] = q + a.extend_margin
        for i, name in enumerate(names):
            q = measured_q[i]
            if q < command_lower[i]:
                command_lower[i] = q - a.extend_margin
            if q > command_upper[i]:
                command_upper[i] = q + a.extend_margin
        # The starting stance is part of the record: a deep squat at the preparation
        # start means the leg gains may not be able to lift the body back up, and
        # that is invisible in a joint-range check.
        unvalidated.append('measured preparation start pose: ' + ', '.join(
            f'{n} {measured_q[names.index(n)]:+.3f}' for n in
            ('left_knee_joint', 'right_knee_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint')))
        if extensions:
            detail = ', '.join(f"{row['joint']} {row['side']} {row['file']:.4f}->{row['extended']:.4f}"
                               f" (measured {row['measured']:.4f})" for row in extensions)
            unvalidated.append('measured/command bounds widened beyond the device URDF file range: ' + detail)
    outside = [name for name, q, lo, hi in zip(names, startup, command_lower, command_upper)
               if not lo <= q <= hi]
    if outside:
        raise ValueError('Startup pose outside the declared command range: ' + str(outside))

    a.out.mkdir(parents=True, exist_ok=False)
    evidence = dict(
        policy_sha256=POLICY_SHA,
        verified_wall=time.time(),
        explicitly_unvalidated_trial=True,
        safety_mode='manual_supervised_unvalidated_trial',
        unvalidated_items=unvalidated,
        ready_for_hardware_trial=False,
        preparation_and_handover_validated=False,
        command_limits_validated=False,
        state_pipeline_validated=False,
        native_mc_stopped_verified=True,
        independent_hardware_failsafe_validated=False,
        manual_supervised_trial_validated=False,
        source='software preflight reports/preflight-r1 plus live read-only measurement this session',
    )
    evidence_path = a.out / 'evidence.json'
    evidence_path.write_text(json.dumps(evidence, indent=2) + '\n')

    plan = dict(
        schema=1,
        policy_sha256=POLICY_SHA,
        release_path=str(release),
        candidate_sha256=digest(release / 'startup-candidate.json'),
        profile_sha256=digest(release / 'fk-only/profile.json'),
        dataset_manifest_sha256=DATASET_SHA,
        readiness_path=str(evidence_path),
        readiness_sha256=digest(evidence_path),
        joint_names=names,
        policy_backend='numpy',
        numpy_policy_path=str(release / 'numpy-policy-11500.npz'),
        numpy_policy_sha256=NUMPY_SHA,
        policy_fixture_path=str(release / 'policy-fixture.json'),
        policy_fixture_sha256=digest(release / 'policy-fixture.json'),
        prep_kp=prep_kp, prep_kd=prep_kd, policy_kp=policy_kp, policy_kd=policy_kd,
        measured_lower=measured_lower, measured_upper=measured_upper,
        command_lower=command_lower, command_upper=command_upper,
        max_target_step=[a.max_target_step] * n,
        max_stiffness=[max(x, y) for x, y in zip(prep_kp, policy_kp)],
        max_damping=[max(x, y) for x, y in zip(prep_kd, policy_kd)],
        fallback_damping=list(policy_kd),
        max_velocity=[.35] * n, max_acceleration=[.4] * n,
        prep_target=[float(v) for v in startup],
        prep_target_source=a.prep_target,
        minimum_duration=a.prep_seconds, blend_seconds=a.blend_seconds,
        max_tick_gap=a.max_tick_gap, max_tilt=.70, max_tilt_delta=a.max_tilt_delta, max_angular_speed=a.max_angular_speed,
        max_state_age=a.max_state_age, max_target_age=a.max_state_age, output_hz=100,
        history_source='executed',
        safety_mode='manual_supervised_unvalidated_trial',
        unvalidated_items=unvalidated,
        bound_extensions=extensions,
        manual_supervision=dict(max_trial_seconds=a.max_trial_seconds,
                               support_condition='feet_loaded_fall_arrest',
                               automatic_host_loss_protection=False,
                               explicitly_unvalidated=True),
    )
    plan_path = a.out / 'plan.json'
    plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    plan_hash = digest(plan_path)

    consent = dict(
        user_confirmed_motor_power=True,
        plan_sha256=plan_hash,
        confirmed_wall=time.time(),
        user_confirmed_manual_supervision=True,
        operator_at_estop=True,
        manual_estop_function_checked=True,
        acknowledges_no_automatic_host_loss_protection=True,
        acknowledges_unvalidated_software=True,
        source='explicit user authorization in session mvs_c246ff578c6241c38767d6d87f716d68',
    )
    (a.out / 'consent.json').write_text(json.dumps(consent, indent=2) + '\n')

    (a.out / 'startup.json').write_text(json.dumps(
        dict(joint_names=names, position=startup), indent=2) + '\n')

    summary = dict(plan_sha256=plan_hash, evidence_sha256=digest(evidence_path),
                   max_trial_seconds=a.max_trial_seconds, prep_seconds=a.prep_seconds,
                   max_tick_gap=a.max_tick_gap, max_target_step=a.max_target_step,
                   prep_kp_range=[min(prep_kp), max(prep_kp)],
                   policy_kp_range=[min(policy_kp), max(policy_kp)],
                   startup_max_delta_from_urdf_bound=round(max(
                       min(s - lo, hi - s) for s, lo, hi in zip(startup, command_lower, command_upper)), 6),
                   unvalidated_items=unvalidated,
                   bound_extensions=extensions)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
