"""Bounded live 11500 policy probe. Subscriptions and local files only.

Raw measured joints are preserved even when file bounds disagree. Such frames
are explicitly ineligible for control. Previous actions are counterfactual
shadow outputs, never executed actions. This program has no actuator path.
"""
import argparse
from collections import Counter, deque
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    return dict(zip(('min', 'p50', 'p95', 'p99', 'max'),
                    np.percentile(values, [0, 50, 95, 99, 100]).tolist())) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('release', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=20.)
    parser.add_argument('--diagnostic-only', action='store_true', required=True)
    args = parser.parse_args()
    if not 2 <= args.seconds <= 60:
        raise ValueError('Expected bounded 2–60 second probe')
    r = args.release.resolve()
    if digest(r/'policy.pt') != '4424f231fc1896da098b2fca366e8522df6791596638b2b9257ba682520bbe4c':
        raise ValueError('Expected selected 11500 model')
    if digest(r/'dataset/manifest.json') != '40a2a31c1b84160f2ea6aa26548eb966f2e3c9ec23e10fa46ec1e9b2532b5b7d':
        raise ValueError('Expected selected reference library')
    candidate = json.loads((r/'startup-candidate.json').read_text())
    names = candidate['joint_names']; n = len(names)
    if n != 31 or len(set(names)) != n:
        raise ValueError('Invalid joint contract')
    default = np.asarray(candidate['action_default'])
    scale = np.asarray(candidate['action_scale'])
    lower = np.asarray(candidate['measured_lower_candidate'])
    upper = np.asarray(candidate['measured_upper_candidate'])
    os.environ['SM_ROBOT'] = 'x2_ultra'
    os.environ['SM_ROBOT_SPEC'] = str(r/'fk-only/profile.json')
    import torch
    from supermimic.deploy.skill_goal import SkillGoal
    from supermimic.deploy.torch_policy import TorchScriptPolicy
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from aimdk_msgs.msg import JointStateArray
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    torch.set_num_threads(1)
    policy = TorchScriptPolicy(r/'policy.pt')
    tracker = SkillGoal(r/'dataset', r/'fk-only/profile.json', [0., 0., .65], [1., 0., 0., 0.])
    args.out.mkdir(parents=True, exist_ok=False)
    rclpy.init()
    node = Node('supermimic_online_policy_probe', enable_rosout=False, start_parameter_services=False)
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
    groups, counts, pose, status = {}, Counter(), {}, {}
    def stamp(msg): return msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
    def joints(msg, group):
        groups[group] = (time.monotonic(), stamp(msg),
                         [(j.name, j.position, j.velocity, j.error_code) for j in msg.joints])
        counts[group] += 1
    def odometry(msg):
        p, q, w = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist.angular
        pose.update(received=time.monotonic(), stamp=stamp(msg), position=[p.x, p.y, p.z],
                    quat=[q.w, q.x, q.y, q.z], omega=[w.x, w.y, w.z],
                    parent=msg.header.frame_id, child=msg.child_frame_id)
        counts['pose'] += 1
    def observe(msg):
        status.update(received=time.monotonic(), data=json.loads(msg.data)); counts['status'] += 1
    subs = [node.create_subscription(JointStateArray, '/aima/hal/joint/'+g+'/state',
            lambda m, g=g: joints(m, g), qos) for g in ('leg', 'waist', 'head', 'arm')]
    subs += [node.create_subscription(Odometry, '/supermimic/pelvis/odometry', odometry, qos),
             node.create_subscription(String, '/supermimic/x2_online_lio/status', observe, qos)]
    history = deque(maxlen=10); previous = np.zeros(n); last_frame = None
    session = None; initialized = False; rejected = Counter(); outside = Counter()
    latencies, total, header_ages, intervals, max_pose_errors, target_deltas = [], [], [], [], [], []
    valid = 0; frames = 0; raw_target_outside = 0; fatal = None
    started = next_tick = time.monotonic()
    try:
        with (args.out/'frames.jsonl').open('x') as log:
            while time.monotonic()-started < args.seconds:
                now = time.monotonic()
                if now < next_tick:
                    rclpy.spin_once(node, timeout_sec=min(.002, next_tick-now)); continue
                next_tick = now+.02
                try:
                    if not pose or set(groups) != {'leg', 'waist', 'head', 'arm'} or not status:
                        raise ValueError('waiting for all live inputs')
                    if now-status['received'] > 2.:
                        raise ValueError('localization status stale')
                    current_session = status['data']['session_id']
                    if session is not None and session != current_session:
                        raise RuntimeError('Localization session changed; restart probe')
                    session = current_session
                    if node.count_publishers('/supermimic/pelvis/odometry') != 1:
                        raise RuntimeError('Expected exactly one pelvis publisher')
                    if (pose['parent'], pose['child']) != ('supermimic_world', 'pelvis'):
                        raise RuntimeError('Unexpected pelvis frames')
                    wall = time.time()
                    ages = [wall-pose['stamp']] + [wall-row[1] for row in groups.values()]
                    receives = [pose['received']] + [row[0] for row in groups.values()]
                    stamps = [pose['stamp']] + [row[1] for row in groups.values()]
                    if min(ages) < 0 or max(ages) > .05 or any(now-t > .05 for t in receives):
                        raise ValueError('input stale or future')
                    if max(stamps)-min(stamps) > .03:
                        raise ValueError('acquisition skew over 30 ms')
                    measured = [j for row in groups.values() for j in row[2]]
                    by_name = {j[0]: j for j in measured}
                    if len(measured) != n or set(by_name) != set(names) or any(j[3] for j in measured):
                        raise ValueError('joint set duplicate missing or faulted')
                    q = np.asarray([by_name[name][1] for name in names])
                    qd = np.asarray([by_name[name][2] for name in names])
                    quat = np.asarray(pose['quat']); omega = np.asarray(pose['omega'])
                    if not np.isfinite(np.r_[q, qd, quat, omega, pose['position']]).all() or abs(np.linalg.norm(quat)-1) > .01:
                        raise ValueError('nonfinite state or invalid quaternion')
                    if last_frame is not None and now-last_frame > .06:
                        history.clear(); previous = np.zeros(n); rejected['history reset after gap'] += 1
                    if last_frame is not None: intervals.append(now-last_frame)
                    last_frame = now
                    w, x, y, z = quat
                    gravity = np.array([2*(w*y-x*z), -2*(w*x+y*z), -1+2*(x*x+y*y)])
                    history.append((q-default, qd, omega, gravity, previous.copy()))
                    bad = (q < lower) | (q > upper)
                    outside.update(name for name, b in zip(names, bad) if b)
                    max_pose_errors.append(float(np.max(np.abs(q-candidate['startup']['position']))))
                    valid += 1
                    if not initialized:
                        tracker.position = np.asarray(pose['position']).copy()
                        tracker.yaw = tracker.rotation(quat).as_euler('xyz')[2]; initialized = True
                    if len(history) < 10: continue
                    obs = np.concatenate([frame[term] for term in range(5) for frame in history])
                    # The selected ac6c755 runtime has the original four-argument
                    # baseline goal interface; retain that exact scheduler.
                    goal = tracker.goal(pose['position'], quat, q, 'stand')
                    if obs.shape != (990,) or goal.shape != (729,) or not np.isfinite(np.r_[obs, goal]).all():
                        raise RuntimeError('Unexpected or invalid model input')
                    begin = time.perf_counter(); action = np.asarray(policy.infer(obs, goal))
                    latencies.append((time.perf_counter()-begin)*1000)
                    if action.shape != (31,) or not np.isfinite(action).all():
                        raise RuntimeError('Invalid model output')
                    target = default+scale*action
                    raw_target_outside += int(np.any((target < lower) | (target > upper)))
                    target_deltas.append(float(np.max(np.abs(target-q))))
                    header_ages.append(wall-pose['stamp'])
                    log.write(json.dumps({'elapsed': now-started, 'session_id': session, 'position': pose['position'],
                        'quat_wxyz': quat.tolist(), 'joint_pos': q.tolist(), 'joint_vel': qd.tolist(),
                        'header_ages_s': ages, 'measured_outside_file_bounds': [name for name, b in zip(names, bad) if b],
                        'observation': obs.tolist(), 'goal': goal.tolist(), 'action': action.tolist(),
                        'raw_pd_target': target.tolist(), 'eligible_for_control': False,
                        'previous_action_scope': 'counterfactual unexecuted shadow output'})+'\n')
                    previous = action; tracker.advance(); frames += 1
                    total.append((time.monotonic()-now)*1000)
                except ValueError as exc:
                    rejected[str(exc)] += 1
                    history.clear(); previous = np.zeros(n); last_frame = None
    except Exception as exc:
        fatal = repr(exc)
    finally:
        report = {'scope': __doc__, 'selected_checkpoint': 11500, 'policy_sha256': digest(r/'policy.pt'),
            'session_id': session, 'duration_s': time.monotonic()-started, 'received': dict(counts),
            'fresh_input_samples': valid, 'diagnostic_policy_frames': frames, 'rejected': dict(rejected),
            'measured_outside_file_bounds_sample_counts': dict(outside), 'raw_pd_target_outside_file_bounds_frames': raw_target_outside,
            'inference_ms': stats(latencies), 'sample_processing_ms': stats(total),
            'fresh_sample_interval_s': stats(intervals), 'pose_header_age_s': stats(header_ages),
            'max_startup_joint_error_rad': stats(max_pose_errors), 'max_target_minus_measured_rad': stats(target_deltas),
            'fatal': fatal, 'motor_output_enabled': False, 'control_transferred': False,
            'control_eligible_frames': 0, 'physical_accuracy_validated': False,
            'blockers': ['Measured joint calibration/envelope disagreement unresolved',
                         'Hardware PD target bounds and step contract not supplied',
                         'Startup tolerances and preparation/handover not validated',
                         'No hardware motor controller or independent fault handover implemented']}
        (args.out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(report, indent=2, allow_nan=False))
        node.destroy_node(); rclpy.try_shutdown()
    if fatal or frames == 0: raise SystemExit(1)


if __name__ == '__main__': main()
