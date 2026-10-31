"""Bounded diagnostic policy worker on atomic online state, without actuation."""
import argparse
from collections import Counter, deque
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def percentiles(values):
    return dict(zip(('p50', 'p95', 'p99', 'max'), np.percentile(values, [50, 95, 99, 100]).tolist())) if values else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('release', 'out', 'fixture'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--seconds', type=float, default=60.)
    p.add_argument('--numpy-policy', type=Path)
    p.add_argument('--numpy-sha256')
    p.add_argument('--diagnostic-only', action='store_true', required=True)
    a = p.parse_args()
    if not 10 <= a.seconds <= 180: raise ValueError('Expected bounded diagnostic duration')
    r = a.release.resolve()
    digest = hashlib.sha256((r/'policy.pt').read_bytes()).hexdigest()
    if digest != '4424f231fc1896da098b2fca366e8522df6791596638b2b9257ba682520bbe4c':
        raise ValueError('Expected selected 11500 export')
    candidate = json.loads((r/'startup-candidate.json').read_text())
    names = candidate['joint_names']; default = np.asarray(candidate['action_default']); scale = np.asarray(candidate['action_scale'])
    lower = np.asarray(candidate['measured_lower_candidate']); upper = np.asarray(candidate['measured_upper_candidate'])
    os.environ['SM_ROBOT'] = 'x2_ultra'; os.environ['SM_ROBOT_SPEC'] = str(r/'fk-only/profile.json')
    import torch
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from std_msgs.msg import String
    from supermimic.deploy.torch_policy import BufferedTorchPolicy
    from supermimic.deploy.policy_packet import validate_policy_packet
    from supermimic.deploy.deferred_policy_log import DeferredPolicyLog
    from supermimic.deploy.skill_goal import SkillGoal
    torch.set_num_threads(1)
    if bool(a.numpy_policy) != bool(a.numpy_sha256): raise ValueError('Both NumPy archive and expected SHA required')
    if a.numpy_policy:
        from supermimic.deploy.numpy_policy import NumpyPolicy
        policy = NumpyPolicy(a.numpy_policy, expected_sha256=a.numpy_sha256)
    else:
        policy = BufferedTorchPolicy(r/'policy.pt')
    fixture = json.loads(a.fixture.read_text())
    for _ in range(20): output = policy.infer(fixture['obs'], fixture['goal'])
    fixture_error = float(np.max(np.abs(output-fixture['expected'])))
    if fixture_error > 1e-4: raise ValueError('Fixture mismatch')
    warm_tracker = SkillGoal(r/'dataset', r/'fk-only/profile.json', [0., 0., .65], [1., 0., 0., 0.])
    for _ in range(10): warm_tracker.goal([0., 0., .65], [1., 0., 0., 0.], default, 'stand')
    # Discard all warmup state; real observation history is still collected later.
    del warm_tracker
    tracker = SkillGoal(r/'dataset', r/'fk-only/profile.json', [0., 0., .65], [1., 0., 0., 0.])
    a.out.mkdir(parents=True, exist_ok=False)
    rclpy.init(); node = Node('supermimic_stream_policy_probe', enable_rosout=False, start_parameter_services=False)
    latest = {}; received = 0; last_arrival = None
    arrival_intervals = []; acquisition_intervals = []; publication_intervals = []
    gap_attribution = Counter(); gap_examples = []
    def receive(msg):
        nonlocal latest, received, last_arrival
        latest = json.loads(msg.data); received += 1
        arrival = (time.monotonic(), latest['published_wall'], latest['state_stamp_ns'])
        if last_arrival is not None:
            recv_gap, pub_gap = arrival[0]-last_arrival[0], arrival[1]-last_arrival[1]
            source_gap = (arrival[2]-last_arrival[2])*1e-9
            arrival_intervals.append(recv_gap); publication_intervals.append(pub_gap)
            acquisition_intervals.append(source_gap)
            if source_gap > .03:
                cause = 'producer publication gap' if pub_gap > .03 else 'acquisition jump within regular publications'
                gap_attribution[cause] += 1
                if len(gap_examples) < 20:
                    gap_examples.append(dict(receive_gap_s=recv_gap, publication_gap_s=pub_gap,
                        acquisition_gap_s=source_gap, source_age_at_publication_s=arrival[1]-arrival[2]*1e-9))
            elif recv_gap > .03:
                gap_attribution['receive gap without acquisition gap'] += 1
        last_arrival = arrival
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
    sub = node.create_subscription(String, '/supermimic/x2_online_lio/policy_state', receive, qos)
    executor = SingleThreadedExecutor(); executor.add_node(node)
    history = deque(maxlen=10); previous = np.zeros(31); session = None; last_stamp = None
    initialized = False; rejected = Counter(); outside = Counter(); frames = 0
    durations, inference, intervals, ages, steps = [], [], [], [], []
    started = time.monotonic(); processed_received = 0; previous_target = None; last_begin = None; fatal = None
    slowest_frame = None
    log = DeferredPolicyLog(a.out/'frames.jsonl', names, capacity=int(a.seconds*60)+64,
                            observation_size=len(fixture['obs']), goal_size=len(fixture['goal']))
    try:
        with log:
            while time.monotonic()-started < a.seconds:
                # The producer already samples at 50 Hz. A second independent
                # 50 Hz timer aliases its phase/jitter and can skip a real sample,
                # breaking the ten-frame history even when both loops keep up.
                # Consume each received state once, using its acquisition clock
                # for history validation; never pad a missing observation.
                executor.spin_once(timeout_sec=.02)
                if received == processed_received:
                    if latest and time.time()-latest['state_stamp'] > .05 and history:
                        history.clear(); previous = np.zeros(31); last_stamp = None
                        rejected['state stream stopped or stale'] += 1
                    continue
                processed_received = received
                begin = time.monotonic()
                try:
                    if not latest: raise ValueError('waiting for atomic state')
                    row = validate_policy_packet(latest, names, time.time())
                    if node.count_publishers('/supermimic/x2_online_lio/policy_state') != 1:
                        raise RuntimeError('Expected one atomic state publisher')
                    if session is not None and session != row['session_id']:
                        raise RuntimeError('State session changed')
                    session = row['session_id']; stamp = row['state_stamp']
                    if last_stamp is not None:
                        if stamp <= last_stamp:
                            rejected['no new acquisition'] += 1
                            continue
                        if stamp-last_stamp > .03:
                            history.clear(); previous = np.zeros(31); rejected['history gap over 30 ms'] += 1
                    q = np.asarray(row['joint_position']); qd = np.asarray(row['joint_velocity'])
                    quat = np.asarray(row['quat_wxyz']); omega = np.asarray(row['angular_velocity_body'])
                    w, x, y, z = quat
                    gravity = np.array([2*(w*y-x*z), -2*(w*x+y*z), -1+2*(x*x+y*y)])
                    history.append((q-default, qd, omega, gravity, previous.copy())); last_stamp = stamp
                    if not initialized:
                        tracker.position = np.asarray(row['position']).copy()
                        tracker.yaw = tracker.rotation(quat).as_euler('xyz')[2]; initialized = True
                    if len(history) < 10: continue
                    goal_begin=time.monotonic()
                    obs = np.concatenate([h[term] for term in range(5) for h in history])
                    goal = tracker.goal(row['position'], quat, q, 'stand')
                    goal_end=time.monotonic()
                    infer_begin = time.perf_counter(); action = policy.infer(obs, goal)
                    inference.append((time.perf_counter()-infer_begin)*1000)
                    target = default+scale*action
                    bad = [name for name, v, lo, hi in zip(names, q, lower, upper) if not lo <= v <= hi]
                    outside.update(bad)
                    if last_begin is not None: intervals.append(begin-last_begin)
                    last_begin = begin; ages.append(time.time()-stamp)
                    if previous_target is not None: steps.append(float(np.max(np.abs(target-previous_target))))
                    log_begin=time.monotonic()
                    log.append(elapsed=begin-started, state_stamp=stamp, session_id=session,
                        joint_position=q, position=row['position'], observation=obs, goal=goal,
                        action=action, raw_pd_target=target, measured_outside_file_bounds=bad)
                    previous = action; previous_target = target; tracker.advance(); frames += 1
                    duration=(time.monotonic()-begin)*1000;durations.append(duration)
                    if slowest_frame is None or duration>slowest_frame['total_ms']:
                        slowest_frame={'elapsed':begin-started,'total_ms':duration,
                            'validation_graph_and_history_ms':(goal_begin-begin)*1000,
                            'observation_goal_ms':(goal_end-goal_begin)*1000,
                            'inference_ms':inference[-1],'log_ms':(time.monotonic()-log_begin)*1000}
                except ValueError as exc:
                    rejected[str(exc)] += 1; history.clear(); previous = np.zeros(31); last_stamp = None
    except Exception as exc: fatal = repr(exc)
    finally:
        result = dict(scope=__doc__, policy_sha256=digest,
            seconds=(log.capture_finished or time.monotonic())-started,
            log_export_seconds=log.export_seconds, log_records=log.count,
            log_capacity_bytes=log.rows.nbytes+log.outside.nbytes,
            backend='numpy-fp32' if a.numpy_policy else 'torch-frozen-fp32', numpy_archive_sha256=a.numpy_sha256,
            sampling='one evaluation per received state; producer 50 Hz; acquisition-time history',
            received=received, frames=frames, session_id=session, rejected=dict(rejected),
            received_interval_s=percentiles(arrival_intervals), publication_interval_s=percentiles(publication_intervals),
            acquisition_interval_s=percentiles(acquisition_intervals), gap_attribution=dict(gap_attribution),
            gap_examples=gap_examples,
            fixture_error=fixture_error, inference_ms=percentiles(inference), processing_ms=percentiles(durations),
            policy_frame_interval_s=percentiles(intervals), state_age_after_inference_s=percentiles(ages),
            raw_target_step_rad=percentiles(steps), measured_outside_file_bounds_frame_counts=dict(outside),
            processing_over_20ms=sum(t>20 for t in durations), fatal=fatal,
            slowest_frame=slowest_frame,
            motor_output_enabled=False, control_transferred=False, ready_for_hardware_trial=False)
        (a.out/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps(result, indent=2, allow_nan=False))
        executor.shutdown(timeout_sec=1.); node.destroy_node(); rclpy.try_shutdown()
    if fatal or not frames: raise SystemExit(1)


if __name__ == '__main__': main()
