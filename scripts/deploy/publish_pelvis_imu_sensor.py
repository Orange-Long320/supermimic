"""State-only hardware bridge: raw pelvis IMU + encoders, no firmware attitude.

Run inside PC2 lxh. --observe-only creates no ROS publishers. Calibration needs
an operator declaration of physical rest; thresholds alone cannot prove rest.
"""
import argparse
from collections import deque
import json
from math import isfinite
from pathlib import Path
import time
import uuid
import xml.etree.ElementTree as ET

GROUPS = ('leg', 'waist', 'head', 'arm')
# Operator requested modest transport-jitter tolerance on 2026-11-04.
MAX_IMU_GAP_S = .040
MAX_IMU_RECEIVE_GAP_S = .050
ENCODER_PERIOD_S = .010  # 100 Hz encoder reads for 50 Hz policy states.


def pelvis_mount(urdf):
    root = ET.parse(urdf).getroot()
    joint = root.find("joint[@name='imu_in_pelvis_joint']")
    if joint is None or joint.attrib['type'] != 'fixed' or joint.find('parent').attrib['link'] != 'pelvis':
        raise ValueError('Expected SDK pelvis IMU fixed mount')
    origin = joint.find('origin')
    if any(float(x) != 0. for x in origin.attrib.get('rpy', '0 0 0').split()):
        raise ValueError('Nonidentity IMU axes need an explicitly verified transform')
    return [float(x) for x in origin.attrib['xyz'].split()]


def causal_joints(history, stamp, names, max_age=.025):
    rows = [next((row for row in reversed(history[g]) if row['stamp_ns'] <= stamp), None) for g in GROUPS]
    if any(r is None or not 0 <= (stamp-r['stamp_ns'])*1e-9 <= max_age for r in rows):
        return None
    if any(r['domain'] != 0 for r in rows):
        raise ValueError('Joint state domain is not position')
    positions = {k: v for r in rows for k, v in r['position'].items()}
    velocities = {k: v for r in rows for k, v in r['velocity'].items()}
    return rows, [positions[n] for n in names], [velocities[n] for n in names]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--urdf', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--observe-only', action='store_true')
    p.add_argument('--record-all-raw', action='store_true', help='Full-rate diagnostic capture; default deployment logs at 50 Hz')
    p.add_argument('--imu-reader', type=Path, help='SDK raw reader executable; no fusion or motor output')
    p.add_argument('--diagnostic-calibration', type=Path,
                   help='Exercise full processing with saved calibration; observe-only, no ROS publisher')
    p.add_argument('--assume-initial-rest', action='store_true')
    p.add_argument('--seconds', type=float, default=0.)
    a = p.parse_args()
    if a.diagnostic_calibration and not a.observe_only:
        p.error('Saved calibration is allowed only for bounded --observe-only diagnostics')
    if not a.observe_only and not a.assume_initial_rest:
        p.error('State output requires --assume-initial-rest after physical rest is confirmed')
    if a.observe_only and a.seconds <= 0:
        p.error('Read-only observation requires a positive bounded --seconds')
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from rclpy.impl.implementation_singleton import rclpy_implementation as ros_impl
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import Imu
    from std_msgs.msg import String
    from aimdk_msgs.msg import JointStateArray
    from supermimic.deploy.imu_dead_reckoning import EncoderImuState, fit_static_initialization
    from supermimic.deploy.policy_packet import validate_policy_packet
    from supermimic.deploy.async_sim_diagnostic_log import AsyncDiagnosticLog
    names = json.loads(a.profile.read_text())['joint_names']
    offset = pelvis_mount(a.urdf)
    a.out.mkdir(parents=True, exist_ok=False)
    log = AsyncDiagnosticLog(a.out)
    processing = not a.observe_only or a.diagnostic_calibration is not None
    estimator = EncoderImuState(
        a.model, names, offset, interval_average=False, point_trapezoid=True,
        encoder_stationary_gate=False, max_gap_seconds=MAX_IMU_GAP_S) if processing else None
    session = 'pelvis-raw-imu-'+uuid.uuid4().hex
    history = {g: deque(maxlen=32) for g in GROUPS}
    joint_seen_stamps = {}
    window = deque()
    started = time.monotonic()
    stats = {'imu': 0, 'packets': 0, 'joints': 0, 'calibrated': False,
             'max_header_gap_s': 0., 'max_receive_age_s': 0., 'aligned_imu': 0,
             'calibration_compute_ms': None, 'startup_waiting_fresh': 0,
             'withheld_stale_candidates': 0, 'max_imu_callback_ms': 0.,
             'max_joint_callback_ms': 0., 'max_heartbeat_work_ms': 0., 'alignment_checks': 0,
             'startup_stale_imu': 0, 'joints_received': 0, 'encoder_polls': 0,
             'max_imu_batch': 0, 'buffered_imu_integrated': 0, 'max_output_silence_s': 0.,
             'raw_records': 0}
    last_stamp = None
    origin_ns = None
    last_publish = None
    last_publish_mono = None
    last_heartbeat_mono = None
    last_receive = None
    last_raw_record_stamp = None
    calibration = None
    calibration_stamp = None
    calibrated_at = None
    fault = None
    node = None
    native_reader = None
    if a.diagnostic_calibration:
        calibration = json.loads(a.diagnostic_calibration.read_text())
        estimator.filter.initialize(calibration)
        stats['calibrated'] = True
    rclpy.init()
    try:
        node = Node('supermimic_pelvis_raw_imu', enable_rosout=False, start_parameter_services=False)
        qos = QoSProfile(depth=3, reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        pub = None if a.observe_only else node.create_publisher(String, '/supermimic/x2_online_lio/policy_state', qos)

        def on_joint(group, msg):
            stamp = msg.header.stamp.sec*10**9+msg.header.stamp.nanosec
            stats['joints_received'] += 1
            if stamp <= joint_seen_stamps.get(group, 0):
                raise ValueError('Joint timestamp duplicate/rollback')
            joint_seen_stamps[group] = stamp
            if int(msg.state.value) != 0:
                raise ValueError('Joint state domain is not position')
            row = {'event': 'joint', 'group': group, 'stamp_ns': stamp, 'recv': time.time(),
                   'domain': int(msg.state.value),
                   'position': {j.name: float(j.position) for j in msg.joints},
                   'velocity': {j.name: float(j.velocity) for j in msg.joints}}
            log.emit('support-events', row)
            if stamp <= 0 or (history[group] and stamp <= history[group][-1]['stamp_ns']):
                raise ValueError('Joint timestamp duplicate/rollback')
            if len(row['position']) != len(msg.joints):
                raise ValueError('Duplicate joint names')
            if not all(isfinite(v) for key in ('position', 'velocity') for v in row[key].values()):
                raise ValueError('Nonfinite encoders')
            history[group].append(row)
            stats['joints'] += 1

        def on_imu(msg, emit_state=True):
            nonlocal last_stamp, origin_ns, last_publish, last_receive, last_publish_mono, last_heartbeat_mono
            nonlocal calibration, calibration_stamp, calibrated_at, last_raw_record_stamp
            stamp = msg.header.stamp.sec*10**9+msg.header.stamp.nanosec
            gyro = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
            acc = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
            wall = time.time()
            sample = {'event': 'imu', 'stamp_ns': stamp, 'recv': wall,
                      'gyro': gyro, 'acceleration': acc, 'frame': msg.header.frame_id}
            if hasattr(msg, 'transport_received_wall_ns'):
                sample['transport_received_wall_ns'] = msg.transport_received_wall_ns
                if msg.native_joint_snapshot is not None:
                    rows, q, qd = msg.native_joint_snapshot
                    sample.update(joint_pos=q, joint_vel=qd,
                                  joint_stamps_ns=[r['stamp_ns'] for r in rows])
            # Logging every 500 Hz frame adds a serialization thread to the
            # control hot path. Preserve full calibration data; runtime capture
            # defaults to the policy rate, without skipping any integration.
            if (a.record_all_raw or not processing or not stats['calibrated'] or
                    last_raw_record_stamp is None or stamp-last_raw_record_stamp >= 20_000_000):
                log.emit('support-events', sample)
                last_raw_record_stamp = stamp
                stats['raw_records'] += 1
            stats['imu'] += 1
            age = wall-stamp*1e-9
            stats['max_receive_age_s'] = max(stats['max_receive_age_s'], age)
            gap = None if last_stamp is None else (stamp-last_stamp)*1e-9
            if gap is not None:
                stats['max_header_gap_s'] = max(stats['max_header_gap_s'], gap)
            last_stamp = stamp
            last_receive = time.monotonic()
            needs_encoders = not stats['calibrated'] or (emit_state and (last_publish is None or stamp-last_publish >= 20_000_000))
            aligned = (msg.native_joint_snapshot if native_reader else causal_joints(history, stamp, names)) if needs_encoders else None
            if needs_encoders:
                stats['alignment_checks'] += 1
            if aligned is not None:
                stats['aligned_imu'] += 1
            if not processing:
                return
            # This mode skips the normal three-second calibration acquisition.
            # Give its subscriptions the same warm-up before timing processing.
            if a.diagnostic_calibration and time.monotonic()-started < 3.:
                return
            if origin_ns is None and age > .035:
                # No integration/calibration has begun. Discovery can initially
                # deliver an old sample; wait for fresh input, never retimestamp it.
                stats['startup_stale_imu'] += 1
                return
            if stamp <= 0 or not np.isfinite(gyro+acc).all() or age < 0:
                raise ValueError('Invalid/future IMU input')
            if age > .035:
                if not stats['calibrated']:
                    window.clear()
                    stats['startup_stale_imu'] += 1
                    return
                stats['buffered_imu_integrated'] += 1
            if gap is not None and not 0 < gap <= MAX_IMU_GAP_S:
                raise ValueError(f'IMU duplicate/rollback or gap {gap*1000:.3f} ms exceeds {MAX_IMU_GAP_S*1000:g} ms')
            if origin_ns is None:
                origin_ns = stamp
                if a.diagnostic_calibration:
                    calibration_stamp = stamp
                    calibrated_at = time.monotonic()
            sample['time_s'] = (stamp-origin_ns)*1e-9
            if not stats['calibrated']:
                if aligned is None:
                    window.clear()
                    return
                _, q, qd = aligned
                sample.update(joint_pos=q, joint_vel=qd)
                window.append(sample)
                while len(window) > 1 and window[1]['time_s'] <= sample['time_s']-3.:
                    window.popleft()
                if sample['time_s']-window[0]['time_s'] >= 3.:
                    calibration_started = time.monotonic()
                    try:
                        calibration = fit_static_initialization(list(window), declared_stationary=True,
                                                                max_gap_seconds=MAX_IMU_GAP_S)
                    except ValueError:
                        return
                    estimator.filter.initialize(calibration)
                    stats['calibration_compute_ms'] = (time.monotonic()-calibration_started)*1000
                    calibration_stamp = stamp
                    calibrated_at = time.monotonic()
                    log.emit('support-events', {'event': 'calibration', 'stamp_ns': stamp,
                                               'fit_ms': stats['calibration_compute_ms'],
                                               'calibration': calibration})
                    stats['calibrated'] = True
                    window.clear()
                    print('Stationary initialization passed; gyro bias:', calibration['gyro_bias'], flush=True)
                    # Bootstrap at the exact end of the measured rest window.
                    # Fitting takes wall time: this observation must never become
                    # the first live packet. Drain subsequent sensor callbacks and
                    # wait for all four encoder groups to advance past this point.
                    estimator.update(sample['time_s'], gyro, acc)
                    return
                else:
                    return
            estimator.update(sample['time_s'], gyro, acc)
            if not emit_state:
                return
            if last_publish is not None and stamp-last_publish < 20_000_000:
                return
            if aligned is None:
                return
            rows, q, qd = aligned
            if last_publish is None and any(r['stamp_ns'] <= calibration_stamp for r in rows):
                stats['startup_waiting_fresh'] += 1
                return
            pos, quat, omega, gravity = estimator.state(q)
            packet = {'schema': 2, 'session_id': session, 'joint_names': names,
                      'motor_output_enabled': False, 'position': pos.tolist(),
                      'quat_wxyz': quat.tolist(), 'angular_velocity_body': omega.tolist(),
                      'gravity_body': gravity.tolist(), 'joint_position': q, 'joint_velocity': qd,
                      'joint_domains': [r['domain'] for r in rows],
                      'joint_stamps_ns': [r['stamp_ns'] for r in rows],
                      'joint_stamps': [r['stamp_ns']*1e-9 for r in rows],
                      'joint_received_wall': [r['recv'] for r in rows],
                      'state_stamp_ns': stamp, 'state_stamp': stamp*1e-9, 'published_wall': time.time()}
            checked_at = time.time()
            try:
                validate_policy_packet(packet, names, checked_at, max_age=.035)
            except ValueError as exc:
                now = checked_at
                log.emit('support-events', {'event': 'packet_rejected', 'reason': str(exc),
                    'stamp_ns': stamp, 'calibration_compute_ms': stats['calibration_compute_ms'],
                    'ages_ms': {
                        'imu_header': (now-stamp*1e-9)*1000,
                        **{g+'_header': (now-r['stamp_ns']*1e-9)*1000 for g, r in zip(GROUPS, rows)},
                        **{g+'_receive': (now-r['recv'])*1000 for g, r in zip(GROUPS, rows)}}})
                # A stale combination can be replaced by the next causal
                # encoder/IMU snapshot. Propagation above is never skipped.
                # Published packets still pass the unchanged 35 ms validator;
                # consumers enforce their own state freshness and target leases.
                if str(exc) == 'policy packet stale or future':
                    values = [packet['state_stamp'], *packet['joint_stamps'],
                              packet['published_wall'], *packet['joint_received_wall']]
                    if all(t <= now for t in values):
                        stats['withheld_stale_candidates'] += 1
                        return
                raise
            encoded = json.dumps(packet, allow_nan=False)
            log.emit('packets', packet)
            if pub is not None:
                pub.publish(String(data=encoded))
            last_publish = stamp
            last_publish_mono = time.monotonic()
            stats['packets'] += 1
            # A live heartbeat, not hardware accuracy or motor readiness approval.
            if last_heartbeat_mono is None or time.monotonic()-last_heartbeat_mono >= .2:
                heartbeat_start = time.monotonic()
                log.emit('support-events', {'event': 'state_heartbeat', 'wall': time.time(),
                         'session_id': session, 'packets': stats['packets'],
                         'motor_readiness_approved': False})
                last_heartbeat_mono = time.monotonic()
                stats['max_heartbeat_work_ms'] = max(stats['max_heartbeat_work_ms'],
                                                    (time.monotonic()-heartbeat_start)*1000)

        def timed_callback(callback, key):
            def invoke(*args):
                begin = time.monotonic()
                try:
                    return callback(*args)
                finally:
                    stats[key] = max(stats[key], (time.monotonic()-begin)*1000)
            return invoke

        # Use Humble's wait_for_message take/wait API directly. A general
        # executor rebuilds task/waitable machinery for each high-rate packet.
        imu_subscription = None
        if a.imu_reader:
            from supermimic.deploy.native_imu_reader import NativeImuReader
            native_reader = NativeImuReader(a.imu_reader, a.out/'imu-reader.log', names)
        else:
            imu_subscription = node.create_subscription(Imu, '/aima/hal/imu/torso/state',
                lambda msg: None, QoSProfile(depth=64, reliability=ReliabilityPolicy.BEST_EFFORT,
                                             durability=DurabilityPolicy.VOLATILE))
        receive_imu = timed_callback(on_imu, 'max_imu_callback_ms')
        encoder_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                                 durability=DurabilityPolicy.VOLATILE)
        encoder_subscriptions = {g: node.create_subscription(
            JointStateArray, '/aima/hal/joint/'+g+'/state', lambda msg: None, encoder_qos)
            for g in GROUPS} if not native_reader else {}
        encoder_callbacks = {g: timed_callback(lambda msg, group=g: on_joint(group, msg),
                                               'max_joint_callback_ms') for g in GROUPS}
        wait_set = None if native_reader else ros_impl.WaitSet(1, 0, 0, 0, 0, 0, node.context.handle)
        last_encoder_poll = None
        while rclpy.ok() and (a.seconds <= 0 or time.monotonic()-started < a.seconds):
            if native_reader:
                batch = native_reader.read_batch()
            else:
                wait_set.clear_entities()
                wait_set.add_subscription(imu_subscription.handle)
                wait_set.wait(5_000_000)
                batch = []
                if imu_subscription.handle.pointer in wait_set.get_ready_entities('subscription'):
                    # Drain a bounded FIFO. Older samples still matter to integration;
                    # only its newest state is eligible for the live publisher.
                    with imu_subscription.handle:
                        for _ in range(64):
                            received = imu_subscription.handle.take_message(imu_subscription.msg_type, imu_subscription.raw)
                            if received is None:
                                break
                            batch.append(received[0])
            stats['max_imu_batch'] = max(stats['max_imu_batch'], len(batch))
            # Test transport idleness at the read, not after potentially slow
            # encoder decoding/integration. A nonempty batch is not a dead IMU.
            if not batch and processing and last_receive is not None and time.monotonic()-last_receive > MAX_IMU_RECEIVE_GAP_S:
                raise ValueError('IMU receive timeout')
            now = time.monotonic()
            if not stats['calibrated'] or last_encoder_poll is None or now-last_encoder_poll >= ENCODER_PERIOD_S:
                last_encoder_poll = now
                stats['encoder_polls'] += 1
                for group, subscription in encoder_subscriptions.items():
                    with subscription.handle:
                        received = subscription.handle.take_message(subscription.msg_type, subscription.raw)
                    if received is not None:
                        encoder_callbacks[group](received[0])
            for index, message in enumerate(batch):
                receive_imu(message, index == len(batch)-1)
            if processing and (not a.diagnostic_calibration or origin_ns is not None):
                if time.monotonic()-started > 20 and not stats['calibrated']:
                    raise ValueError('No valid three-second stationary window within 20 seconds')
                if calibrated_at is not None and last_publish is None and time.monotonic()-calibrated_at > .5:
                    raise ValueError('No fresh post-calibration state within 500 ms')
                if last_publish_mono is not None:
                    # This process publishes measurements, not motor commands.
                    # Missing coherent encoders withholds output; consumer packet
                    # freshness and publisher target leases own control stop logic.
                    stats['max_output_silence_s'] = max(stats['max_output_silence_s'],
                                                        time.monotonic()-last_publish_mono)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        fault = str(exc)
        raise
    finally:
        if native_reader is not None:
            stats['native_reader'] = native_reader.close()
        (a.out/'ready.json').unlink(missing_ok=True)
        if calibration is not None:
            (a.out/'calibration.json').write_text(json.dumps(calibration, indent=2)+'\n')
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()
        writer = log.close()
        # The writer can drain an already queued heartbeat during shutdown.
        (a.out/'ready.json').unlink(missing_ok=True)
        stats.update(fault=fault, duration_s=time.monotonic()-started, writer=writer,
                     observe_only=a.observe_only, firmware_quaternion_used=False,
                     record_all_raw=a.record_all_raw, runtime_raw_log_target_hz=500 if a.record_all_raw else 50,
                     diagnostic_calibration=str(a.diagnostic_calibration) if a.diagnostic_calibration else None,
                     stationary_calibration_this_run=stats['calibrated'] and not bool(a.diagnostic_calibration),
                     interval_average=False, point_trapezoid=True, online_zupt=False,
                     max_imu_gap_s=MAX_IMU_GAP_S, max_imu_receive_gap_s=MAX_IMU_RECEIVE_GAP_S,
                     max_buffered_imu_age_s=None,
                     output_progress_timeout_s=None,
                     online_gravity_leveling=False, imu_offset=offset, motor_readiness_approved=False)
        (a.out/'summary.json').write_text(json.dumps(stats, indent=2)+'\n')
        print(json.dumps(stats), flush=True)
        if not writer['success']:
            raise RuntimeError('Raw diagnostic log did not drain successfully')


if __name__ == '__main__':
    main()
