"""State-only hardware bridge: raw pelvis IMU + encoders, no firmware attitude.

Run inside PC2 lxh. --observe-only creates no ROS publishers. Calibration needs
an operator declaration of physical rest; thresholds alone cannot prove rest.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import time
import uuid
import xml.etree.ElementTree as ET

GROUPS = ('leg', 'waist', 'head', 'arm')


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
    p.add_argument('--assume-initial-rest', action='store_true')
    p.add_argument('--seconds', type=float, default=0.)
    a = p.parse_args()
    if not a.observe_only and not a.assume_initial_rest:
        p.error('State output requires --assume-initial-rest after physical rest is confirmed')
    if a.observe_only and a.seconds <= 0:
        p.error('Read-only observation requires a positive bounded --seconds')
    import numpy as np
    import rclpy
    from rclpy.node import Node
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
    estimator = None if a.observe_only else EncoderImuState(
        a.model, names, offset, interval_average=False, point_trapezoid=True,
        encoder_stationary_gate=False)
    session = 'pelvis-raw-imu-'+uuid.uuid4().hex
    history = {g: deque(maxlen=128) for g in GROUPS}
    window = deque()
    started = time.monotonic()
    stats = {'imu': 0, 'packets': 0, 'joints': 0, 'calibrated': False,
             'max_header_gap_s': 0., 'max_receive_age_s': 0., 'aligned_imu': 0}
    last_stamp = None
    origin_ns = None
    last_publish = None
    last_receive = None
    fault = None
    node = None
    rclpy.init()
    try:
        node = Node('supermimic_pelvis_raw_imu', enable_rosout=False, start_parameter_services=False)
        qos = QoSProfile(depth=3, reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        pub = None if a.observe_only else node.create_publisher(String, '/supermimic/x2_online_lio/policy_state', qos)

        def on_joint(group, msg):
            stamp = msg.header.stamp.sec*10**9+msg.header.stamp.nanosec
            row = {'event': 'joint', 'group': group, 'stamp_ns': stamp, 'recv': time.time(),
                   'domain': int(msg.state.value),
                   'position': {j.name: float(j.position) for j in msg.joints},
                   'velocity': {j.name: float(j.velocity) for j in msg.joints}}
            log.emit('support-events', row)
            if stamp <= 0 or (history[group] and stamp <= history[group][-1]['stamp_ns']):
                raise ValueError('Joint timestamp duplicate/rollback')
            if len(row['position']) != len(msg.joints):
                raise ValueError('Duplicate joint names')
            if not all(np.isfinite(v) for key in ('position', 'velocity') for v in row[key].values()):
                raise ValueError('Nonfinite encoders')
            history[group].append(row)
            stats['joints'] += 1

        def on_imu(msg):
            nonlocal last_stamp, origin_ns, last_publish, last_receive
            stamp = msg.header.stamp.sec*10**9+msg.header.stamp.nanosec
            gyro = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
            acc = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
            wall = time.time()
            sample = {'event': 'imu', 'stamp_ns': stamp, 'recv': wall,
                      'gyro': gyro, 'acceleration': acc, 'frame': msg.header.frame_id}
            log.emit('support-events', sample)
            stats['imu'] += 1
            age = wall-stamp*1e-9
            stats['max_receive_age_s'] = max(stats['max_receive_age_s'], age)
            gap = None if last_stamp is None else (stamp-last_stamp)*1e-9
            if gap is not None:
                stats['max_header_gap_s'] = max(stats['max_header_gap_s'], gap)
            last_stamp = stamp
            last_receive = time.monotonic()
            aligned = causal_joints(history, stamp, names)
            if aligned is not None:
                stats['aligned_imu'] += 1
            if a.observe_only:
                return
            if stamp <= 0 or not np.isfinite(gyro+acc).all() or not 0 <= age <= .035:
                raise ValueError('Invalid/stale/future IMU input')
            if gap is not None and not 0 < gap <= .03:
                raise ValueError('IMU duplicate/rollback or gap above 30 ms; no gap bridging')
            if origin_ns is None:
                origin_ns = stamp
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
                    try:
                        calibration = fit_static_initialization(list(window), declared_stationary=True)
                    except ValueError:
                        return
                    estimator.filter.initialize(calibration)
                    (a.out/'calibration.json').write_text(json.dumps(calibration, indent=2)+'\n')
                    stats['calibrated'] = True
                    print('Stationary initialization passed; gyro bias:', calibration['gyro_bias'], flush=True)
                else:
                    return
            estimator.update(sample['time_s'], gyro, acc)
            if last_publish is not None and stamp-last_publish < 20_000_000:
                return
            if aligned is None:
                return
            rows, q, qd = aligned
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
            validate_policy_packet(packet, names, time.time(), max_age=.035)
            log.emit('packets', packet)
            pub.publish(String(data=json.dumps(packet, allow_nan=False)))
            last_publish = stamp
            stats['packets'] += 1
            # A live heartbeat, not hardware accuracy or motor readiness approval.
            ready = a.out/'ready.json'
            temporary = a.out/'ready.tmp'
            if stats['packets'] == 1 or stats['packets'] % 10 == 0:
                temporary.write_text(json.dumps({'wall': time.time(), 'session_id': session,
                                            'packets': stats['packets'], 'motor_readiness_approved': False}))
                temporary.replace(ready)

        subscriptions = [node.create_subscription(Imu, '/aima/hal/imu/torso/state', on_imu, qos)]
        subscriptions += [node.create_subscription(JointStateArray, '/aima/hal/joint/'+g+'/state',
                           lambda msg, group=g: on_joint(group, msg), qos) for g in GROUPS]
        while rclpy.ok() and (a.seconds <= 0 or time.monotonic()-started < a.seconds):
            rclpy.spin_once(node, timeout_sec=.005)
            if not a.observe_only:
                if time.monotonic()-started > 20 and not stats['calibrated']:
                    raise ValueError('No valid three-second stationary window within 20 seconds')
                if last_receive is not None and time.monotonic()-last_receive > .035:
                    raise ValueError('IMU receive timeout')
                if last_publish is not None and time.time()-last_publish*1e-9 > .05:
                    raise ValueError('Coherent state output timeout')
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        fault = str(exc)
        raise
    finally:
        (a.out/'ready.json').unlink(missing_ok=True)
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()
        writer = log.close()
        stats.update(fault=fault, duration_s=time.monotonic()-started, writer=writer,
                     observe_only=a.observe_only, firmware_quaternion_used=False,
                     interval_average=False, point_trapezoid=True, online_zupt=False,
                     online_gravity_leveling=False, imu_offset=offset, motor_readiness_approved=False)
        (a.out/'summary.json').write_text(json.dumps(stats, indent=2)+'\n')
        print(json.dumps(stats), flush=True)
        if not writer['success']:
            raise RuntimeError('Raw diagnostic log did not drain successfully')


if __name__ == '__main__':
    main()
