"""Publish diagnostic pelvis/joint state packets from the 500 Hz Kalman estimator.

Replaces the lidar/FAST-LIO producer for the policy state topic. The estimator is
the one validated on the robot: callback-rate strapdown, Kalman with a world-frame
acceleration-bias state, still-detected zero-velocity updates, and a fixed attitude
extrinsic.

Physics safety: this process publishes STATE ONLY. It never publishes joint
commands, never creates a motor publisher, and every packet carries
motor_output_enabled=false, which the downstream contract requires.
"""
import argparse
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=120.)
    p.add_argument('--publish-rate', type=float, default=50.)
    p.add_argument('--topic', default='/supermimic/x2_online_lio/policy_state')
    p.add_argument('--fix-deg-x', type=float, default=13.11)
    p.add_argument('--fix-deg-y', type=float, default=-1.04)
    p.add_argument('--fix-deg-z', type=float, default=-0.12)
    p.add_argument('--init-bias-seconds', type=float, default=3.)
    p.add_argument('--velocity-sigma', type=float, default=.02)
    p.add_argument('--still-accel-tol', type=float, default=.5)
    p.add_argument('--still-gyro-tol', type=float, default=.08)
    p.add_argument('--still-hold', type=float, default=.2)
    p.add_argument('--max-age', type=float, default=.05)
    p.add_argument('--src', type=Path, default=None,
                   help='source root containing supermimic/deploy/policy_packet.py; '
                        'falls back to $SM_SRC. Validation must not be skipped.')
    a = p.parse_args()
    if not 1. <= a.seconds <= 600.:
        raise ValueError('Bounded duration required')
    if os.environ.get('SM_MOTOR_OUTPUT') not in (None, '', '0'):
        raise SystemExit('State producer must never run with motor output enabled')

    import rclpy
    from rclpy.node import Node
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from std_msgs.msg import String
    from aimdk_msgs.msg import JointStateArray
    from sensor_msgs.msg import Imu
    from scipy.spatial.transform import Rotation
    import sys
    src = a.src or (Path(os.environ['SM_SRC']) if os.environ.get('SM_SRC') else None)
    if src is None:
        raise SystemExit('A source root is required to import the state-packet validator')
    sys.path.insert(0, str(src))
    from supermimic.deploy.policy_packet import validate_policy_packet

    names = json.loads(a.profile.read_text())['joint_names']
    a.out.mkdir(parents=True, exist_ok=False)
    session = 'imu-kalman-500hz-' + uuid.uuid4().hex[:12]
    mount = Rotation.from_euler('xyz', [a.fix_deg_x, a.fix_deg_y, a.fix_deg_z], degrees=True).as_matrix()
    state = {'last': None, 'still_since': None, 'integrate': False,
             'x': np.zeros(6), 'P': np.diag([.01, .01, .1, .1, .1, .1])}
    H = np.zeros((2, 6))
    H[0, 2] = H[1, 3] = 1.
    Rm = np.eye(2) * (a.velocity_sigma ** 2)
    Q = np.diag([1e-8, 1e-8, 4e-5, 4e-5, 1e-8, 1e-8])
    identity = np.eye(6)
    counters = {'imu': 0, 'packets': 0, 'invalid': 0, 'skipped_age': 0, 'zupt': 0}
    timing = {'age_max': 0., 'publish_ms_max': 0.}

    class Producer(Node):
        def __init__(self):
            super().__init__('supermimic_imu_state_producer')
            qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE)
            self.imu = None
            self.groups = {}
            self.pub = self.create_publisher(String, a.topic, qos)
            self.subs = [self.create_subscription(Imu, '/aima/hal/imu/torso/state', self.on_imu, qos)]
            self.subs += [self.create_subscription(JointStateArray, f'/aima/hal/joint/{g}/state',
                                                   lambda msg, group=g: self.on_joint(group, msg), qos)
                          for g in ('leg', 'waist', 'head', 'arm')]

        def on_imu(self, msg):
            q, w, acc = msg.orientation, msg.angular_velocity, msg.linear_acceleration
            stamp_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            self.imu = {'recv': time.time(), 'mono': time.monotonic(), 'stamp_ns': stamp_ns,
                        'q': [q.w, q.x, q.y, q.z], 'w': [w.x, w.y, w.z],
                        'a': [acc.x, acc.y, acc.z], 'frame': msg.header.frame_id}
            counters['imu'] += 1
            if state['integrate']:
                propagate(self.imu)

        def on_joint(self, group, msg):
            self.groups[group] = {
                'recv': time.time(),
                'stamp_ns': msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec,
                'domain': int(msg.state.value),
                'position': {j.name: j.position for j in msg.joints},
                'velocity': {j.name: j.velocity for j in msg.joints}}

    def propagate(sample):
        x, P = state['x'], state['P']
        now = sample['mono']
        dt = now - state['last'] if state['last'] is not None else .002
        state['last'] = now
        dt = float(min(max(dt, 1e-4), .05))
        acc_body = np.array(sample['a'])
        gyro = np.array(sample['w'])
        world = mount @ (-Rotation.from_quat(sample['q']).apply(acc_body))
        u = world[:2]
        F = np.eye(6)
        F[0, 2] = F[1, 3] = dt
        F[0, 4] = F[1, 5] = -.5 * dt ** 2
        F[2, 4] = F[3, 5] = -dt
        B = np.zeros((6, 2))
        B[0, 0] = B[1, 1] = .5 * dt ** 2
        B[2, 0] = B[3, 1] = dt
        x = F @ x + B @ u
        P = F @ P @ F.T + Q * (dt / .02)
        still = (abs(np.linalg.norm(acc_body) - 9.81) <= a.still_accel_tol
                 and np.linalg.norm(gyro) <= a.still_gyro_tol)
        state['still_since'] = (state['still_since'] if state['still_since'] is not None else now) if still else None
        if still and now - state['still_since'] >= a.still_hold:
            innovation = np.zeros(2) - H @ x
            S = H @ P @ H.T + Rm
            K = np.linalg.solve(S, (P @ H.T).T).T
            x = x + K @ innovation
            P = (identity - K @ H) @ P @ (identity - K @ H).T + K @ Rm @ K.T
            counters['zupt'] += 1
        state['x'], state['P'] = x, P

    rclpy.init()
    node = Producer()

    # Static settle: bias is not fitted here, the extrinsic fixes the model, so a
    # short window only removes any residual offset.
    if a.init_bias_seconds > 0.:
        window = []
        end = time.monotonic() + a.init_bias_seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=.01)
            if node.imu is not None:
                window.append(mount @ (-Rotation.from_quat(node.imu['q']).apply(np.array(node.imu['a']))))
        if len(window) < 100:
            raise SystemExit('Static window produced too few samples')
        state['x'][4:6] = np.array(window).mean(axis=0)[:2]
        print('initial bias %s' % np.round(state['x'][4:6], 5), flush=True)

    frames = (a.out / 'packets.jsonl').open('x')
    started = time.monotonic()
    state['integrate'] = True
    last_publish = 0.
    publish_period = 1. / a.publish_rate
    try:
        while time.monotonic() - started < a.seconds:
            rclpy.spin_once(node, timeout_sec=.001)
            now = time.monotonic()
            if now - last_publish < publish_period:
                continue
            last_publish = now
            if node.imu is None or len(node.groups) < 4:
                counters['skipped_age'] += 1
                continue
            age = time.time() - node.imu['recv']
            timing['age_max'] = max(timing['age_max'], age)
            if age > a.max_age:
                counters['skipped_age'] += 1
                continue
            try:
                position_by_name, velocity_by_name = {}, {}
                for group in node.groups.values():
                    position_by_name.update(group['position'])
                    velocity_by_name.update(group['velocity'])
                if any(n not in position_by_name for n in names):
                    counters['skipped_age'] += 1
                    continue
                q = [position_by_name[n] for n in names]
                qd = [velocity_by_name[n] for n in names]
            except KeyError:
                counters['skipped_age'] += 1
                continue
            x = state['x']
            gravity = (Rotation.from_quat(node.imu['q']).as_matrix().T @ np.array([0., 0., -1.]))
            group_stamps = [node.groups[g]['stamp_ns'] for g in ('leg', 'waist', 'head', 'arm')]
            group_recv = [node.groups[g]['recv'] for g in ('leg', 'waist', 'head', 'arm')]
            # The observation is complete only once every stream has been acquired, so
            # the packet timestamp is the newest of the five. Joints must never appear
            # to be acquired after the state they belong to.
            state_stamp_ns = max([int(node.imu['stamp_ns']), *[int(t) for t in group_stamps]])
            packet = {
                'schema': 2, 'joint_names': list(names), 'session_id': session,
                'motor_output_enabled': False,
                'position': [float(x[0]), float(x[1]), 0.],
                'quat_wxyz': [float(v) for v in (Rotation.from_matrix(mount) *
                                                 Rotation.from_quat(node.imu['q'])).as_quat()[[3, 0, 1, 2]]],
                'angular_velocity_body': [float(v) for v in node.imu['w']],
                'gravity_body': [float(v) for v in gravity],
                'joint_position': [float(v) for v in q],
                'joint_velocity': [float(v) for v in qd],
                'joint_domains': [int(node.groups[g]['domain']) for g in ('leg', 'waist', 'head', 'arm')],
                'joint_stamps': [t * 1e-9 for t in group_stamps],
                'joint_stamps_ns': [int(t) for t in group_stamps],
                'joint_received_wall': group_recv,
                'state_stamp': state_stamp_ns * 1e-9,
                'state_stamp_ns': int(state_stamp_ns),
                'published_wall': time.time(),
            }
            try:
                validate_policy_packet(packet, names, time.time(), max_age=a.max_age)
            except ValueError as exc:
                counters['invalid'] += 1
                frames.write(json.dumps({'invalid': str(exc)}, allow_nan=False) + '\n')
                continue
            publish_start = time.monotonic()
            msg = String()
            msg.data = json.dumps(packet, allow_nan=False)
            node.pub.publish(msg)
            counters['packets'] += 1
            timing['publish_ms_max'] = max(timing['publish_ms_max'], (time.monotonic() - publish_start) * 1e3)
            frames.write(json.dumps({'t': now - started, 'position': packet['position'][:2],
                                     'velocity': [float(x[2]), float(x[3])],
                                     'bias': [float(x[4]), float(x[5])],
                                     'age': age}, allow_nan=False) + '\n')
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        frames.flush()
        frames.close()
        node.destroy_node()
        rclpy.try_shutdown()
        elapsed = time.monotonic() - started
        summary = {'seconds': elapsed, 'session_id': session, 'topic': a.topic,
                   'publish_rate_hz': counters['packets'] / elapsed if elapsed else None,
                   'counters': counters, 'timing': {k: float(v) for k, v in timing.items()},
                   'extrinsic_deg': [a.fix_deg_x, a.fix_deg_y, a.fix_deg_z],
                   'motor_output_enabled': False, 'control_ready': False,
                   'reason': 'state producer only; no policy targets, no actuator path',
                   'limitations': ('Horizontal position only; height field is zero because no height is '
                                   'estimated. Motion accuracy untested; the zero-velocity update only '
                                   'applies while the IMU reports a still robot.')}
        (a.out / 'summary.json').write_text(json.dumps(summary, allow_nan=False) + '\n')
        print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
