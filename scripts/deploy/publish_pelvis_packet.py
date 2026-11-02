"""Publish diagnostic pelvis/joint state packets from the 500 Hz Kalman estimator.

Diagnostic only: callback-rate strapdown, a horizontal world-frame bias state,
and still-detected zero-velocity updates. The optional moving-mount output is
experimental and requires explicitly supplied geometry. It is not deployment
validated. Its three-dimensional position is relative to the first output;
vertical acceleration bias and ZUPT are not estimated.

Physics safety: this process publishes STATE ONLY. It never publishes joint
commands, never creates a motor publisher, and every packet carries
motor_output_enabled=false, which the downstream contract requires.
"""
import argparse
import json
import os
import time
import uuid
from collections import deque
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=120.)
    p.add_argument('--publish-rate', type=float, default=50.)
    p.add_argument('--topic', default='/supermimic/x2_online_lio/policy_state')
    p.add_argument('--fix-deg-x', type=float, default=0.)
    p.add_argument('--fix-deg-y', type=float, default=0.)
    p.add_argument('--fix-deg-z', type=float, default=0.)
    p.add_argument('--auto-mount', action='store_true',
                   help='fit the world attitude correction from the static settle window')
    p.add_argument('--init-bias-seconds', type=float, default=3.)
    p.add_argument('--velocity-sigma', type=float, default=.02)
    p.add_argument('--still-accel-tol', type=float, default=.5)
    p.add_argument('--still-gyro-tol', type=float, default=.08)
    p.add_argument('--still-hold', type=float, default=.2)
    p.add_argument('--max-age', type=float, default=.05)
    p.add_argument('--model', type=Path, default=None,
                   help='FK-only model used to estimate pelvis height above the feet')
    p.add_argument('--sole-offset', type=float, default=.05,
                   help='ankle-body-origin to sole distance; declared, not measured')
    p.add_argument('--src', type=Path, default=None,
                   help='source root containing supermimic/deploy/policy_packet.py; '
                        'falls back to $SM_SRC. Validation must not be skipped.')
    p.add_argument('--imu-topic', default='/aima/hal/imu/torso/state')
    p.add_argument('--mount-urdf', type=Path, default=None,
                   help='Explicit pelvis-to-torso geometry; experimental synchronized output')
    p.add_argument('--mount-xyz', type=float, nargs=3, default=[0., 0., 0.])
    p.add_argument('--mount-rpy', type=float, nargs=3, default=[0., 0., 0.], help='Radians, torso-to-IMU')
    p.add_argument('--max-joint-extrapolation', type=float, default=.005)
    a = p.parse_args()
    if not 1. <= a.seconds <= 600.:
        raise ValueError('Bounded duration required')
    if not 0. < a.max_joint_extrapolation <= .01:
        raise ValueError('Joint extrapolation limit must be in (0, 10 ms]')
    if os.environ.get('SM_MOTOR_OUTPUT') not in (None, '', '0'):
        raise SystemExit('State producer must never run with motor output enabled')

    import rclpy
    from rclpy.node import Node
    from rclpy.executors import ExternalShutdownException
    from rclpy.impl.implementation_singleton import rclpy_implementation
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
    from supermimic.deploy.pelvis_state import (UrdfPelvisMount, pelvis_from_imu,
        pelvis_angular_velocity, pelvis_velocity_from_imu)

    names = json.loads(a.profile.read_text())['joint_names']
    waist_names = ('waist_yaw_joint', 'waist_pitch_joint', 'waist_roll_joint')
    moving_mount = (UrdfPelvisMount(a.mount_urdf, waist_names, a.mount_xyz,
                    Rotation.from_euler('xyz', a.mount_rpy).as_matrix()) if a.mount_urdf else None)
    fk = None
    if a.model is not None:
        import mujoco
        fk_model = mujoco.MjModel.from_xml_path(str(a.model))
        fk_data = mujoco.MjData(fk_model)
        fk_ids = [mujoco.mj_name2id(fk_model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
        fk_feet = [mujoco.mj_name2id(fk_model, mujoco.mjtObj.mjOBJ_BODY, s + '_ankle_roll_link')
                   for s in ('left', 'right')]
        if min(*fk_ids, *fk_feet) < 0:
            raise SystemExit('Missing FK joint or foot body')
        fk = {'model': fk_model, 'data': fk_data,
              'qpos_addr': [fk_model.jnt_qposadr[i] for i in fk_ids], 'feet': fk_feet}
    a.out.mkdir(parents=True, exist_ok=False)
    session = 'imu-kalman-500hz-' + uuid.uuid4().hex[:12]
    mount = Rotation.from_euler('xyz', [a.fix_deg_x, a.fix_deg_y, a.fix_deg_z], degrees=True).as_matrix()

    def rot_of(wxyz):
        """The device publishes w,x,y,z; scipy expects x,y,z,w."""
        return Rotation.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
    state = {'last': None, 'still_since': None, 'integrate': False, 'calibration': None,
             'z': 0., 'vz': 0., 'origin_offset': None,
             'last_published_stamp': 0,
             'x': np.zeros(6), 'P': np.diag([.01, .01, .1, .1, .1, .1])}
    H = np.zeros((2, 6))
    H[0, 2] = H[1, 3] = 1.
    Rm = np.eye(2) * (a.velocity_sigma ** 2)
    Q = np.diag([1e-8, 1e-8, 4e-5, 4e-5, 1e-8, 1e-8])
    identity = np.eye(6)
    counters = {'imu': 0, 'packets': 0, 'invalid': 0, 'skipped_age': 0, 'skipped_skew': 0,
                'skipped_joints_missing': 0, 'zupt': 0}
    timing = {'age_max': 0., 'publish_ms_max': 0.}

    class Producer(Node):
        def __init__(self):
            super().__init__('supermimic_imu_state_producer')
            qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE)
            self.imu = None
            self.imu_history = deque(maxlen=128)
            self.groups = {}
            self.history = {g: deque(maxlen=128) for g in ('leg', 'waist', 'head', 'arm')}
            self.pub = self.create_publisher(String, a.topic, qos)
            self.subs = [self.create_subscription(Imu, a.imu_topic, self.on_imu, qos)]
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
            if state['calibration'] is not None:
                state['calibration'].append(self.imu)
            if state['integrate']:
                propagate(self.imu)
                if moving_mount is not None:
                    self.imu_history.append((self.imu, state['x'].copy(), state['z'], state['vz']))

        def on_joint(self, group, msg):
            stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            if moving_mount is not None and (stamp <= 0 or
                    (self.history[group] and stamp <= self.history[group][-1]['stamp_ns'])):
                raise ValueError('Joint acquisition timestamp rollback or duplicate')
            self.groups[group] = {
                'recv': time.time(),
                'stamp_ns': stamp,
                'domain': int(msg.state.value),
                'position': {j.name: j.position for j in msg.joints},
                'velocity': {j.name: j.velocity for j in msg.joints}}
            self.history[group].append(self.groups[group])

    def propagate(sample):
        x, P = state['x'], state['P']
        now = sample['stamp_ns'] * 1e-9
        if state['last'] is None:
            state['last'] = now
            return
        dt = now - state['last']
        if not 0. < dt <= .03:
            raise ValueError('IMU acquisition timestamp rollback or gap')
        state['last'] = now
        acc_body = np.array(sample['a'])
        gyro = np.array(sample['w'])
        world = mount @ rot_of(sample['q']).apply(acc_body)
        u = world[:2]
        az = float(world[2] - 9.81)
        state['z'] += state['vz'] * dt + .5 * az * dt * dt
        state['vz'] += az * dt
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
        state['calibration'] = []
        end = time.monotonic() + a.init_bias_seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=.01)
        samples = state['calibration']
        state['calibration'] = None
        if len(samples) < 100:
            raise SystemExit('Static window produced too few samples')
        stamps = np.array([s['stamp_ns'] for s in samples], dtype=np.int64)
        acc = np.array([s['a'] for s in samples])
        gyro = np.array([s['w'] for s in samples])
        if (stamps[-1] - stamps[0] < (a.init_bias_seconds - .1) * 1e9
                or np.any(np.diff(stamps) <= 0) or np.any(np.diff(stamps) > 30_000_000)
                or not np.isfinite(acc).all() or not np.isfinite(gyro).all()
                or np.any(abs(np.linalg.norm(acc, axis=1)-9.81) > a.still_accel_tol)
                or np.any(np.linalg.norm(gyro, axis=1) > a.still_gyro_tol)):
            raise SystemExit('Calibration window is moving, invalid, or has an IMU timestamp gap')
        window = np.array([mount @ rot_of(s['q']).apply(np.array(s['a'])) for s in samples])
        if a.auto_mount:
            mean = window.mean(axis=0)
            mount = Rotation.align_vectors([[0., 0., 1.]], [mean])[0].as_matrix()
            fitted = Rotation.from_matrix(mount).as_euler('xyz', degrees=True)
            print('fitted world correction x=%.2f y=%.2f z=%.2f deg residual %.4f'
                  % (fitted[0], fitted[1], fitted[2],
                     float(np.linalg.norm(mount @ mean - np.array([0., 0., np.linalg.norm(mean)])))),
                  flush=True)
            window = np.array([mount @ v for v in window])
        state['x'][4:6] = window.mean(axis=0)[:2]
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
            imu = node.imu
            x, z, vz = state['x'], state['z'], state['vz']
            chosen = None
            if moving_mount is not None:
                for candidate, cx, cz, cvz in reversed(node.imu_history):
                    target = candidate['stamp_ns']
                    if target <= state['last_published_stamp'] or time.time()-candidate['recv'] > a.max_age:
                        break
                    selected = {}
                    for g, history in node.history.items():
                        options = [row for row in history if row['stamp_ns'] <= target]
                        if not options:
                            break
                        row = options[-1]
                        delta = (target-row['stamp_ns'])*1e-9
                        if not 0. <= delta <= a.max_joint_extrapolation:
                            break
                        selected[g] = dict(row, position={n: v + delta*row['velocity'][n]
                                                           for n, v in row['position'].items()})
                    if len(selected) == 4:
                        imu, x, z, vz, chosen = candidate, cx, cz, cvz, selected
                        break
                if chosen is None:
                    counters['skipped_skew'] += 1
                    continue
            age = time.time() - imu['recv']
            timing['age_max'] = max(timing['age_max'], age)
            if age > a.max_age:
                counters['skipped_age'] += 1
                continue
            try:
                if moving_mount is not None:
                    groups = chosen
                else:
                    groups = node.groups
                position_by_name, velocity_by_name = {}, {}
                for group in groups.values():
                    position_by_name.update(group['position'])
                    velocity_by_name.update(group['velocity'])
                if any(n not in position_by_name for n in names):
                    counters['skipped_joints_missing'] += 1
                    continue
                q = [position_by_name[n] for n in names]
                qd = [velocity_by_name[n] for n in names]
            except KeyError:
                counters['skipped_age'] += 1
                continue
            height = 0.
            if fk is not None:
                data = fk['data']
                data.qpos[:] = 0.
                data.qpos[:7] = np.r_[[0., 0., 0.],
                                      (Rotation.from_matrix(mount) * rot_of(imu['q'])).as_quat()[[3, 0, 1, 2]]]
                data.qpos[fk['qpos_addr']] = q
                mujoco.mj_kinematics(fk['model'], data)
                height = float(-min(data.xpos[b][2] for b in fk['feet'])) + a.sole_offset
            gravity = (rot_of(imu['q']).as_matrix().T @ np.array([0., 0., -1.]))
            group_stamps = [groups[g]['stamp_ns'] for g in ('leg', 'waist', 'head', 'arm')]
            group_recv = [groups[g]['recv'] for g in ('leg', 'waist', 'head', 'arm')]
            # The observation is complete only once every stream has been acquired, so
            # the packet timestamp is the newest of the five. Joints must never appear
            # to be acquired after the state they belong to.
            state_stamp_ns = (int(imu['stamp_ns']) if moving_mount is not None else
                              max([int(imu['stamp_ns']), *[int(t) for t in group_stamps]]))
            # The consumer rejects a set of acquisitions spread over more than 30 ms.
            # Skipping here keeps the publish slot for a coherent observation.
            if state_stamp_ns - min([int(t) for t in group_stamps]) > 25_000_000:
                counters['skipped_skew'] += 1
                continue
            packet = {
                'schema': 2, 'joint_names': list(names), 'session_id': session,
                'motor_output_enabled': False,
                'position': [float(x[0]), float(x[1]), float(height)],
                'quat_wxyz': [float(v) for v in (Rotation.from_matrix(mount) *
                                                 rot_of(imu['q'])).as_quat()[[3, 0, 1, 2]]],
                'angular_velocity_body': [float(v) for v in imu['w']],
                'gravity_body': [float(v) for v in gravity],
                'joint_position': [float(v) for v in q],
                'joint_velocity': [float(v) for v in qd],
                'joint_domains': [int(groups[g]['domain']) for g in ('leg', 'waist', 'head', 'arm')],
                'joint_stamps': [t * 1e-9 for t in group_stamps],
                'joint_stamps_ns': [int(t) for t in group_stamps],
                'joint_received_wall': group_recv,
                'state_stamp': state_stamp_ns * 1e-9,
                'state_stamp_ns': int(state_stamp_ns),
                'published_wall': time.time(),
            }
            if moving_mount is not None:
                waist_q = [position_by_name[n] for n in waist_names]
                waist_v = [velocity_by_name[n] for n in waist_names]
                ppi, rpi, wm, vm = moving_mount.mount_with_velocity(waist_q, waist_v)
                rwi = mount @ rot_of(imu['q']).as_matrix()
                pwi = np.array([x[0], x[1], z])
                pwp, rwp = pelvis_from_imu(pwi, rwi, ppi, rpi)
                if state['origin_offset'] is None:
                    state['origin_offset'] = -pwp.copy()
                omega = pelvis_angular_velocity(imu['w'], rpi, wm)
                velocity = pelvis_velocity_from_imu([x[2], x[3], vz], rwp, ppi, omega, vm)
                packet['position'] = (pwp + state['origin_offset']).tolist()
                packet['quat_wxyz'] = Rotation.from_matrix(rwp).as_quat()[[3, 0, 1, 2]].tolist()
                packet['angular_velocity_body'] = omega.tolist()
                packet['gravity_body'] = (rwp.T @ [0., 0., -1.]).tolist()
                packet['linear_velocity_world'] = velocity.tolist()
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
            state['last_published_stamp'] = state_stamp_ns
            counters['packets'] += 1
            timing['publish_ms_max'] = max(timing['publish_ms_max'], (time.monotonic() - publish_start) * 1e3)
            frames.write(json.dumps({'t': now - started, 'position': packet['position'],
                                     'stamp_ns': state_stamp_ns, 'quat_wxyz': packet['quat_wxyz'],
                                     'angular_velocity_body': packet['angular_velocity_body'],
                                     'velocity': packet.get('linear_velocity_world', [float(x[2]), float(x[3])]),
                                     'bias': [float(x[4]), float(x[5])],
                                     'age': age}, allow_nan=False) + '\n')
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except rclpy_implementation.RCLError:
        if rclpy.ok():
            raise
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
                   'limitations': ('Experimental moving-mount position is relative to the first output; '
                                   'vertical bias is not estimated or corrected. The stationary detector '
                                   'does not prove absolute rest. No output is validated for control.'
                                   if moving_mount is not None else
                                   'Legacy output uses IMU pose as pelvis pose; height uses the lowest ankle '
                                   'and a declared sole offset. No output is validated for control.')}
        (a.out / 'summary.json').write_text(json.dumps(summary, allow_nan=False) + '\n')
        print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
