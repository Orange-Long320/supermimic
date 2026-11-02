"""Read-only shadow pelvis estimator: contact-Jacobian support fusion at 50 Hz.

Deploys the configuration that scored best offline (contact-Jacobian foot
velocity, height/vertical-speed support gate, innovation-gated adaptive slip
guard) onto the onboard chain, replacing the lidar/point-cloud correction with
leg-kinematic support updates.

This process NEVER publishes. It creates no ROS publisher, no service client and
no actuator path; every status file carries control_ready=false and
motor_output_enabled=false. Bounded runtime, then it exits.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=120.)
    p.add_argument('--tick', type=float, default=.02)
    p.add_argument('--max-imu-age', type=float, default=.05)
    p.add_argument('--max-joint-age', type=float, default=.05)
    p.add_argument('--foot-dwell', type=float, default=.2)
    p.add_argument('--calibrate-seconds', type=float, default=3.,
                   help='static window used to estimate the IMU-to-base mount rotation')
    p.add_argument('--ros-domain', type=int, default=None)
    a = p.parse_args()
    if not 1. <= a.seconds <= 600.:
        raise ValueError('Bounded duration required')
    if os.environ.get('SM_MOTOR_OUTPUT') not in (None, '', '0'):
        raise SystemExit('This shadow process must never run with motor output enabled')
    return a


def main():
    a = parse()
    import mujoco
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from aimdk_msgs.msg import JointStateArray
    from sensor_msgs.msg import Imu
    from scipy.spatial.transform import Rotation

    spec = json.loads(a.profile.read_text())
    names = spec['joint_names']
    nj = len(names)
    model = mujoco.MjModel.from_xml_path(str(a.model))
    data = mujoco.MjData(model)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, s + '_ankle_roll_link')
            for s in ('left', 'right')]
    if min(*ids, *feet) < 0:
        raise ValueError('Missing model joint or foot body')
    qpos_addr = [model.jnt_qposadr[i] for i in ids]
    dof_addr = [model.jnt_dofadr[i] for i in ids]
    sole = 0.0  # contact point taken at the ankle body origin; recorded below

    a.out.mkdir(parents=True, exist_ok=False)
    frames = (a.out / 'frames.jsonl').open('x')
    status_path = a.out / 'status.json'
    counters = {'ticks': 0, 'rejected_imu_age': 0, 'rejected_joint_age': 0,
                'updates': 0, 'rejected_innovation': 0, 'reduced_confidence': 0,
                'switches': 0, 'no_candidate': 0}
    timing = {'fk_ms_max': 0., 'tick_ms_max': 0., 'imu_age_max': 0., 'joint_age_max': 0.}

    class Shadow(Node):
        def __init__(self):
            super().__init__('supermimic_shadow_contact_fusion')
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE)
            self.imu = None
            self.groups = {}
            self.subs = [self.create_subscription(Imu, '/aima/hal/imu/torso/state', self.on_imu, qos)]
            self.subs += [self.create_subscription(JointStateArray, f'/aima/hal/joint/{g}/state',
                                                   lambda msg, group=g: self.on_joint(group, msg), qos)
                          for g in ('leg', 'waist', 'head', 'arm')]

        def on_imu(self, msg):
            q, w, acc = msg.orientation, msg.angular_velocity, msg.linear_acceleration
            self.imu = {'recv': time.monotonic(),
                        'stamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                        'q_wxyz': [q.w, q.x, q.y, q.z], 'w': [w.x, w.y, w.z],
                        'a': [acc.x, acc.y, acc.z], 'frame': msg.header.frame_id}

        def on_joint(self, group, msg):
            self.groups[group] = {'recv': time.monotonic(),
                                  'position': {j.name: j.position for j in msg.joints},
                                  'velocity': {j.name: j.velocity for j in msg.joints},
                                  'error': {j.name: j.error_code for j in msg.joints}}

    rclpy.init()
    node = Shadow()

    # Static mount calibration: the published quaternion and the accelerometer do not
    # share a frame on this robot, and the residual is large enough to inject several
    # m/s^2 of gravity leakage. Estimate the rotation that maps the measured specific
    # force onto +z. A single-vector fit leaves yaw free, which does not affect gravity
    # cancellation; it is NOT a full extrinsic calibration.
    calibration = {'samples': 0, 'mount_quat_wxyz': [1., 0., 0., 0.], 'residual_m_s2': None}
    cal_acc, cal_gyro = [], []
    cal_end = time.monotonic() + a.calibrate_seconds
    while time.monotonic() < cal_end:
        rclpy.spin_once(node, timeout_sec=.01)
        if node.imu is not None:
            cal_acc.append(node.imu['a'])
            cal_gyro.append(node.imu['w'])
    cal_acc = np.array(cal_acc)
    cal_gyro = np.array(cal_gyro)
    mount = np.eye(3)
    if len(cal_acc) >= 100:
        magnitude = np.linalg.norm(cal_acc.mean(axis=0))
        rate = np.linalg.norm(cal_gyro.mean(axis=0))
        if not (9. <= magnitude <= 11.):
            raise SystemExit('Static calibration rejected: specific force %.3f m/s^2' % magnitude)
        if rate > .05:
            raise SystemExit('Static calibration rejected: gyro %.3f rad/s, robot is moving' % rate)
        direction = cal_acc.mean(axis=0) / magnitude
        mount = Rotation.align_vectors([[0., 0., 1.]], [direction])[0].as_matrix()
        calibration['samples'] = int(len(cal_acc))
        calibration['mount_quat_wxyz'] = [float(v) for v in Rotation.from_matrix(mount).as_quat()[[3, 0, 1, 2]]]
        calibration['residual_m_s2'] = float(np.linalg.norm(mount @ cal_acc.mean(axis=0) - np.array([0., 0., magnitude])))
        calibration['specific_force_norm'] = float(magnitude)
        calibration['gyro_norm'] = float(rate)
    else:
        raise SystemExit('Static calibration failed: too few IMU samples')
    # Filter state: horizontal position, velocity, world-frame acceleration bias.
    x = np.zeros(6)
    P = np.diag([.01, .01, .1, .1, .01, .01])
    H = np.zeros((2, 6))
    H[0, 2] = H[1, 3] = 1.
    R = np.eye(2) * (.10 ** 2)
    Q = np.diag([1e-7, 1e-7, 4e-5, 4e-5, 1e-7, 1e-7])
    identity = np.eye(6)
    foot_v = np.zeros((2, 3))
    last_foot = None
    last_switch_time = None
    started = time.monotonic()
    last_tick = None
    try:
        while time.monotonic() - started < a.seconds:
            rclpy.spin_once(node, timeout_sec=.002)
            now = time.monotonic()
            if last_tick is not None and now - last_tick < a.tick:
                continue
            if node.imu is None or len(node.groups) < 4:
                continue
            tick_start = time.monotonic()
            counters['ticks'] += 1
            imu_age = tick_start - node.imu['recv']
            joint_age = max(tick_start - g['recv'] for g in node.groups.values())
            timing['imu_age_max'] = max(timing['imu_age_max'], imu_age)
            timing['joint_age_max'] = max(timing['joint_age_max'], joint_age)
            if imu_age > a.max_imu_age:
                counters['rejected_imu_age'] += 1
                last_tick = tick_start
                continue
            if joint_age > a.max_joint_age:
                counters['rejected_joint_age'] += 1
                last_tick = tick_start
                continue
            position_by_name, velocity_by_name = {}, {}
            for group in node.groups.values():
                position_by_name.update(group['position'])
                velocity_by_name.update(group['velocity'])
            if any(n not in position_by_name or n not in velocity_by_name for n in names):
                counters['rejected_joint_age'] += 1
                last_tick = tick_start
                continue
            q = np.array([position_by_name[n] for n in names])
            qd = np.array([velocity_by_name[n] for n in names])
            if not (np.isfinite(q).all() and np.isfinite(qd).all()):
                counters['rejected_joint_age'] += 1
                last_tick = tick_start
                continue

            quat = node.imu['q_wxyz']
            gyro = np.array(node.imu['w'])
            acc_body = np.array(node.imu['a'])
            # Corrected world attitude: mount maps the accelerometer frame to world,
            # so the FK base pose must carry the same correction for foot geometry.
            quat_world = (Rotation.from_matrix(mount) * Rotation.from_quat(quat)).as_quat()[[3, 0, 1, 2]]
            fk_start = time.monotonic()
            data.qpos[:] = 0.
            data.qpos[:7] = np.r_[[0., 0., 0.], quat_world]
            data.qpos[qpos_addr] = q
            data.qvel[:] = 0.
            data.qvel[3:6] = gyro
            data.qvel[dof_addr] = qd
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            rel = np.zeros((2, 3))
            raw = np.zeros((2, 3))
            for foot, body in enumerate(feet):
                point = data.xpos[body].copy()
                point[2] -= sole
                rel[foot] = point
                jac = np.zeros((3, model.nv))
                mujoco.mj_jac(model, data, jac, None, point, body)
                raw[foot] = jac @ data.qvel
            fk_ms = (time.monotonic() - fk_start) * 1e3
            timing['fk_ms_max'] = max(timing['fk_ms_max'], fk_ms)

            dt = a.tick if last_tick is None else tick_start - last_tick
            dt = float(min(max(dt, .005), .1))
            alpha = math.exp(-dt / (.02 / math.log(2.)))
            for foot in range(2):
                foot_v[foot] = (1. - alpha) * raw[foot] + alpha * foot_v[foot]

            acc_world = mount @ acc_body - np.array([0., 0., 9.81])
            F = np.eye(6)
            F[0, 2] = F[1, 3] = dt
            F[0, 4] = F[1, 5] = -.5 * dt ** 2
            F[2, 4] = F[3, 5] = -dt
            B = np.zeros((6, 2))
            B[0, 0] = B[1, 1] = .5 * dt ** 2
            B[2, 0] = B[3, 1] = dt
            u = acc_world[:2]
            x = F @ x + B @ u
            P = F @ P @ F.T + Q * (dt / .02)

            rel_world = rel
            low = float(np.min(rel_world[:, 2]))
            candidates = []
            for foot in range(2):
                measurement = -foot_v[foot, :2]
                if (rel_world[foot, 2] - low <= .015 and abs(foot_v[foot, 2]) <= .12
                        and np.linalg.norm(measurement) <= 1.5):
                    candidates.append((float(np.linalg.norm(measurement - H @ x)), foot, measurement))
            accepted = False
            if candidates:
                diff, foot, z = min(candidates, key=lambda c: c[0])
                too_soon = (last_foot is not None and foot != last_foot
                            and last_switch_time is not None
                            and tick_start - last_switch_time < a.foot_dwell)
                if diff < .6 and not too_soon:
                    innovation = z - H @ x
                    effective_r = R
                    S0 = H @ P @ H.T + R
                    nis = float(innovation @ np.linalg.solve(S0, innovation))
                    if nis > 9.21:
                        counters['rejected_innovation'] += 1
                    else:
                        disagreement = (float(np.linalg.norm(candidates[0][2] - candidates[1][2]))
                                        if len(candidates) == 2 else 0.)
                        scale = min(16., max(1., (diff / .1) ** 2, (disagreement / .15) ** 2))
                        if scale > 1.01:
                            counters['reduced_confidence'] += 1
                        effective_r = R * scale
                        S = H @ P @ H.T + effective_r
                        K = np.linalg.solve(S, (P @ H.T).T).T
                        x = x + K @ innovation
                        P = (identity - K @ H) @ P @ (identity - K @ H).T + K @ effective_r @ K.T
                        counters['updates'] += 1
                        accepted = True
                        if last_foot is not None and foot != last_foot:
                            counters['switches'] += 1
                            last_switch_time = tick_start
                        last_foot = foot
            else:
                counters['no_candidate'] += 1

            frames.write(json.dumps({
                't': tick_start - started,
                'tick_ms': (time.monotonic() - tick_start) * 1e3,
                'fk_ms': fk_ms,
                'imu_age': imu_age,
                'joint_age': joint_age,
                'imu_frame': node.imu['frame'],
                'position': [float(x[0]), float(x[1]), float(-low)],
                'velocity': [float(x[2]), float(x[3])],
                'bias': [float(x[4]), float(x[5])],
                'foot_z': rel[:, 2].tolist(),
                'foot_vz': foot_v[:, 2].tolist(),
                'support': int(last_foot) if accepted else -1,
                'acc_body': acc_body.tolist(),
                'gyro': gyro.tolist(),
                'quat_wxyz': [float(v) for v in quat],
                'acc_world': acc_world.tolist(),
                'waist': [float(node.groups['waist']['position'].get(n, 0.))
                          for n in ('waist_pitch_joint', 'waist_roll_joint', 'waist_yaw_joint')],
            }, allow_nan=False) + '\n')
            timing['tick_ms_max'] = max(timing['tick_ms_max'], (time.monotonic() - tick_start) * 1e3)
            last_tick = tick_start
            if counters['ticks'] % 10 == 0:
                status_path.write_text(json.dumps({
                    'control_ready': False,
                    'motor_output_enabled': False,
                    'reason': 'shadow estimator; no policy, no actuator path, no publisher',
                    'counters': counters,
                    'timing': {k: float(v) for k, v in timing.items()},
                    'position': [float(x[0]), float(x[1]), float(-low)],
                    'velocity': [float(x[2]), float(x[3])],
                    'elapsed_s': tick_start - started,
                }, allow_nan=False) + '\n')
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        frames.flush()
        frames.close()
        node.destroy_node()
        rclpy.try_shutdown()
        summary = {'seconds': time.monotonic() - started, 'counters': counters,
                   'timing': {k: float(v) for k, v in timing.items()},
                   'control_ready': False, 'motor_output_enabled': False,
                   'algorithm': ('contact-Jacobian foot velocity, height/vertical-speed support gate, '
                                 'innovation-gated adaptive slip guard, 50 Hz tick'),
                   'calibration': calibration,
                   'limitations': ('Shadow estimate only. The mount rotation is fitted from a single static '
                                   'gravity vector, so yaw is unconstrained and the fit is only valid near '
                                   'the calibration waist pose; the contact point is the ankle body origin; '
                                   'no measured contact force; no ground truth on hardware.')}
        status_path.write_text(json.dumps(summary, allow_nan=False) + '\n')
        print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
