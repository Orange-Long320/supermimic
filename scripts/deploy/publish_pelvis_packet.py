"""Publish diagnostic pelvis/joint state packets from the 500 Hz Kalman estimator.

Diagnostic only: callback-rate strapdown and optional IMU-only horizontal
zero-velocity updates. The optional moving-mount output is
experimental and requires explicitly supplied geometry. It is not deployment
validated. Its three-dimensional position is relative to the first output;
The three-axis body-frame bias is calibrated only under an explicit rest and
trusted-attitude assumption. Vertical ZUPT is not implemented.

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
    p.add_argument('--init-bias-seconds', type=float, default=0.)
    p.add_argument('--assume-initial-rest', action='store_true',
                   help='declare the calibration window stationary with trusted attitude')
    p.add_argument('--allow-unobserved-initial-velocity', action='store_true',
                   help='diagnostic only: explicitly assume zero velocity without observing rest')
    p.add_argument('--integration-method', choices=('right', 'trapezoid'), default='right')
    p.add_argument('--sim-only', action='store_true', help='declare isolated simulation')
    p.add_argument('--sim-support-xml', type=Path, help='experimental X2 simulation collision-sphere support')
    p.add_argument('--sim-flat-ground-height', action='store_true',
                   help='experimental flat-ground height constraint when support is accepted')
    p.add_argument('--imu-only-zupt', action='store_true',
                   help='experimental horizontal ZUPT; IMU thresholds do not prove rest')
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
    if a.sim_flat_ground_height and not a.sim_support_xml:
        raise SystemExit('Flat-ground height requires sim-only support')
    if a.sim_support_xml and (not a.sim_only or a.mount_urdf is None or
            a.imu_topic != '/aima/hal/imu/torso/state' or
            not np.allclose(a.mount_xyz, [.0239465, -.000228682, .04171]) or
            not np.allclose(a.mount_rpy, [0., 0., 0.])):
        raise SystemExit('Sim support requires --sim-only and the explicit X2 pelvis IMU mount')
    sim_domain = os.environ.get('ROS_DOMAIN_ID', '')
    if a.sim_support_xml and (os.environ.get('ROS_LOCALHOST_ONLY') != '1' or
            not sim_domain.isdigit() or not 128 <= int(sim_domain) <= 232 or
            not os.environ.get('SM_SIM_STATE_PORT')):
        raise SystemExit('Sim support requires localhost-only ROS, a domain, and an isolated simulator state port')
    if a.sim_support_xml and (a.init_bias_seconds > 0. or a.imu_only_zupt):
        raise SystemExit('Sim support cannot be combined with rest calibration or IMU-only ZUPT')
    if a.auto_mount and a.init_bias_seconds <= 0.:
        raise SystemExit('--auto-mount requires a declared-rest calibration window')
    if a.init_bias_seconds > 0. and not a.assume_initial_rest:
        raise SystemExit('Calibration requires --assume-initial-rest; motion tests cannot prove rest')
    if a.init_bias_seconds <= 0. and not a.allow_unobserved_initial_velocity and not a.sim_support_xml:
        raise SystemExit('Initial velocity is unobserved; provide a declared-rest calibration or use the diagnostic override')
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
    from supermimic.deploy.async_diagnostic_log import AsyncDiagnosticLog
    from supermimic.deploy.imu_calibration import check_static_window, body_bias_at_declared_rest
    from supermimic.deploy.sim_foot_support import SimFootSupport
    from supermimic.deploy.policy_packet import validate_policy_packet
    from supermimic.deploy.pelvis_state import (UrdfPelvisMount, pelvis_from_imu,
        pelvis_angular_velocity, pelvis_velocity_from_imu)

    names = json.loads(a.profile.read_text())['joint_names']
    waist_names = ('waist_yaw_joint', 'waist_pitch_joint', 'waist_roll_joint')
    moving_mount = (UrdfPelvisMount(a.mount_urdf, waist_names, a.mount_xyz,
                    Rotation.from_euler('xyz', a.mount_rpy).as_matrix()) if a.mount_urdf else None)
    support = SimFootSupport(a.sim_support_xml) if a.sim_support_xml else None
    if support is not None:
        ppi, rpi, _ = moving_mount.mount([0., 0., 0.])
        if not np.allclose(ppi, a.mount_xyz, atol=1e-9) or not np.allclose(rpi, np.eye(3), atol=1e-9):
            raise SystemExit('Sim support requires the fixed pelvis mount')
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
    state = {'last': None, 'last_world_accel': None, 'still_since': None, 'integrate': False, 'calibration': None,
             'z': 0., 'vz': 0., 'origin_offset': None, 'bias_body': np.zeros(3),
             'last_published_stamp': 0, 'support_initialized': support is None,
             'support_last_try': None, 'support_last_accept': None, 'support_height_origin': None,
             'x': np.zeros(4), 'P': np.diag([.01, .01, .1, .1])}
    H = np.zeros((2, 4))
    H[0, 2] = H[1, 3] = 1.
    Rm = np.eye(2) * (a.velocity_sigma ** 2)
    Q = np.diag([1e-8, 1e-8, 4e-5, 4e-5])
    identity = np.eye(4)
    counters = {'imu': 0, 'packets': 0, 'invalid': 0, 'skipped_age': 0, 'skipped_skew': 0,
                'skipped_joints_missing': 0, 'zupt': 0, 'support_accepted': 0,
                'support_hold': 0, 'support_disagree': 0, 'support_missing': 0,
                'support_stale': 0, 'support_waiting': 0, 'support_output_stale': 0,
                'support_queue_timeout': 0, 'support_queue_overflow': 0}
    timing = {'age_max': 0., 'publish_ms_max': 0.}
    diagnostics = {'imu_callback_ms': [], 'joint_callback_ms': [], 'publish_work_ms': [],
                   'imu_header_gaps_ms': [], 'imu_receive_age_ms': [], 'support_joint_lag_ms': [],
                   'support_innovation_m_s': [], 'support_queue_wait_ms': [], 'support_height_correction_m': []}
    last_imu_stamp = None

    class Producer(Node):
        def __init__(self):
            super().__init__('supermimic_imu_state_producer')
            qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE)
            self.imu = None
            self.imu_history = deque(maxlen=128)
            self.pending_imu = deque()
            self.groups = {}
            self.history = {g: deque(maxlen=128) for g in ('leg', 'waist', 'head', 'arm')}
            self.pub = self.create_publisher(String, a.topic, qos)
            self.subs = [self.create_subscription(Imu, a.imu_topic, self.on_imu, qos)]
            self.subs += [self.create_subscription(JointStateArray, f'/aima/hal/joint/{g}/state',
                                                   lambda msg, group=g: self.on_joint(group, msg), qos)
                          for g in ('leg', 'waist', 'head', 'arm')]

        def on_imu(self, msg):
            nonlocal last_imu_stamp
            callback_start = time.perf_counter()
            q, w, acc = msg.orientation, msg.angular_velocity, msg.linear_acceleration
            stamp_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            self.imu = {'recv': time.time(), 'mono': time.monotonic(), 'stamp_ns': stamp_ns,
                        'q': [q.w, q.x, q.y, q.z], 'w': [w.x, w.y, w.z],
                        'a': [acc.x, acc.y, acc.z], 'frame': msg.header.frame_id}
            counters['imu'] += 1
            diagnostics['imu_receive_age_ms'].append((self.imu['recv'] - stamp_ns * 1e-9) * 1e3)
            if last_imu_stamp is not None:
                diagnostics['imu_header_gaps_ms'].append((stamp_ns - last_imu_stamp) * 1e-6)
            last_imu_stamp = stamp_ns
            if state['calibration'] is not None:
                state['calibration'].append(self.imu)
            if state['integrate']:
                if support is None:
                    self.process_imu(self.imu)
                else:
                    if self.pending_imu and time.time() - self.pending_imu[0]['recv'] > a.max_age:
                        counters['support_queue_timeout'] += 1
                        raise ValueError('IMU support synchronization timed out')
                    if len(self.pending_imu) >= 128:
                        counters['support_queue_overflow'] += 1
                        raise ValueError('IMU support synchronization queue overflow')
                    self.pending_imu.append(self.imu)
                    self.drain_pending()
            diagnostics['imu_callback_ms'].append((time.perf_counter() - callback_start) * 1e3)

        def process_imu(self, sample):
            propagate(sample)
            if support is not None:
                initialized = state['support_initialized']
                apply_support(sample, self.history['leg'])
                if not initialized and state['support_initialized']:
                    self.imu_history.clear()
            if moving_mount is not None:
                self.imu_history.append((sample, state['x'].copy(), state['z'], state['vz'],
                                         state['support_last_accept']))

        def drain_pending(self):
            history = self.history['leg']
            while (self.pending_imu and history and
                   history[-1]['stamp_ns'] >= self.pending_imu[0]['stamp_ns']):
                sample = self.pending_imu.popleft()
                wait = time.time() - sample['recv']
                diagnostics['support_queue_wait_ms'].append(wait*1000.)
                if wait > a.max_age:
                    counters['support_queue_timeout'] += 1
                    raise ValueError('IMU support synchronization timed out')
                self.process_imu(sample)

        def on_joint(self, group, msg):
            callback_start = time.perf_counter()
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
            if support is not None and group == 'leg' and state['integrate']:
                self.drain_pending()
            diagnostics['joint_callback_ms'].append((time.perf_counter() - callback_start) * 1e3)

    def propagate(sample):
        x, P = state['x'], state['P']
        now = sample['stamp_ns'] * 1e-9
        acc_body = np.array(sample['a'])
        gyro = np.array(sample['w'])
        world = mount @ rot_of(sample['q']).apply(acc_body - state['bias_body'])
        world -= np.array([0., 0., 9.81])
        if state['last'] is None:
            state['last'] = now
            state['last_world_accel'] = world
            return
        dt = now - state['last']
        if not 0. < dt <= .03:
            raise ValueError('IMU acquisition timestamp rollback or gap')
        state['last'] = now
        previous = state['last_world_accel']
        state['last_world_accel'] = world
        average = .5 * (previous + world) if a.integration_method == 'trapezoid' else world
        position_accel = ((2. * previous + world) / 6. if a.integration_method == 'trapezoid'
                          else .5 * world)
        state['z'] += state['vz'] * dt + position_accel[2] * dt * dt
        state['vz'] += average[2] * dt
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        x = F @ x
        x[:2] += position_accel[:2] * dt ** 2
        x[2:4] += average[:2] * dt
        P = F @ P @ F.T + Q * (dt / .02)
        still = (abs(np.linalg.norm(acc_body) - 9.81) <= a.still_accel_tol
                 and np.linalg.norm(gyro) <= a.still_gyro_tol)
        state['still_since'] = (state['still_since'] if state['still_since'] is not None else now) if still else None
        if a.imu_only_zupt and still and now - state['still_since'] >= a.still_hold:
            innovation = np.zeros(2) - H @ x
            S = H @ P @ H.T + Rm
            K = np.linalg.solve(S, (P @ H.T).T).T
            x = x + K @ innovation
            P = (identity - K @ H) @ P @ (identity - K @ H).T + K @ Rm @ K.T
            counters['zupt'] += 1
        state['x'], state['P'] = x, P

    def apply_support(sample, history):
        now = sample['stamp_ns'] * 1e-9
        if state['support_last_try'] is not None and now-state['support_last_try'] < .02:
            return
        state['support_last_try'] = now
        row = next((j for j in reversed(history) if j['stamp_ns'] <= sample['stamp_ns']), None)
        if row is None:
            counters['support_missing'] += 1
            diagnostics_log.emit('support-events', {'stamp_ns':sample['stamp_ns'],'reason':'missing'})
            return
        lag = (sample['stamp_ns']-row['stamp_ns'])*1e-9
        diagnostics['support_joint_lag_ms'].append(lag*1000.)
        if lag > a.max_joint_extrapolation:
            counters['support_stale'] += 1
            diagnostics_log.emit('support-events', {'stamp_ns':sample['stamp_ns'],'reason':'stale','lag_s':lag})
            return
        position = {n: q + lag*row['velocity'][n] for n, q in row['position'].items()}
        rwp = mount @ rot_of(sample['q']).as_matrix()
        pelvis_v, reason = support.observe(now, rwp, sample['w'], position, row['velocity'])
        if pelvis_v is None:
            counters['support_' + reason] += 1
            diagnostics_log.emit('support-events', {'stamp_ns':sample['stamp_ns'],'reason':reason,'lag_s':lag})
            return
        site_v = pelvis_v + rwp @ np.cross(sample['w'], np.asarray(a.mount_xyz))
        height_site = support.accepted_relative_height + (rwp @ np.asarray(a.mount_xyz))[2]
        if state['support_initialized']:
            old = np.r_[state['x'][2:4], state['vz']]
            diagnostics['support_innovation_m_s'].append(float(np.linalg.norm(site_v-old)))
        else:
            # Discard propagation before the first observed support hypothesis.
            state['x'][:2] = 0.
            state['z'] = 0.
            state['support_initialized'] = True
            state['support_height_origin'] = height_site
        if a.sim_flat_ground_height:
            corrected_z = float(height_site - state['support_height_origin'])
            diagnostics['support_height_correction_m'].append(corrected_z - state['z'])
            state['z'] = corrected_z
        diagnostics_log.emit('support-events', {'stamp_ns':sample['stamp_ns'],'reason':'accepted',
            'lag_s':lag,'height_site_m':float(height_site),'site_velocity_m_s':site_v.tolist()})
        state['x'][2:4] = site_v[:2]
        state['vz'] = float(site_v[2])
        state['support_last_accept'] = now
        counters['support_accepted'] += 1

    rclpy.init()
    node = Producer()

    # A threshold-passing window does not prove rest: the caller must declare it.
    # The estimated bias is conditional on that declaration and trusted attitude.
    if a.init_bias_seconds > 0.:
        state['calibration'] = []
        end = time.monotonic() + a.init_bias_seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=.01)
        samples = state['calibration']
        state['calibration'] = None
        stamps = np.array([s['stamp_ns'] for s in samples], dtype=np.int64)
        acc = np.array([s['a'] for s in samples])
        gyro = np.array([s['w'] for s in samples])
        try:
            check_static_window(stamps, acc, gyro, a.init_bias_seconds,
                                a.still_accel_tol, a.still_gyro_tol)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        window = np.array([mount @ rot_of(s['q']).apply(np.array(s['a'])) for s in samples])
        if a.auto_mount:
            mean = window.mean(axis=0)
            correction = Rotation.align_vectors([[0., 0., 1.]], [mean])[0].as_matrix()
            mount = correction @ mount
            fitted = Rotation.from_matrix(mount).as_euler('xyz', degrees=True)
            print('fitted world correction x=%.2f y=%.2f z=%.2f deg residual %.4f'
                  % (fitted[0], fitted[1], fitted[2],
                     float(np.linalg.norm(correction @ mean - np.array([0., 0., np.linalg.norm(mean)])))),
                  flush=True)
            window = np.array([correction @ v for v in window])
        # Bias is observable here only conditional on rest, known gravity, and
        # trusted attitude. Express it in the sensor frame so it rotates with it.
        sensor_to_world = [mount @ rot_of(s['q']).as_matrix() for s in samples]
        state['bias_body'] = body_bias_at_declared_rest(acc, sensor_to_world)
        print('initial body bias (conditional on rest and attitude) %s' %
              np.round(state['bias_body'], 5), flush=True)

    diagnostics_log = AsyncDiagnosticLog(a.out)
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
            if not state['support_initialized']:
                counters['support_waiting'] += 1
                continue
            if node.imu is None or len(node.groups) < 4:
                counters['skipped_age'] += 1
                continue
            work_start = time.perf_counter()
            imu = node.imu
            x, z, vz = state['x'], state['z'], state['vz']
            chosen = None
            selected_support_stamp = state['support_last_accept']
            if moving_mount is not None:
                for candidate, cx, cz, cvz, support_stamp in reversed(node.imu_history):
                    target = candidate['stamp_ns']
                    if target <= state['last_published_stamp'] or time.time()-candidate['recv'] > a.max_age:
                        break
                    selected = {}
                    for g, history in node.history.items():
                        row = next((row for row in reversed(history) if row['stamp_ns'] <= target), None)
                        if row is None:
                            break
                        delta = (target-row['stamp_ns'])*1e-9
                        if not 0. <= delta <= a.max_joint_extrapolation:
                            break
                        selected[g] = dict(row, position={n: v + delta*row['velocity'][n]
                                                           for n, v in row['position'].items()})
                    if len(selected) == 4:
                        imu, x, z, vz, chosen, selected_support_stamp = candidate, cx, cz, cvz, selected, support_stamp
                        break
                if chosen is None:
                    counters['skipped_skew'] += 1
                    continue
            if (support is not None and (selected_support_stamp is None or
                    imu['stamp_ns']*1e-9 - selected_support_stamp > .1)):
                counters['support_output_stale'] += 1
                diagnostics_log.emit('packets', {'event': 'support_output_stale', 't': now-started,
                    'imu_stamp_ns': int(imu['stamp_ns']), 'support_stamp_ns':
                    (None if selected_support_stamp is None else int(round(selected_support_stamp*1e9)))})
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
                diagnostics_log.emit('packets', {'invalid': str(exc)})
                continue
            publish_start = time.monotonic()
            msg = String()
            msg.data = json.dumps(packet, allow_nan=False)
            node.pub.publish(msg)
            state['last_published_stamp'] = state_stamp_ns
            counters['packets'] += 1
            timing['publish_ms_max'] = max(timing['publish_ms_max'], (time.monotonic() - publish_start) * 1e3)
            diagnostics['publish_work_ms'].append((time.perf_counter() - work_start) * 1e3)
            diagnostics_log.emit('packets', {'t': now - started, 'position': packet['position'],
                                     'stamp_ns': state_stamp_ns, 'quat_wxyz': packet['quat_wxyz'],
                                     'angular_velocity_body': packet['angular_velocity_body'],
                                     'velocity': packet.get('linear_velocity_world', [float(x[2]), float(x[3])]),
                                     'bias_body': state['bias_body'].tolist(),
                                     'support_age': (None if selected_support_stamp is None else
                                         imu['stamp_ns']*1e-9-selected_support_stamp),
                                     'age': age})
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except rclpy_implementation.RCLError:
        if rclpy.ok():
            raise
    finally:
        logger_status = diagnostics_log.close()
        node.destroy_node()
        rclpy.try_shutdown()
        elapsed = time.monotonic() - started
        summary = {'seconds': elapsed, 'session_id': session, 'topic': a.topic,
                   'publish_rate_hz': counters['packets'] / elapsed if elapsed else None,
                   'counters': counters, 'diagnostic_logger': logger_status, 'timing': {k: float(v) for k, v in timing.items()},
                   'extrinsic_deg': [a.fix_deg_x, a.fix_deg_y, a.fix_deg_z],
                   'motor_output_enabled': False, 'control_ready': False,
                   'initial_velocity_assumed_zero': support is None,
                   'simulation_support': support is not None,
                   'sim_flat_ground_height': a.sim_flat_ground_height,
                   'support_observation': ('lowest sphere, flat ground and stationary contact assumed; 3D hard replacement'
                                           ' and output stops after 100 ms without a support update'
                                           if support is not None else None),
                   'support_last_observation_age_s': (None if state['support_last_accept'] is None or state['last'] is None else state['last']-state['support_last_accept']),
                   'initial_rest_declared': bool(a.init_bias_seconds > 0. and a.assume_initial_rest),
                   'imu_only_zupt': a.imu_only_zupt, 'integration_method': a.integration_method,
                   'bias_body': state['bias_body'].tolist(),
                   'reason': 'state producer only; no policy targets, no actuator path',
                   'limitations': ('Experimental moving-mount position is relative to the first output; '
                                   'Bias calibration depends on declared rest and trusted attitude. The stationary detector '
                                   'does not prove absolute rest. No output is validated for control.'
                                   if moving_mount is not None else
                                   'Legacy output uses IMU pose as pelvis pose; height uses the lowest ankle '
                                   'and a declared sole offset. No output is validated for control.')}
        summary['diagnostics'] = {key: ({'count': len(values), 'p50': float(np.percentile(values, 50)),
                                          'p95': float(np.percentile(values, 95)),
                                          'p99': float(np.percentile(values, 99)), 'max': float(max(values))}
                                         if values else {'count': 0})
                                  for key, values in diagnostics.items()}
        (a.out / 'summary.json').write_text(json.dumps(summary, allow_nan=False) + '\n')
        print(json.dumps(summary, allow_nan=False), flush=True)
        if not logger_status['success']:
            raise RuntimeError('Diagnostic logger failed; output evidence incomplete')


if __name__ == '__main__':
    main()
