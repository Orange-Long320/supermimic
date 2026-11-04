"""Sensor-only diagnostic IMU propagation; no simulator pose/velocity inputs.

Includes the hardware producer's four-state Kalman/ZUPT equations for diagnosis.
Unsupported online ZUPT/gravity guesses are off by default; attitude propagates
from a measured static initialization. This is not real IMU firmware fusion.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from collections import deque


def fit_static_initialization(samples, *, declared_stationary=False, seconds=3.):
    """Validate a complete stationary sensor window, then fit tilt and gyro bias.

    The declaration must come from the operator/fixture, not the IMU threshold.
    Constant-velocity translation cannot be detected from a static IMU alone.
    Accelerometer bias and tilt cannot both be identified from one orientation.
    """
    if not declared_stationary:
        raise ValueError('An independently declared stationary window is required')
    if seconds < 1. or len(samples) < 100:
        raise ValueError('Insufficient static calibration samples')
    t = np.array([s['time_s'] for s in samples], dtype=float)
    gyro = np.array([s['gyro'] for s in samples], dtype=float)
    acc = np.array([s['acceleration'] for s in samples], dtype=float)
    q = np.array([s['joint_pos'] for s in samples], dtype=float)
    qd = np.array([s['joint_vel'] for s in samples], dtype=float)
    if (gyro.shape != (len(t), 3) or acc.shape != gyro.shape or q.ndim != 2 or
            qd.shape != q.shape or q.shape[0] != len(t) or
            not all(np.isfinite(x).all() for x in (t, gyro, acc, q, qd)) or
            t[-1]-t[0] < seconds-1e-6 or
            np.any(np.diff(t) <= 0) or np.max(np.diff(t)) > .03):
        raise ValueError('Invalid, incomplete or discontinuous calibration window')
    if (np.max(np.linalg.norm(gyro, axis=1)) > .08 or np.max(gyro.std(axis=0)) > .005 or
            np.max(np.abs(np.linalg.norm(acc, axis=1)-9.81)) > .5 or
            np.max(acc.std(axis=0)) > .05 or np.max(np.abs(qd)) > .03 or
            np.max(np.ptp(q, axis=0)) > .015):
        raise ValueError('Calibration window contains sensor or encoder motion')
    mean = acc.mean(axis=0)
    roll = np.arctan2(mean[1], mean[2])
    pitch = np.arctan2(-mean[0], np.hypot(mean[1], mean[2]))
    rotation = Rotation.from_euler('xyz', [roll, pitch, 0.])
    return {'gyro_bias': gyro.mean(axis=0).tolist(),
            'quat_wxyz': rotation.as_quat()[[3, 0, 1, 2]].tolist(),
            'acceleration_mean': mean.tolist(), 'gyro_std': gyro.std(axis=0).tolist(),
            'acceleration_std': acc.std(axis=0).tolist(),
            'duration_s': float(t[-1]-t[0]), 'samples': len(t),
            'max_encoder_speed': float(np.max(np.abs(qd))),
            'max_encoder_excursion': float(np.max(np.ptp(q, axis=0))),
            'declaration': 'independent stationary fixture/operator',
            'yaw_observable': False, 'accelerometer_bias_identified': False}


class EncoderQuietWindow:
    """Causal encoder veto for IMU stillness; never proof of no slip/translation."""
    def __init__(self):
        self.rows = deque()

    def update(self, stamp, joints, velocity):
        q, v = np.asarray(joints, dtype=float), np.asarray(velocity, dtype=float)
        if q.ndim != 1 or v.shape != q.shape or not np.isfinite(np.r_[stamp, q, v]).all():
            raise ValueError('Invalid encoder stillness sample')
        self.rows.append((stamp, q.copy(), float(np.max(np.abs(v)))))
        while self.rows and self.rows[0][0] < stamp-.2-1e-9:
            self.rows.popleft()
        return (stamp-self.rows[0][0] >= .2-1e-6 and max(r[2] for r in self.rows) <= .03
                and np.max(np.ptp(np.stack([r[1] for r in self.rows]), axis=0)) <= .015)


class ImuDeadReckoning:
    def __init__(self, *, zupt=False, gravity_leveling=False, require_initialization=True,
                 interval_average=False, point_trapezoid=False):
        # Declared zero-velocity prior, yaw zero is an arbitrary local heading.
        # Neither prior is obtained from simulator state.
        self.rotation = Rotation.identity()
        self.x = np.zeros(4)
        self.P = np.diag([.01, .01, .1, .1])
        self.H = np.zeros((2, 4)); self.H[0, 2] = self.H[1, 3] = 1.
        self.Q = np.diag([1e-8, 1e-8, 4e-5, 4e-5])
        self.R = np.eye(2)*.02**2
        self.zupt = zupt; self.gravity_leveling = gravity_leveling
        self.stamp = None; self.still_since = None
        self.gyro = np.zeros(3); self.previous_gyro = None
        self.samples = 0; self.zupt_updates = 0
        self.gyro_bias = np.zeros(3)
        self.initialized = not require_initialization
        self.initial_rotation = self.rotation
        self.last_world_acceleration = np.zeros(3)
        self.last_zupt = False
        self.interval_average = interval_average
        self.point_trapezoid = point_trapezoid
        if interval_average and point_trapezoid:
            raise ValueError("Point integration cannot be combined with interval means")

    def initialize(self, calibration):
        if self.samples:
            raise ValueError('Calibration must precede propagation; no in-run state resets')
        bias = np.asarray(calibration['gyro_bias'], dtype=float)
        quat = np.asarray(calibration['quat_wxyz'], dtype=float)
        if bias.shape != (3,) or quat.shape != (4,) or not np.isfinite(np.r_[bias, quat]).all():
            raise ValueError('Invalid static calibration')
        self.gyro_bias = bias.copy()
        self.rotation = Rotation.from_quat(quat[[1, 2, 3, 0]])
        self.initial_rotation = self.rotation
        self.initialized = True

    def update(self, stamp, gyro, acceleration, *, gravity_static_evidence=True, zupt_static_evidence=True):
        if not self.initialized:
            raise ValueError('Static initialization has not passed')
        gyro, acc = np.asarray(gyro, dtype=float), np.asarray(acceleration, dtype=float)
        if (gyro.shape != (3,) or acc.shape != (3,) or
                not np.isfinite(np.r_[stamp, gyro, acc]).all()):
            raise ValueError('Invalid timestamped IMU sample')
        gyro = gyro-self.gyro_bias
        if self.stamp is None:
            self.stamp = float(stamp); self.previous_gyro = gyro.copy(); self.gyro = gyro.copy()
            self.last_world_acceleration = self.rotation.apply(acc)-[0., 0., 9.81]
            self.samples += 1
            return
        dt = float(stamp-self.stamp)
        if not 0. < dt <= .03:
            raise ValueError('IMU duplicate/rollback or gap above 30 ms')
        increment = gyro if self.interval_average else .5*(self.previous_gyro+gyro)
        self.rotation *= Rotation.from_rotvec(increment*dt)
        still = abs(np.linalg.norm(acc)-9.81) <= .5 and np.linalg.norm(gyro) <= .08
        self.still_since = (self.still_since if self.still_since is not None else stamp) if still else None
        settled = still and stamp-self.still_since >= .2
        if self.gravity_leveling and settled and gravity_static_evidence:
            # Accelerometer gravity is only trusted under the same stillness
            # hypothesis as ZUPT. These thresholds cannot establish real rest.
            up = self.rotation.apply(acc/np.linalg.norm(acc))
            correction = np.cross(up, [0., 0., 1.])
            self.rotation = Rotation.from_rotvec((1.-np.exp(-dt/.5))*correction)*self.rotation
        # Interval means belong to the just-completed interval, rather than its
        # endpoint. Rotate their specific force using the midpoint attitude.
        acceleration_rotation = (self.rotation*Rotation.from_rotvec(-.5*gyro*dt)
                                 if self.interval_average else self.rotation)
        world = acceleration_rotation.apply(acc)-[0., 0., 9.81]
        previous_world = self.last_world_acceleration.copy()
        self.last_world_acceleration = world.copy()
        self.last_zupt = False
        F = np.eye(4); F[0, 2] = F[1, 3] = dt
        self.x = F@self.x
        if self.point_trapezoid:
            # Exact for linearly varying world acceleration between point samples.
            self.x[:2] += (2*previous_world[:2]+world[:2])*dt**2/6
            self.x[2:] += .5*(previous_world[:2]+world[:2])*dt
        else:
            self.x[:2] += .5*world[:2]*dt**2
            self.x[2:] += world[:2]*dt
        self.P = F@self.P@F.T+self.Q*(dt/.02)
        if self.zupt and settled and zupt_static_evidence:
            K = np.linalg.solve(self.H@self.P@self.H.T+self.R, (self.P@self.H.T).T).T
            self.x -= K@(self.H@self.x)
            residual = np.eye(4)-K@self.H
            self.P = residual@self.P@residual.T+K@self.R@K.T
            self.zupt_updates += 1
            self.last_zupt = True
        self.stamp = float(stamp); self.previous_gyro = gyro.copy(); self.gyro = gyro.copy()
        self.samples += 1


class EncoderImuState:
    """Pelvis pose from IMU propagation and encoder FK on assumed flat support.

Only immutable mount geometry, a separate FK model, encoder values and raw IMU
samples are accepted. There is no simulator-data object or truth-pose argument.
"""
    def __init__(self, fk_model_path, joint_names, imu_offset, *, sole_offset=.05,
                 require_initialization=True, encoder_stationary_gate=True, interval_average=True, point_trapezoid=False):
        import mujoco
        self.mj = mujoco
        self.filter = ImuDeadReckoning(require_initialization=require_initialization,
                                       interval_average=interval_average, point_trapezoid=point_trapezoid)
        self.model = mujoco.MjModel.from_xml_path(str(fk_model_path))
        self.data = mujoco.MjData(self.model)
        ids = [self.model.joint(name).id for name in joint_names]
        self.qadr = self.model.jnt_qposadr[ids]
        self.feet = [self.model.body(side+'_ankle_roll_link').id for side in ('left', 'right')]
        self.offset = np.asarray(imu_offset, dtype=float).copy()
        if self.offset.shape != (3,) or not np.isfinite(self.offset).all():
            raise ValueError('Invalid known pelvis IMU mount')
        self.sole_offset = sole_offset
        self.encoder_stationary_gate = encoder_stationary_gate
        self.quiet = EncoderQuietWindow()

    def update(self, stamp, gyro, acceleration, joints=None, velocity=None):
        if self.encoder_stationary_gate:
            if joints is None or velocity is None:
                raise ValueError('Encoder evidence required for online stillness updates')
            quiet = self.quiet.update(stamp, joints, velocity)
        else:
            quiet = True
        self.filter.update(stamp, gyro, acceleration,
                           gravity_static_evidence=quiet, zupt_static_evidence=quiet)

    def state(self, joints):
        if not self.filter.initialized:
            raise ValueError('Static initialization has not passed')
        joints = np.asarray(joints, dtype=float)
        if joints.shape != self.qadr.shape or not np.isfinite(joints).all():
            raise ValueError('Invalid encoder sample')
        f = self.filter
        quat = f.rotation.as_quat()[[3, 0, 1, 2]]
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[:7] = np.r_[np.zeros(3), quat]
        self.data.qpos[self.qadr] = joints
        self.mj.mj_kinematics(self.model, self.data)
        height = -min(self.data.xpos[body, 2] for body in self.feet)+self.sole_offset
        # Integrate at the physical IMU site, then remove its known lever arm.
        position = np.r_[f.x[:2], 0.]-f.rotation.apply(self.offset)+f.initial_rotation.apply(self.offset)
        position[2] = height
        gravity = f.rotation.inv().apply([0., 0., -1.])
        return position, quat, f.gyro.copy(), gravity
