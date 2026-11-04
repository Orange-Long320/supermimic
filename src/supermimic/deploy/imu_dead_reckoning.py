"""Sensor-only diagnostic IMU propagation; no simulator pose/velocity inputs.

XY and velocity use the current hardware producer's four-state Kalman/ZUPT
equations. Attitude is integrated here from gyro with gated gravity leveling;
this is not a reproduction of the real IMU firmware's attitude fusion.
"""
import numpy as np
from scipy.spatial.transform import Rotation


class ImuDeadReckoning:
    def __init__(self, *, zupt=True, gravity_leveling=True):
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

    def update(self, stamp, gyro, acceleration):
        gyro, acc = np.asarray(gyro, dtype=float), np.asarray(acceleration, dtype=float)
        if (gyro.shape != (3,) or acc.shape != (3,) or
                not np.isfinite(np.r_[stamp, gyro, acc]).all()):
            raise ValueError('Invalid timestamped IMU sample')
        if self.stamp is None:
            self.stamp = float(stamp); self.previous_gyro = gyro.copy(); self.gyro = gyro.copy()
            self.samples += 1
            return
        dt = float(stamp-self.stamp)
        if not 0. < dt <= .03:
            raise ValueError('IMU duplicate/rollback or gap above 30 ms')
        self.rotation *= Rotation.from_rotvec(.5*(self.previous_gyro+gyro)*dt)
        still = abs(np.linalg.norm(acc)-9.81) <= .5 and np.linalg.norm(gyro) <= .08
        self.still_since = (self.still_since if self.still_since is not None else stamp) if still else None
        settled = still and stamp-self.still_since >= .2
        if self.gravity_leveling and settled:
            # Accelerometer gravity is only trusted under the same stillness
            # hypothesis as ZUPT. These thresholds cannot establish real rest.
            up = self.rotation.apply(acc/np.linalg.norm(acc))
            correction = np.cross(up, [0., 0., 1.])
            self.rotation = Rotation.from_rotvec((1.-np.exp(-dt/.5))*correction)*self.rotation
        world = self.rotation.apply(acc)-[0., 0., 9.81]
        F = np.eye(4); F[0, 2] = F[1, 3] = dt
        self.x = F@self.x
        self.x[:2] += .5*world[:2]*dt**2
        self.x[2:] += world[:2]*dt
        self.P = F@self.P@F.T+self.Q*(dt/.02)
        if self.zupt and settled:
            K = np.linalg.solve(self.H@self.P@self.H.T+self.R, (self.P@self.H.T).T).T
            self.x -= K@(self.H@self.x)
            residual = np.eye(4)-K@self.H
            self.P = residual@self.P@residual.T+K@self.R@K.T
            self.zupt_updates += 1
        self.stamp = float(stamp); self.previous_gyro = gyro.copy(); self.gyro = gyro.copy()
        self.samples += 1


class EncoderImuState:
    """Pelvis pose from IMU propagation and encoder FK on assumed flat support.

Only immutable mount geometry, a separate FK model, encoder values and raw IMU
samples are accepted. There is no simulator-data object or truth-pose argument.
"""
    def __init__(self, fk_model_path, joint_names, imu_offset, *, sole_offset=.05):
        import mujoco
        self.mj = mujoco
        self.filter = ImuDeadReckoning()
        self.model = mujoco.MjModel.from_xml_path(str(fk_model_path))
        self.data = mujoco.MjData(self.model)
        ids = [self.model.joint(name).id for name in joint_names]
        self.qadr = self.model.jnt_qposadr[ids]
        self.feet = [self.model.body(side+'_ankle_roll_link').id for side in ('left', 'right')]
        self.offset = np.asarray(imu_offset, dtype=float).copy()
        if self.offset.shape != (3,) or not np.isfinite(self.offset).all():
            raise ValueError('Invalid known pelvis IMU mount')
        self.sole_offset = sole_offset

    def update(self, stamp, gyro, acceleration):
        self.filter.update(stamp, gyro, acceleration)

    def state(self, joints):
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
        position = np.r_[f.x[:2], 0.]-f.rotation.apply(self.offset)+self.offset
        position[2] = height
        gravity = f.rotation.inv().apply([0., 0., -1.])
        return position, quat, f.gyro.copy(), gravity
