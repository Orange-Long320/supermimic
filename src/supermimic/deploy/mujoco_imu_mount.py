"""MuJoCo-model IMU mount kinematics for simulation diagnostics only."""
import numpy as np


class MujocoImuMount:
    """Pelvis-to-site pose and velocity for measured waist joint values.

    This does not validate the real sensor mount or encoder calibration.
    """
    def __init__(self, model_path, site_name, waist_names):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        pelvis = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
        joints = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in waist_names]
        if self.site < 0 or pelvis < 0 or not joints or min(joints) < 0:
            raise ValueError('Missing pelvis, site, or waist joint')
        if self.model.body_parentid[self.model.site_bodyid[self.site]] < 0:
            raise ValueError('Invalid IMU site')
        self.pelvis = pelvis
        self.qadr = np.array([self.model.jnt_qposadr[j] for j in joints])
        self.dadr = np.array([self.model.jnt_dofadr[j] for j in joints])
        if any(self.model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in joints):
            raise ValueError('Waist joints must be hinges')
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros_like(self.jacp)

    def mount_with_velocity(self, positions, velocities):
        q, qd = np.asarray(positions, dtype=float), np.asarray(velocities, dtype=float)
        if q.shape != self.qadr.shape or qd.shape != q.shape or not np.isfinite(np.r_[q, qd]).all():
            raise ValueError('Invalid waist sample')
        d = self.data
        d.qpos[:] = self.model.qpos0
        d.qpos[self.qadr] = q
        d.qvel[:] = 0.
        d.qvel[self.dadr] = qd
        self.mj.mj_fwdPosition(self.model, d)
        self.mj.mj_jacSite(self.model, d, self.jacp, self.jacr, self.site)
        pelvis_r = d.xmat[self.pelvis].reshape(3, 3)
        p = pelvis_r.T @ (d.site_xpos[self.site] - d.xpos[self.pelvis])
        r = pelvis_r.T @ d.site_xmat[self.site].reshape(3, 3)
        v = pelvis_r.T @ (self.jacp @ d.qvel)
        omega = pelvis_r.T @ (self.jacr @ d.qvel)
        return p.copy(), r.copy(), omega.copy(), v.copy()
