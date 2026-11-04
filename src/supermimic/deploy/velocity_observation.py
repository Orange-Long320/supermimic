"""Conditional velocity observations. No policy mode, truth pose or motor APIs.

Foot observations require an external declaration of flat, loaded, nonslipping
support. Encoder/IMU quietness alone never grants that declaration.
"""
import numpy as np


class DeclaredRestGyroBias:
    """Causal mean of raw gyro during independently declared physical rest."""
    def __init__(self, minimum_seconds=3., max_gap=.04):
        if minimum_seconds<=0 or max_gap<=0:raise ValueError('Invalid rest window')
        self.minimum_seconds=minimum_seconds;self.max_gap=max_gap
        self.reset()

    def reset(self):
        self.first=self.last=None;self.count=0;self.mean=np.zeros(3)

    def update(self, stamp, gyro, *, stationary_declared=False):
        g=np.asarray(gyro,dtype=float)
        if g.shape!=(3,) or not np.isfinite(np.r_[stamp,g]).all():
            raise ValueError('Invalid timestamped gyro')
        if not stationary_declared:
            self.reset();return None
        if self.last is not None and not 0<stamp-self.last<=self.max_gap:
            raise ValueError('Invalid gyro interval')
        if np.linalg.norm(g)>.08:
            self.reset();return None
        if self.first is None:self.first=stamp
        self.last=stamp;self.count+=1;self.mean+=(g-self.mean)/self.count
        return self.mean.copy() if stamp-self.first>=self.minimum_seconds else None


class VelocityObservationCorrection:
    """Fuse timestamp-matched velocity with full-rate inertial increments.

    Native IMU propagation remains unchanged. Subtract its previous velocity
    contribution from each position increment, replace it by the corrected
    velocity, then apply a trapezoidal endpoint velocity correction. Never
    resets position or uses an absolute position observation.
    """
    def __init__(self, time_constant=.03, max_gap=.1):
        if time_constant<=0 or max_gap<=0:raise ValueError('Invalid filter timescale')
        self.tau=time_constant;self.max_gap=max_gap
        self.stamp=None;self.native=None;self.x=None;self.last_observation=None

    def update(self, stamp, native_xy_state, velocity=None):
        raw=np.asarray(native_xy_state,dtype=float)
        z=None if velocity is None else np.asarray(velocity,dtype=float)
        if raw.shape!=(4,) or not np.isfinite(np.r_[stamp,raw]).all():raise ValueError('Invalid native state')
        if z is not None and (z.shape!=(2,) or not np.isfinite(z).all()):raise ValueError('Invalid velocity observation')
        if self.stamp is None:self.x=raw.copy()
        else:
            dt=stamp-self.stamp
            if not 0<dt<=self.max_gap:raise ValueError('Invalid native state interval')
            self.x[:2]+=raw[:2]-self.native[:2]+(self.x[2:]-self.native[2:])*dt
            self.x[2:]+=raw[2:]-self.native[2:]
            if z is not None:
                correction=(1-np.exp(-dt/self.tau))*(z-self.x[2:])
                self.x[:2]+=.5*dt*correction
                self.x[2:]+=correction
        if z is not None:self.last_observation=stamp
        self.stamp=stamp;self.native=raw.copy()
        return self.x.copy()

    def observation_fresh(self, max_age=.1):
        return self.last_observation is not None and self.stamp-self.last_observation<=max_age


class FootVelocityObservation:
    """FK/Jacobian observation for an explicitly assumed stationary support foot.

    Uses the SDK ankle origins, without inventing a measured contact point.
    Rejects appreciable foot rotation, lift velocity and disagreement between
    near-level feet. Cannot detect common-mode slip or suspension.
    """
    def __init__(self, model_path, joint_names, imu_offset):
        import mujoco
        self.mj=mujoco;self.model=mujoco.MjModel.from_xml_path(str(model_path));self.data=mujoco.MjData(self.model)
        joints=[self.model.joint(n).id for n in joint_names]
        self.qadr=self.model.jnt_qposadr[joints];self.dadr=self.model.jnt_dofadr[joints]
        if self.model.jnt_type[0]!=mujoco.mjtJoint.mjJNT_FREE:raise ValueError('Expected free root joint')
        self.root=int(self.model.jnt_bodyid[0])
        self.feet=[self.model.body(s+'_ankle_roll_link').id for s in ('left','right')]
        self.offset=np.asarray(imu_offset,dtype=float)
        if self.offset.shape!=(3,) or not np.isfinite(self.offset).all():raise ValueError('Invalid IMU offset')
        self.since=None;self.last_stamp=None;self.support_indices=None

    def observe(self, stamp, quat_wxyz, omega_body, q, qd, *, support_confirmed=False):
        if self.last_stamp is not None and stamp<=self.last_stamp:raise ValueError('Nonmonotonic support timestamp')
        self.last_stamp=stamp
        if not support_confirmed:
            self.since=None;self.support_indices=None;return None,'support-not-confirmed'
        q,qd,quat,omega=map(lambda a:np.asarray(a,dtype=float),(q,qd,quat_wxyz,omega_body))
        if q.shape!=self.qadr.shape or qd.shape!=q.shape or quat.shape!=(4,) or omega.shape!=(3,) or not np.isfinite(np.r_[stamp,q,qd,quat,omega]).all() or abs(np.linalg.norm(quat)-1)>.01:
            raise ValueError('Invalid kinematic observation')
        d,m=self.data,self.model;d.qpos[:]=m.qpos0;d.qpos[:3]=0;d.qpos[3:7]=quat;d.qpos[self.qadr]=q
        d.qvel[:]=0;d.qvel[3:6]=omega;d.qvel[self.dadr]=qd
        self.mj.mj_kinematics(m,d);self.mj.mj_comPos(m,d)
        lowest=min(d.xpos[f,2] for f in self.feet);rows=[];selected=[]
        for f in self.feet:
            jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv))
            self.mj.mj_jac(m,d,jp,jr,d.xpos[f],f)
            v=jp@d.qvel;w=jr@d.qvel
            if d.xpos[f,2]-lowest<=.015 and abs(v[2])<=.12 and np.linalg.norm(w)<=.2 and np.linalg.norm(v[:2])<=1.5:
                rows.append(-v);selected.append(f)
        if not rows:
            self.since=None;self.support_indices=None;return None,'rotating-or-moving-foot'
        if len(rows)==2 and np.linalg.norm(rows[0]-rows[1])>.2:
            self.since=None;self.support_indices=None;return None,'feet-disagree'
        if self.since is None or tuple(selected)!=self.support_indices:self.since=stamp
        self.support_indices=tuple(selected)
        if stamp-self.since<.04:return None,'hold'
        # Observation refers to the IMU site, where native integration runs.
        r=d.xmat[self.root].reshape(3,3)
        return (np.mean(rows,axis=0)+r@np.cross(omega,self.offset))[:2],'accepted'
