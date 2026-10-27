"""Exact planted-foot parameterization for a coordinated support-leg reference."""
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def root_rate_for_momentum(model,pose,foot_body,joint_rates,com_velocity,foot_velocity,rotation_vector,momentum):
    """Match endpoint H while preserving COM velocity and the planted-foot twist."""
    d=mujoco.MjData(model);d.qpos[:]=pose;mujoco.mj_forward(model,d)
    root=int(model.jnt_bodyid[0]);nv=model.nv
    com=np.zeros((3,nv));jp=np.zeros_like(com);jr=np.zeros_like(com);angular=np.zeros_like(com)
    mujoco.mj_jacSubtreeCom(model,d,com,root);mujoco.mj_jacBody(model,d,jp,jr,foot_body)
    mujoco.mj_angmomMat(model,d,angular,root)
    foot_name=model.body(foot_body).name;prefix=foot_name[:-len('left_ankle_roll_link')]
    unknown=np.r_[np.arange(6),[model.jnt_dofadr[model.joint(prefix+'left_'+s+'_joint').id]
        for s in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']]].astype(int)
    constraints=np.r_[com,jp,jr,np.eye(nv)[3:6]];matrix=constraints[:,unknown]
    if np.linalg.matrix_rank(matrix)!=12:raise ValueError('Endpoint support chain is singular')
    known=np.r_[np.zeros(6),joint_rates].astype(float);known[unknown]=0
    base=known.copy();base[unknown]=np.linalg.solve(matrix,np.r_[com_velocity,foot_velocity,np.zeros(3)]-constraints@known)
    epsilon=1e-6;root_rotation=Rotation.from_quat(np.asarray(pose)[3:7][[1,2,3,0]])
    deltas=[]
    for axis in np.eye(3):
        omega=(Rotation.from_rotvec(rotation_vector+epsilon*axis)*Rotation.from_rotvec(rotation_vector-epsilon*axis).inv()).as_rotvec()/(2*epsilon)
        dv=np.zeros(nv);dv[unknown]=np.linalg.solve(matrix,np.r_[np.zeros(9),root_rotation.inv().apply(omega)])
        deltas.append(dv)
    deltas=np.stack(deltas,axis=1);rates=np.linalg.solve(angular@deltas,np.asarray(momentum)-angular@base)
    velocity=base+deltas@rates
    residual=np.r_[com@velocity-com_velocity,np.r_[jp,jr]@velocity-foot_velocity,angular@velocity-momentum]
    return rates,velocity,float(np.max(abs(residual)))


class SupportPoseIK:
    def __init__(self,model,initial,foot_body,margin=.025):
        self.model=model;self.data=mujoco.MjData(model);self.local=mujoco.MjData(model)
        self.root=int(model.jnt_bodyid[0]);self.foot=foot_body;self.initial=np.asarray(initial).copy()
        foot_name=model.body(foot_body).name
        if not foot_name.endswith('left_ankle_roll_link'):raise ValueError('Expected the left support foot body')
        prefix=foot_name[:-len('left_ankle_roll_link')]
        self.joints=np.array([model.joint(prefix+'left_'+suffix+'_joint').id for suffix in
                              ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']])
        self.columns=model.jnt_qposadr[self.joints]
        self.lower=model.jnt_range[self.joints,0]+margin;self.upper=model.jnt_range[self.joints,1]-margin
        self.previous=self.initial[self.columns].copy()

    def place(self,posture,values,foot_position,foot_rotation):
        m=self.model;local=self.local;q=np.asarray(posture).copy();q[self.columns]=values
        local.qpos[:]=q;local.qpos[:3]=0.;local.qpos[3:7]=[1,0,0,0];mujoco.mj_kinematics(m,local)
        root_rotation=foot_rotation*Rotation.from_matrix(local.xmat[self.foot].reshape(3,3)).inv()
        q[:3]=foot_position-root_rotation.apply(local.xpos[self.foot]);q[3:7]=root_rotation.as_quat()[[3,0,1,2]]
        return q,root_rotation

    def fit(self,posture,com_target,foot_position,foot_rotation,root_rotation):
        m=self.model;d=self.data
        def residual(values):
            q,rotation=self.place(posture,values,foot_position,foot_rotation)
            d.qpos[:]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
            return np.r_[(d.subtree_com[self.root]-com_target)*1000.,
                         (root_rotation.inv()*rotation).as_rotvec()*200.]
        fit=least_squares(residual,np.clip(self.previous,self.lower,self.upper),bounds=(self.lower,self.upper),
                          xtol=1e-11,ftol=1e-11,gtol=1e-9,max_nfev=100)
        q,rotation=self.place(posture,fit.x,foot_position,foot_rotation);error=residual(fit.x)
        self.previous=fit.x.copy()
        return q,{'com_error_m':float(np.linalg.norm(error[:3])/1000),
                  'root_orientation_error_rad':float(np.linalg.norm(error[3:])/200),
                  'nfev':int(fit.nfev),'solver_success':bool(fit.success)}


class SupportReference:
    """Linear sampling of a densely sampled kinematic design, with a terminal hold."""
    def __init__(self,path):
        with np.load(path) as z:self.arrays={k:z[k].copy() for k in z.files}
        self.times=self.arrays['time'];self.duration=float(self.times[-1])
        if len(self.times)<2 or not np.all(np.diff(self.times)>0):raise ValueError('Reference time must increase')

    def sample(self,time_s):
        t=float(np.clip(time_s,0,self.duration));right=int(np.clip(np.searchsorted(self.times,t,side='right'),1,len(self.times)-1))
        left=right-1;alpha=(t-self.times[left])/(self.times[right]-self.times[left])
        keys=['qpos','qvel','qacc','com','com_velocity','com_acceleration','momentum','momentum_rate',
              'foot_position','foot_quat','foot_velocity','foot_acceleration']
        result={k:(1-alpha)*self.arrays[k][left]+alpha*self.arrays[k][right] for k in keys}
        for k,cols in [('qpos',slice(3,7)),('foot_quat',slice(None))]:
            # Adjacent source quaternions are stored in one hemisphere.
            result[k][cols]/=np.linalg.norm(result[k][cols])
        if time_s<0 or time_s>self.duration:
            for k in ['qvel','qacc','com_velocity','com_acceleration','momentum','momentum_rate','foot_velocity','foot_acceleration']:
                result[k][:]=0
        return result
