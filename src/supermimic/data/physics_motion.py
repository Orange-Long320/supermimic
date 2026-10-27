"""Export consistent generalized velocities without changing legacy CSV imports."""
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from supermimic.data.motion_convert import ResampledMotion,forward_kinematics


def differentiate_qpos(model,poses,fps):
    """Use matching stencils for translations, rotations and joint positions."""
    poses=np.asarray(poses,dtype=float)
    if poses.ndim!=2 or poses.shape[1]!=model.nq or fps<=0:
        raise ValueError('Invalid generalized positions or frame rate')
    velocity=np.zeros((len(poses),model.nv))
    first,second=np.zeros(model.nv),np.zeros(model.nv)
    dt=1/fps
    for i in range(len(poses)):
        if len(poses)==1:break
        if len(poses)==2:
            j=1-i
            mujoco.mj_differentiatePos(model,velocity[i],(j-i)*dt,poses[i],poses[j])
        elif i==0 or i==len(poses)-1:
            direction=1 if i==0 else -1
            mujoco.mj_differentiatePos(model,first,direction*dt,poses[i],poses[i+direction])
            mujoco.mj_differentiatePos(model,second,2*direction*dt,poses[i],poses[i+2*direction])
            velocity[i]=2*first-second
        else:
            mujoco.mj_differentiatePos(model,first,dt,poses[i],poses[i+1])
            mujoco.mj_differentiatePos(model,second,-dt,poses[i],poses[i-1])
            velocity[i]=(first+second)/2
    return velocity


def export_qpos_motion(model,poses,path,fps,qvel=None,source=''):
    """Preserve supplied simulator qvel; otherwise derive all DOFs consistently."""
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    q=np.asarray(poses,dtype=float)
    supplied=qvel is not None
    v=np.asarray(qvel,dtype=float) if supplied else differentiate_qpos(model,q,fps)
    if q.shape!=(len(q),model.nq) or v.shape!=(len(q),model.nv):
        raise ValueError('Position/velocity dimensions do not match the robot')
    if not np.isfinite(q).all() or not np.isfinite(v).all():raise ValueError('Nonfinite motion')
    rotation=Rotation.from_quat(q[:,[4,5,6,3]])
    motion=ResampledMotion(fps,q[:,:3],q[:,3:7],q[:,7:],v[:,:3],rotation.apply(v[:,3:6]),v[:,6:])
    bodies=forward_kinematics(model,motion)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez(path,fps=np.array([fps],dtype=np.float32),joint_pos=motion.joint_pos.astype(np.float32),
        joint_vel=motion.joint_vel.astype(np.float32),body_pos_w=bodies.pos_w.astype(np.float32),
        body_quat_w=bodies.quat_w.astype(np.float32),body_lin_vel_w=bodies.lin_vel_w.astype(np.float32),
        body_ang_vel_w=bodies.ang_vel_w.astype(np.float32),body_names=np.array(bodies.body_names),
        joint_names=np.array([model.joint(i).name for i in range(1,model.njnt)]),
        source=np.array(str(source)),input_fps=np.array([fps],dtype=np.float32),
        velocity_source=np.array('provided_generalized_velocity' if supplied else 'consistent_generalized_pose_difference'))
