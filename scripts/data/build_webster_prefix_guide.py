"""Build an unqualified kinematic guide from recorded preparation and teacher.

No simulated state is changed. Pose blending has analytic velocity terms;
every exported guide still needs actual policy execution and physical checks.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from supermimic.data.motion_convert import ResampledMotion, forward_kinematics, load_g1_model, quat_mul, quat_conjugate
from supermimic.robots.dimensions import PROFILE


def smooth(time, start, end):
    s=np.clip((time-start)/(end-start),0,1)
    return s**3*(10-15*s+6*s*s),30*s*s*(1-s)**2/(end-start)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher',type=Path,required=True)
    p.add_argument('--prefix',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--blend-start',type=float,default=.5)
    p.add_argument('--blend-end',type=float,default=.72)
    p.add_argument('--height',type=float,default=0.,help='Optional smooth world-Z guide offset in metres; original dynamics remain unchanged')
    args=p.parse_args()
    if not 0<=args.blend_start<args.blend_end or not abs(args.height)<=.1:p.error('Invalid blend or guide offset')
    args.out.mkdir(parents=True,exist_ok=False)
    teacher=np.load(args.teacher);source_path=args.teacher.parent/'source-trajectory.npz'
    source=np.load(source_path);initial_path=args.teacher.parent/'initial-state.npz';initial=np.load(initial_path)
    q=np.vstack([initial['qpos'],source['qpos']]).astype(float)
    v=np.vstack([initial['qvel'],source['qvel']]).astype(float)
    prefix_path=args.prefix/'baseline-trajectory.npz';prefix=np.load(prefix_path)
    prefix_initial_path=args.prefix/'baseline-physics.npz';prefix_initial=np.load(prefix_initial_path)
    pq=np.vstack([prefix_initial['initial_qpos'],prefix['qpos']]).astype(float)
    pv=np.vstack([prefix_initial['initial_qvel'],prefix['qvel']]).astype(float)
    if not np.allclose(q[:,7:],teacher['joint_pos'],atol=1e-7,rtol=0) or len(q)!=len(teacher['joint_pos']):
        raise ValueError('Teacher does not match its recorded source')
    if max(abs(pq[0]-q[0]).max(),abs(pv[0]-v[0]).max())>1e-5:raise ValueError('Preparation and teacher initial states differ')
    if any(abs(float(z['fps'][0])-50)>1e-6 for z in [source,prefix,teacher]):raise ValueError('Expected 50 Hz recordings')
    if (len(pq)-1)/50<args.blend_end:raise ValueError('Preparation recording does not cover the blend')
    q_initial=q[0].copy();v_initial=v[0].copy()
    count=min(len(pq),len(q));t=np.arange(count)/50
    w,dw=smooth(t,args.blend_start,args.blend_end)
    b=q[:count].copy();bv=v[:count].copy();a=pq[:count];av=pv[:count]
    for qs,vs in [(slice(0,3),slice(0,3)),(slice(7,None),slice(6,None))]:
        q[:count,qs]=(1-w[:,None])*a[:,qs]+w[:,None]*b[:,qs]
        v[:count,vs]=(1-w[:,None])*av[:,vs]+w[:,None]*bv[:,vs]+dw[:,None]*(b[:,qs]-a[:,qs])
    qa=a[:,3:7];qb=b[:,3:7].copy()
    da=.5*quat_mul(qa,np.c_[np.zeros(count),av[:,3:6]])
    db=.5*quat_mul(qb,np.c_[np.zeros(count),bv[:,3:6]])
    sign=np.where((qa*qb).sum(1)<0,-1.,1.)[:,None];qb*=sign;db*=sign
    raw=(1-w[:,None])*qa+w[:,None]*qb;norm=np.linalg.norm(raw,axis=1,keepdims=True)
    quat=raw/norm;derivative=(1-w[:,None])*da+w[:,None]*db+dw[:,None]*(qb-qa)
    derivative=(derivative-quat*(quat*derivative).sum(1,keepdims=True))/norm
    q[:count,3:7]=quat;v[:count,3:6]=2*quat_mul(quat_conjugate(quat),derivative)[:,1:]
    # Preserve the exact original reset, including all initial velocity terms.
    q[0]=q_initial;v[0]=v_initial
    time=np.arange(len(q))/50
    rise,drise=smooth(time,.3,.64);fall,dfall=smooth(time,.95,1.8)
    offset=args.height*rise*(1-fall);rate=args.height*(drise*(1-fall)-rise*dfall)
    q[:,2]+=offset;v[:,2]+=rate
    model=load_g1_model()
    omega_world=Rotation.from_quat(q[:,[4,5,6,3]]).apply(v[:,3:6])
    motion=ResampledMotion(50.,q[:,:3],q[:,3:7],q[:,7:],v[:,:3],omega_world,v[:,6:])
    bodies=forward_kinematics(model,motion)
    np.savez(args.out/'webster.npz',fps=np.array([50.],np.float32),joint_pos=q[:,7:].astype(np.float32),
        joint_vel=v[:,6:].astype(np.float32),body_pos_w=bodies.pos_w.astype(np.float32),
        body_quat_w=bodies.quat_w.astype(np.float32),body_lin_vel_w=bodies.lin_vel_w.astype(np.float32),
        body_ang_vel_w=bodies.ang_vel_w.astype(np.float32),joint_names=np.array(PROFILE['joint_names']),
        body_names=np.array(bodies.body_names),source=np.array(str(args.teacher)))
    np.savez(args.out/'guide-state.npz',qpos=q,qvel=v,fps=np.array([50.]))
    output=np.load(args.out/'webster.npz')
    for key in ['joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w']:
        if abs(output[key][0]-teacher[key][0]).max()>1e-6:raise ValueError('Guide changes the initial reference state')
    # Check all body link-origin velocities via native point Jacobians.
    data=mujoco.MjData(model);maximum_velocity_error=0.
    for i in range(len(q)):
        data.qpos[:]=q[i];data.qvel[:]=v[i];mujoco.mj_forward(model,data)
        for j,bid in enumerate(range(1,model.nbody)):
            jp=np.zeros((3,model.nv));jr=np.zeros_like(jp)
            mujoco.mj_jac(model,data,jp,jr,data.xpos[bid],bid)
            maximum_velocity_error=max(maximum_velocity_error,float(abs(jp@v[i]-bodies.lin_vel_w[i,j]).max()),float(abs(jr@v[i]-bodies.ang_vel_w[i,j]).max()))
    if maximum_velocity_error>=1e-4:raise ValueError('Guide velocities do not match native kinematics')
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    result={'guide_only':True,'physical_candidate_passed':False,'approved_for_training':False,
        'scope':'Reference features only; no actual state clipping, dynamics or running training input changed',
        'blend_start_s':args.blend_start,'blend_end_s':args.blend_end,'world_z_offset_m':args.height,
        'velocity_blend':'Analytic position and normalised quaternion derivatives; no invented zero-velocity reset',
        'maximum_native_FK_velocity_error':maximum_velocity_error,'frames':len(q),'fps':50.,
        'sha256':{str(f):digest(f) for f in [args.teacher,source_path,initial_path,prefix_path,prefix_initial_path,args.out/'webster.npz']}}
    (args.out/'GUIDE_ONLY.json').write_text(json.dumps(result,indent=2))
    (args.out/'as-run.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(result))


if __name__=='__main__':main()
