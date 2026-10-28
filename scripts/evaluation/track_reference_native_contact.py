"""Track a reference with existing native whole-body contact feedback on CPU.

No learned policy, root wrench, force-cap change or state reset after entry.
"""
import argparse
import json
import hashlib
import os
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--equilibrium',type=Path,required=True)
    p.add_argument('--start-frame',type=int,default=83)
    p.add_argument('--seconds',type=float,default=5.)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--recover-planned-contacts',action='store_true',
                   help='Keep planned planted-foot tasks during contact loss and add soft normal feedback')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    (args.out/'launch.json').write_text(json.dumps({'pid':os.getpid(),'command':[sys.executable,*sys.argv],
        'mujoco':mujoco.__version__,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')},indent=2))
    from supermimic.policy.native_inverse import NativeContactController
    from supermimic.robots.dimensions import PROFILE
    m=mujoco.MjModel.from_binary_path(str(args.model));actual=mujoco.MjData(m);desired=mujoco.MjData(m)
    z=np.load(args.reference);q=z['qpos'].copy();v=z['qvel'].copy()
    fps=float(z['fps'].reshape(-1)[0]) if 'fps' in z else 50.
    if not 0<=args.start_frame<len(q):raise ValueError('Invalid entry frame')
    names=list(PROFILE['joint_names']);effort=np.array([PROFILE['effort_limit'][n] for n in names])
    speed=np.array([PROFILE['velocity_limit'][n] for n in names])
    eq=np.load(args.equilibrium)['qpos'].copy();root=int(m.jnt_bodyid[0])
    feet={s:m.body('robot/'+s+'_ankle_roll_link').id for s in ['left','right']}
    controller=NativeContactController(m,eq,effort,speed,feet['left'])
    controller.integration_aware=True;controller.foot_task_weight=.1
    controller.stabilize_support_normal=args.recover_planned_contacts
    controller.root_acceleration_weight=10.
    kp=m.actuator_gainprm[:,0].copy();kd=-m.actuator_biasprm[:,2].copy()
    cols=m.jnt_qposadr[m.actuator_trnid[:,0]];vcols=m.jnt_dofadr[m.actuator_trnid[:,0]]
    actual.qpos[:]=q[args.start_frame];actual.qvel[:]=v[args.start_frame]
    actual.ctrl[:]=actual.qpos[cols]+kd*actual.qvel[vcols]/kp
    mujoco.mj_forward(m,actual)
    com=[];momentum=[];jac=np.zeros((3,m.nv));angular=np.zeros_like(jac)
    for pose,vel in zip(q,v):
        desired.qpos[:]=pose;desired.qvel[:]=vel;mujoco.mj_forward(m,desired)
        mujoco.mj_angmomMat(m,desired,angular,root)
        com.append(desired.subtree_com[root].copy());momentum.append(angular@vel)
    com=np.asarray(com);momentum=np.asarray(momentum)
    if len(q)>2:
        cv=np.gradient(com,1/fps,axis=0,edge_order=2);ca=np.zeros_like(cv)
        ca[1:-1]=np.diff(com,n=2,axis=0)*fps**2
        mr=np.gradient(momentum,1/fps,axis=0,edge_order=2)
        joint_acc=np.gradient(v,1/fps,axis=0,edge_order=2)
    else:cv=np.zeros_like(com);ca=cv.copy();mr=np.zeros_like(momentum);joint_acc=np.zeros_like(v)
    rows=[];states=[np.r_[actual.qpos.copy(),actual.qvel.copy()]];controls=[];status='completed';failure=None
    lo,hi=m.jnt_range[1:].T
    for step in range(round(args.seconds/m.opt.timestep)):
        t=step*m.opt.timestep;f=min(args.start_frame+t*fps,len(q)-1);a=int(f);b=min(a+1,len(q)-1);blend=f-a
        def sample(array):return (1-blend)*array[a]+blend*array[b]
        target=q[a].copy();tangent=np.zeros(m.nv)
        mujoco.mj_differentiatePos(m,tangent,1.,q[a],q[b]);mujoco.mj_integratePos(m,target,tangent,blend)
        desired.qpos[:]=target;desired.qvel[:]=sample(v);mujoco.mj_forward(m,desired)
        mujoco.mj_forward(m,actual)
        contact_bodies=set()
        for c in actual.contact[:actual.ncon]:
            bodies=[int(m.geom_bodyid[int(g)]) for g in c.geom]
            if c.dist<=0 and any(m.body_rootid[x]!=root for x in bodies):contact_bodies.update(bodies)
        side='left' if feet['left'] in contact_bodies else 'right' if feet['right'] in contact_bodies else 'left'
        if controller.foot!=feet[side]:
            controller.foot=feet[side]
            if hasattr(controller,'foot_vertices'):del controller.foot_vertices
        body=controller.foot
        controller.foot_position=desired.xpos[body].copy()
        controller.foot_rotation=Rotation.from_matrix(desired.xmat[body].reshape(3,3))
        foot_jp,foot_jr=np.zeros((3,m.nv)),np.zeros((3,m.nv))
        mujoco.mj_jacBody(m,desired,foot_jp,foot_jr,body)
        controller.foot_velocity_ref=np.r_[foot_jp,foot_jr]@sample(v)
        planned=set()
        if args.recover_planned_contacts:
            for x in feet.values():
                for g in range(m.ngeom):
                    if m.geom_bodyid[g]!=x or not(m.geom_contype[g] or m.geom_conaffinity[g]):continue
                    k=m.geom_dataid[g];adr,nv=m.mesh_vertadr[k],m.mesh_vertnum[k]
                    world=m.mesh_vert[adr:adr+nv]@desired.geom_xmat[g].reshape(3,3).T+desired.geom_xpos[g]
                    if world[:,2].min()<.003:planned.add(x)
        else:planned=contact_bodies
        controller.additional_support_feet=[(x,desired.xpos[x].copy(),Rotation.from_matrix(desired.xmat[x].reshape(3,3)))
            for x in feet.values() if x!=body and x in planned]
        current=Rotation.from_quat(actual.qpos[[4,5,6,3]]);wanted=Rotation.from_quat(target[[4,5,6,3]])
        desired_omega=(current.inv()*wanted).apply(sample(v)[3:6])
        controller.root_acceleration_ref=sample(joint_acc)[3:6]+100*(current.inv()*wanted).as_rotvec()+20*(desired_omega-actual.qvel[3:6])
        controller.servo_seed=actual.ctrl.copy()
        try:
            tau,report=controller.solve(actual.qpos.copy(),actual.qvel.copy(),com_target=sample(com),
                com_velocity=sample(cv),com_acceleration=sample(ca),momentum_target=sample(momentum),
                momentum_rate=sample(mr),qref=target,joint_velocity=sample(v)[6:],joint_acceleration=sample(joint_acc)[6:])
        except RuntimeError as e:
            status='controller_rejected';failure={'message':str(e),'details':getattr(controller,'failure',None),'time_s':t}
            if hasattr(controller,'failure_arrays'):np.savez(args.out/'failure-state.npz',**controller.failure_arrays)
            break
        actual.ctrl[:]=actual.qpos[cols]+(tau+kd*actual.qvel[vcols])/kp
        mujoco.mj_forward(m,actual)
        explicit_error=float(abs(actual.qacc-controller.last_explicit_acceleration).max())
        previous_velocity=actual.qvel.copy()
        controls.append(actual.ctrl.copy());mujoco.mj_step(m,actual)
        integrated=(actual.qvel-previous_velocity)/m.opt.timestep
        integrated_error=float(abs(integrated-controller.previous*20).max())
        mujoco.mj_forward(m,actual)
        mujoco.mj_jacSubtreeCom(m,actual,jac,root)
        hard=float(np.maximum(np.maximum(lo-actual.qpos[7:],actual.qpos[7:]-hi),0).max())
        up=float(actual.xmat[root].reshape(3,3)[2,2])
        rows.append({'time_s':t+m.opt.timestep,'root_height_m':float(actual.qpos[2]),'root_up_z':up,
            'com_speed_m_s':float(np.linalg.norm(jac@actual.qvel)),'hard_violation_rad':hard,
            'motor_ratio':float((abs(actual.actuator_force)/effort).max()),'solver_root_residual':report['equality_residual'],
            'explicit_acceleration_error_max':explicit_error,'integrated_acceleration_error_max':integrated_error,
            'desired_com_acceleration':report['desired_com_acceleration'],
            'achieved_com_acceleration':report['achieved_com_acceleration']})
        states.append(np.r_[actual.qpos.copy(),actual.qvel.copy()])
        if step%100==0:print(json.dumps(rows[-1]),flush=True)
        if hard>.001 or actual.qpos[2]<.22 or up<.3:
            status='safety_stop';break
    np.savez(args.out/'trajectory.npz',state=np.asarray(states),dt=np.array(m.opt.timestep),controls=np.asarray(controls))
    result={'configuration':{k:str(x) if isinstance(x,Path) else x for k,x in vars(args).items()},
        'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),
        'reference_sha256':hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        'status':status,'seconds_executed':len(rows)*m.opt.timestep,'failure':failure,
        'maximum_hard_violation_rad':max((x['hard_violation_rad'] for x in rows),default=0),
        'maximum_motor_ratio':max((x['motor_ratio'] for x in rows),default=0),
        'last':rows[-1] if rows else None,'rows':rows,'state_resets_after_entry':0,
        'policy_evaluation':False,'approved_for_training':False,
        'limitations':'Existing whole-body contact feedback; completion requires independent steady-state/contact/limit verification.'}
    (args.out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:x for k,x in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
