"""Search toe-support poses using contact geometry and native motor budgets.

The gravity-wrench calculation is a necessary static estimate, not a native
soft-contact certificate. Selected poses still require independent calibration.
"""
import argparse
import hashlib
import itertools
import json
import os
import time
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--equilibrium',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--toe',type=float,nargs='+',default=[.6,.8,1.,1.2])
    p.add_argument('--height',type=float,nargs='+',default=[.5,.58,.66,.74,.82])
    p.add_argument('--pitch',type=float,nargs='+',default=[-.3,0,.3,.6])
    p.add_argument('--roll',type=float,nargs='+',default=[-.2,0,.2])
    p.add_argument('--orientation-weight',type=float,default=20.)
    p.add_argument('--keep',type=int,default=24);p.add_argument('--max-seconds',type=float,default=120.)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    from supermimic.kinematics.support_reference import SupportPoseIK
    from supermimic.robots.dimensions import PROFILE
    m=mujoco.MjModel.from_binary_path(str(args.model));d=mujoco.MjData(m)
    with np.load(args.equilibrium) as z:q0=z['qpos'].copy();assert z['joint_names'].tolist()==list(PROFILE['joint_names'])
    names=list(PROFILE['joint_names']);effort=np.array([PROFILE['effort_limit'][n] for n in names])
    assert [m.joint(i).name.rsplit('/',1)[-1] for i in range(1,m.njnt)]==names
    foot=m.body('robot/left_ankle_roll_link').id;root=int(m.jnt_bodyid[0]);d.qpos[:]=q0;mujoco.mj_forward(m,d)
    initial_position=d.xpos[foot].copy();initial_rotation=Rotation.from_matrix(d.xmat[foot].reshape(3,3))
    yaw=Rotation.from_quat(q0[3:7][[1,2,3,0]]).as_euler('xyz')[2]
    meshes=[];foot_points=[]
    for g in range(m.ngeom):
        if int(m.body_rootid[int(m.geom_bodyid[g])])!=root or not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
        k=int(m.geom_dataid[g]);assert k>=0;a,n=m.mesh_vertadr[k],m.mesh_vertnum[k]
        points=m.mesh_vert[a:a+n].astype(float);points=points[ConvexHull(points).vertices]
        side=1 if int(m.geom_bodyid[g])==foot else 2 if 'foot_' in m.geom(g).name else 0
        meshes.append((g,points,side))
        if side==1:
            world=points@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
            foot_points.extend(initial_rotation.inv().apply(world-initial_position))
    foot_points=np.asarray(foot_points);front=foot_points[foot_points[:,0]>foot_points[:,0].max()-.008]
    pivot_local=np.array([front[:,0].max(),front[:,1].mean(),front[:,2].min()])
    pivot_world=initial_position+initial_rotation.apply(pivot_local)
    original_sole=float((initial_rotation.apply(foot_points)+initial_position)[:,2].min())
    ik=SupportPoseIK(m,q0,foot);lo,hi=m.jnt_range[1:].T
    configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    configuration.update(pid=os.getpid(),model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                         equilibrium_sha256=hashlib.sha256(args.equilibrium.read_bytes()).hexdigest())
    (args.out/'configuration.json').write_text(json.dumps(configuration,indent=2))
    (args.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    trials=[];poses=[];started=time.monotonic()
    seeds=[q0[ik.columns],[-.6,-.1,0,1.5,.3,0],[-1.2,-.15,0,2.,.4,0]]
    for toe,height,pitch,roll in itertools.product(args.toe,args.height,args.pitch,args.roll):
        fr=initial_rotation*Rotation.from_rotvec([0,toe,0])
        fp=pivot_world-fr.apply(pivot_local)
        world=fr.apply(foot_points)+fp;fp[2]+=original_sole-world[:,2].min();world=fr.apply(foot_points)+fp
        low=world[world[:,2]<world[:,2].min()+.003]
        target=np.r_[low[:,:2].mean(0),height];preferred=Rotation.from_euler('xyz',[roll,pitch,yaw])
        def residual(values):
            q,rotation=ik.place(q0,values,fp,fr);d.qpos[:]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
            return np.r_[1000*(d.subtree_com[root]-target),args.orientation_weight*(preferred.inv()*rotation).as_rotvec()]
        best=None
        for seed in seeds:
            fit=least_squares(residual,np.clip(seed,ik.lower,ik.upper),bounds=(ik.lower,ik.upper),
                              max_nfev=80,xtol=1e-10,ftol=1e-10,gtol=1e-7)
            value=float(np.linalg.norm(residual(fit.x)))
            if best is None or value<best[0]:best=(value,fit)
            if value<1e-5:break
        q,rotation=ik.place(q0,best[1].x,fp,fr);d.qpos[:]=q;d.qvel[:]=0;mujoco.mj_forward(m,d)
        com=d.subtree_com[root].copy();error=float(np.linalg.norm(com-target))
        heights={side:min(float((points@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])[:,2].min())
                          for g,points,kind in meshes if kind==side) for side in [0,1,2]}
        jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);cop=np.r_[com[:2],0.]
        mujoco.mj_jac(m,d,jp,jr,cop,foot)
        gravity=-float(m.body_mass.sum())*m.opt.gravity
        wrench_torque=d.qfrc_bias-d.qfrc_passive-jp.T@gravity
        ratio=float((abs(wrench_torque[6:])/effort).max())
        up=float(rotation.as_matrix()[2,2]);hard=float(np.maximum(np.maximum(lo-q[7:],q[7:]-hi),0).max())
        geometric=bool(error<.001 and hard<1e-9 and heights[0]>.005 and heights[2]>.04 and heights[1]>-.005 and up>.6)
        row={'index':len(poses),'toe_angle':toe,'requested_height':height,'preferred_pitch':pitch,'preferred_roll':roll,
             'com':com.tolist(),'target_com':target.tolist(),'com_error_m':error,
             'orientation_error_rad':float(np.linalg.norm((preferred.inv()*rotation).as_rotvec())),
             'actual_root_rpy':rotation.as_euler('xyz').tolist(),'root_up_z':up,'heights':{str(k):v for k,v in heights.items()},
             'hard_violation_rad':hard,'gravity_wrench_motor_ratio_estimate':ratio,
             'gravity_wrench_unactuated_residual':float(abs(wrench_torque[:6]).max()),
             'geometric_candidate':geometric,'screen_candidate':bool(geometric and ratio<.99),
             'native_static_feasibility_verified':False,'approved_for_training':False}
        poses.append(q);trials.append(row)
        if len(trials)%24==0:print(json.dumps({'trials':len(trials),'screen_candidates':sum(r['screen_candidate'] for r in trials),'elapsed_s':time.monotonic()-started}),flush=True)
        if time.monotonic()-started>=args.max_seconds:break
    order=sorted(range(len(trials)),key=lambda i:(not trials[i]['screen_candidate'],not trials[i]['geometric_candidate'],trials[i]['gravity_wrench_motor_ratio_estimate'],trials[i]['com_error_m']))
    saved=[]
    for i in order[:args.keep]:
        if not trials[i]['screen_candidate']:continue
        name=f'candidate-{i:03d}.npz';q=poses[i]
        np.savez(args.out/name,state=np.r_[q,np.zeros(m.nv)][None],qpos=q,joint_names=np.asarray(names))
        trials[i]['state_file']=name;saved.append(trials[i])
    np.savez(args.out/'all-poses.npz',qpos=np.asarray(poses),joint_names=np.asarray(names))
    report={'configuration':configuration,'elapsed_s':time.monotonic()-started,'trials':trials,'selected':saved,
            'approved_for_training':False,'note':'Gravity point-wrench estimates are not native soft-contact feasibility certificates.'}
    (args.out/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'trials':len(trials),'selected':len(saved),'elapsed_s':report['elapsed_s'],'best':saved[:1]}),flush=True)


if __name__=='__main__':main()
