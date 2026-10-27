"""Calibrate a two-foot terminal pose without changing native contact physics."""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--leg-adjustments',action='store_true',help='Also optimize hip pitch/roll and knee positions within 0.25 radians')
    p.add_argument('--refine-from',type=Path,help='Initialize from a previous calibration result')
    p.add_argument('--anchor-left-foot',action='store_true',help='Parameterize root placement by left-foot attitude to reduce contact-branch cancellation')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    from supermimic.robots.dimensions import PROFILE
    m=mujoco.MjModel.from_binary_path(str(args.model));d=mujoco.MjData(m)
    with np.load(args.initial) as z:q0=z['state'][0,:m.nq].copy()
    names=list(PROFILE['joint_names']);effort=np.array([PROFILE['effort_limit'][n] for n in names])
    root=int(m.jnt_bodyid[0]);mass=float(m.body_mass.sum());weight=mass*9.81
    feet={side:m.body('robot/'+side+'_ankle_roll_link').id for side in ['left','right']}
    joints=[m.joint('robot/'+side+'_ankle_'+axis+'_joint').id for side in ['left','right'] for axis in ['pitch','roll']]
    if args.leg_adjustments:
        joints += [m.joint('robot/'+side+'_'+name+'_joint').id for side in ['left','right'] for name in ['hip_pitch','hip_roll','knee']]
    columns=m.jnt_qposadr[joints];rotation0=Rotation.from_quat(q0[3:7][[1,2,3,0]])
    local=mujoco.MjData(m);local.qpos[:]=q0;mujoco.mj_kinematics(m,local)
    left_position=local.xpos[feet['left']].copy()
    left_rotation=Rotation.from_matrix(local.xmat[feet['left']].reshape(3,3))
    meshes=[]
    for g in range(m.ngeom):
        if int(m.body_rootid[int(m.geom_bodyid[g])])!=root or not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
        k=int(m.geom_dataid[g]);assert k>=0;a,n=m.mesh_vertadr[k],m.mesh_vertnum[k]
        side=next((s for s,b in feet.items() if int(m.geom_bodyid[g])==b),'nonfoot')
        meshes.append((g,m.mesh_vert[a:a+n].copy(),side))
    scale=np.r_[np.full(3,weight),np.full(3,weight*.7)]
    radius=np.r_[np.full(4,.15),np.full(len(joints)-4,.25)]
    lower=np.r_[[-.12,-.15,-.15],np.maximum(m.jnt_range[joints,0]+.025-q0[columns],-radius)]
    upper=np.r_[[.03,.15,.15],np.minimum(m.jnt_range[joints,1]-.025-q0[columns],radius)]
    def evaluate(x):
        q=q0.copy();q[2]+=x[0];q[3:7]=(Rotation.from_rotvec([x[1],x[2],0])*rotation0).as_quat()[[3,0,1,2]]
        q[columns]+=x[3:]
        if args.anchor_left_foot:
            local.qpos[:]=q;local.qpos[:3]=0;local.qpos[3:7]=[1,0,0,0];mujoco.mj_kinematics(m,local)
            attitude=Rotation.from_rotvec([x[1],x[2],0])*left_rotation
            root_rotation=attitude*Rotation.from_matrix(local.xmat[feet['left']].reshape(3,3)).inv()
            q[3:7]=root_rotation.as_quat()[[3,0,1,2]]
            q[:3]=left_position+[0,0,x[0]]-root_rotation.apply(local.xpos[feet['left']])
        d.qpos[:]=q;d.qvel[:]=0;d.qacc[:]=0;mujoco.mj_inverse(m,d)
        heights={s:min(float((v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])[:,2].min())
                       for g,v,kind in meshes if kind==s) for s in ['left','right','nonfoot']}
        normal={s:0. for s in feet};contacts=[]
        for i,c in enumerate(d.contact[:d.ncon]):
            bodies=[int(m.geom_bodyid[int(g)]) for g in c.geom];force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
            if force[0]<=1e-5:continue
            for side,body in feet.items():
                if body in bodies and any(int(m.body_rootid[b])!=root for b in bodies):normal[side]+=float(force[0])
            contacts.extend(m.body(b).name for b in bodies if b)
        return q,d.qfrc_inverse.copy(),heights,normal,sorted(set(contacts))
    def residual(x):
        _,force,h,n,_=evaluate(x)
        return np.r_[force[:6]/scale*20,
            [max(h[s]+.0002,0)*50 for s in feet],[max(-.0045-h[s],0)*100 for s in feet],
            [max(.05*weight-n[s],0)/weight*10 for s in feet],max(.005-h['nonfoot'],0)*100,
            np.maximum(abs(force[6:])/effort-.95,0)*10,x*.001]
    def jacobian(x):
        steps=np.r_[1e-6,np.full(len(x)-1,1e-5)];columns=[]
        for i,h in enumerate(steps):
            delta=np.zeros(len(x));delta[i]=h;columns.append((residual(x+delta)-residual(x-delta))/(2*h))
        return np.stack(columns,axis=1)
    trials=[]
    seeds=[np.zeros(len(lower))] if args.anchor_left_foot else []
    seeds += [np.r_[z,roll,0.,np.zeros(len(joints))] for z in [-.008,-.018,-.03] for roll in [-.03,.03]]
    if args.refine_from:
        previous=json.loads(args.refine_from.read_text())['best']['adjustments']
        seeds.insert(0,np.r_[previous,np.zeros(len(lower)-len(previous))])
    for seed in seeds:
        seed=np.clip(seed,lower,upper)
        fit=least_squares(residual,seed,jac=jacobian,bounds=(lower,upper),max_nfev=250,
                          ftol=1e-12,xtol=1e-12,gtol=1e-12)
        q,force,h,n,contacts=evaluate(fit.x);ratio=float((abs(force[6:])/effort).max())
        root_error=float(abs(force[:6]/scale).max());up=float(d.xmat[root].reshape(3,3)[2,2])
        hard=float(np.maximum(np.maximum(m.jnt_range[1:,0]-q[7:],q[7:]-m.jnt_range[1:,1]),0).max())
        passed=bool(root_error<1e-5 and ratio<=1 and hard<1e-9 and min(n.values())>10 and
                    min(h[s] for s in feet)>-.005 and max(h[s] for s in feet)<.005 and h['nonfoot']>.005 and up>.85)
        item={'adjustments':fit.x.tolist(),'cost':float(fit.cost),'root_force_residual_normalized':root_error,
              'maximum_motor_ratio':ratio,'root_force_residual':force[:6].tolist(),'heights':h,'normal_force_n':n,'contact_bodies':contacts,
              'root_up_z':up,'hard_violation_rad':hard,'static_force_feasible':passed,
              'strict_geometry_static_feasible':passed,'evaluations':int(fit.nfev),
              'adjusted_joint_names':[m.joint(j).name for j in joints]}
        trials.append((item,q,force))
        print(json.dumps({'trial':len(trials),'root_residual':root_error,'maximum_motor_ratio':ratio,'static_feasible':passed}),flush=True)
        if passed:break
    trials.sort(key=lambda x:(not x[0]['static_force_feasible'],x[0]['cost']))
    for i,(item,q,force) in enumerate(trials):
        item['state_file']=f'candidate-{i:03d}.npz'
        np.savez(args.out/item['state_file'],qpos=q,qvel=np.zeros(m.nv),required_motor_torque=force[6:],joint_names=np.asarray(names))
    item,q,force=trials[0]
    np.savez(args.out/'equilibrium.npz',qpos=q,qvel=np.zeros(m.nv),required_motor_torque=force[6:],joint_names=np.asarray(names))
    report={'purpose':__doc__,'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),
            'initial_sha256':hashlib.sha256(args.initial.read_bytes()).hexdigest(),'best':item,
            'trials':[x[0] for x in trials],'approved_for_training':False,'configuration':{k:str(v) for k,v in vars(args).items()},
            'note':'Static certificate only. Requires independent dynamic hold and a physically validated transition.'}
    (args.out/'as-run.py').write_bytes(Path(__file__).read_bytes());(args.out/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'best':item,'approved_for_training':False}),flush=True)


if __name__=='__main__':main()
