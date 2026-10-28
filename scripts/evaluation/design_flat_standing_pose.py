"""Fit a level two-foot terminal pose before native dynamic verification."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--native-balance',action='store_true',help='Jointly polish contact loads while retaining near-level soles')
    p.add_argument('--symmetric-feet',action='store_true',help='Parallel feet with aligned toes and symmetric lateral placement')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    from supermimic.robots.dimensions import PROFILE
    m=mujoco.MjModel.from_binary_path(str(args.model));d=mujoco.MjData(m)
    with np.load(args.initial) as z:q0=z['qpos'].copy()
    d.qpos[:]=q0;mujoco.mj_forward(m,d);root=int(m.jnt_bodyid[0]);weight=float(m.body_mass.sum()*9.81)
    feet={s:m.body('robot/'+s+'_ankle_roll_link').id for s in ['left','right']}
    targets={};sole_centers=[];tilts={}
    for side,body in feet.items():
        candidates=[]
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g])!=body or not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
            k=int(m.geom_dataid[g]);a,n=m.mesh_vertadr[k],m.mesh_vertnum[k];vertices=m.mesh_vert[a:a+n].astype(float)
            hull=ConvexHull(vertices);tri=vertices[hull.simplices]
            areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
            planes,index=np.unique(np.round(hull.equations,5),axis=0,return_inverse=True)
            area=np.bincount(index,weights=areas);normals=planes[:,:3]@d.geom_xmat[g].reshape(3,3).T
            for j in np.flatnonzero(normals[:,2]<-.5):candidates.append((area[j],g,vertices,normals[j]))
        _,g,vertices,normal=max(candidates,key=lambda x:x[0]);normal=normal/np.linalg.norm(normal)
        correction=Rotation.align_vectors(np.array([[0.,0.,-1.]]),normal[None])[0]
        position=d.xpos[body].copy();rotation=correction*Rotation.from_matrix(d.xmat[body].reshape(3,3))
        world=vertices@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
        flattened=correction.apply(world-position)+position
        dz=-.0005-flattened[:,2].min();position[2]+=dz;flattened[:,2]+=dz
        targets[side]=(position,rotation);sole_centers.append(flattened[flattened[:,2]<-.0004].mean(axis=0))
        tilts[side]=float(correction.magnitude())
    com_target=d.subtree_com[root].copy();com_target[:2]=np.mean(sole_centers,axis=0)[:2]
    rotation0=Rotation.from_quat(q0[[4,5,6,3]]);heading=rotation0.as_euler('xyz')[2]
    root_target=Rotation.from_euler('z',heading)
    if args.symmetric_feet:
        basis=root_target.as_matrix()
        centre=np.mean(sole_centers,axis=0)
        for side,body in feet.items():
            position,rotation=targets[side]
            # Retain the level sole plane, replacing only its heading.
            yaw=rotation.as_euler('xyz')[2]
            rotation=Rotation.from_euler('z',heading-yaw)*rotation
            vertices=[]
            for g in range(m.ngeom):
                if int(m.geom_bodyid[g])!=body or not(m.geom_contype[g] or m.geom_conaffinity[g]):continue
                k=int(m.geom_dataid[g]);a,n=m.mesh_vertadr[k],m.mesh_vertnum[k]
                world=m.mesh_vert[a:a+n]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
                local=Rotation.from_matrix(d.xmat[body].reshape(3,3)).inv().apply(world-d.xpos[body])
                vertices.append(rotation.apply(local))
            offsets=np.concatenate(vertices)@basis
            desired=centre@basis
            current=position@basis
            current[0]=desired[0]+.12-offsets[:,0].max()
            current[1]=desired[1]+(.12 if side=='left' else -.12)-(offsets[:,1].min()+offsets[:,1].max())/2
            current[2]=-.0005-offsets[:,2].min()
            targets[side]=(basis@current,rotation)
        com_target[:2]=centre[:2]
    joints=[m.joint('robot/'+s+'_'+j+'_joint').id for s in feet for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']]
    columns=m.jnt_qposadr[joints];lo=m.jnt_range[joints,0]+.025;hi=m.jnt_range[joints,1]-.025
    lower=np.r_[q0[:2]-.3,.35,np.full(3,-.6),lo];upper=np.r_[q0[:2]+.3,.9,np.full(3,.6),hi]
    def pose(x):
        q=q0.copy();q[:3]=x[:3];q[3:7]=(Rotation.from_rotvec(x[3:6])*rotation0).as_quat()[[3,0,1,2]];q[columns]=x[6:]
        return q
    def residual(x):
        d.qpos[:]=pose(x);mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        r=[]
        for s,b in feet.items():
            pos,rot=targets[s];current=Rotation.from_matrix(d.xmat[b].reshape(3,3))
            r.extend((d.xpos[b]-pos)*(20000 if args.native_balance else 1000));r.extend((rot.inv()*current).as_rotvec()*(4000 if args.native_balance else 200))
        r.extend((d.subtree_com[root]-com_target)*1000)
        r.extend((root_target.inv()*Rotation.from_quat(d.qpos[[4,5,6,3]])).as_rotvec()*(20 if args.native_balance else 200))
        return np.asarray(r)
    x0=np.r_[q0[:3],np.zeros(3),q0[columns]]
    fit=least_squares(residual,np.clip(x0,lower,upper),bounds=(lower,upper),max_nfev=500,ftol=1e-12,xtol=1e-12,gtol=1e-10)
    balance_history=[]
    for iteration in range(16 if args.native_balance else 1):
        q=pose(fit.x);geometry_residual=residual(fit.x)
        # Balance total load by uniform compression, then move COM toward the
        # measured pressure center while keeping foot transforms as IK targets.
        left,right=-.003,.003
        for _ in range(60):
            shift=(left+right)/2;d.qpos[:]=q;d.qpos[2]+=shift;d.qvel[:]=0;d.qacc[:]=0;mujoco.mj_inverse(m,d)
            if d.qfrc_constraint[2]>weight:left=shift
            else:right=shift
        q[2]+=(left+right)/2;d.qpos[:]=q;d.qvel[:]=0;d.qacc[:]=0;mujoco.mj_inverse(m,d)
        force=d.qfrc_inverse.copy();world_torque=Rotation.from_quat(q[[4,5,6,3]]).apply(force[3:6])
        error=float(abs(force[:6]/np.r_[np.full(3,weight),np.full(3,weight*.7)]).max())
        balance_history.append({'iteration':iteration,'root_residual':error,'world_torque':world_torque.tolist()})
        if not args.native_balance or error<1e-5 or iteration==15:break
        com_target[:2]+=np.clip(np.array([world_torque[1],-world_torque[0]])/weight,-.04,.04)*.7
        seed=fit.x.copy();seed[:3]=q[:3]
        fit=least_squares(residual,np.clip(seed,lower,upper),bounds=(lower,upper),max_nfev=300,
                          ftol=1e-12,xtol=1e-12,gtol=1e-10)
    effort=np.array([PROFILE['effort_limit'][n] for n in PROFILE['joint_names']]);force=d.qfrc_inverse.copy()
    item={'state_file':'candidate-000.npz','maximum_motor_ratio':float((abs(force[6:])/effort).max()),
          'hard_violation_rad':float(np.maximum(np.maximum(m.jnt_range[1:,0]-q[7:],q[7:]-m.jnt_range[1:,1]),0).max()),
          'root_force_residual':force[:6].tolist(),'root_force_residual_normalized':float(abs(force[:6]/np.r_[np.full(3,weight),np.full(3,weight*.7)]).max()),
          'geometry_fit_max_residual':float(abs(geometry_residual).max()),'foot_fit_max_residual':float(abs(geometry_residual[:12]).max()),
          'initial_sole_tilt_rad':tilts,'native_balance':args.native_balance,'balance_history':balance_history,
          'static_force_feasible':False,'strict_geometry_static_feasible':False,
          'heights':{},'approved_for_training':False}
    for side in ['left','right','nonfoot']:
        height=[]
        for g in range(m.ngeom):
            if int(m.body_rootid[int(m.geom_bodyid[g])])!=root or not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
            group=next((s for s,b in feet.items() if int(m.geom_bodyid[g])==b),'nonfoot')
            if group!=side:continue
            k=m.geom_dataid[g];a,n=m.mesh_vertadr[k],m.mesh_vertnum[k]
            height.append(float((m.mesh_vert[a:a+n]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])[:,2].min()))
        item['heights'][side]=min(height)
    normal={s:0. for s in feet}
    for i,c in enumerate(d.contact[:d.ncon]):
        bodies=[int(m.geom_bodyid[int(g)]) for g in c.geom];reaction=np.zeros(6);mujoco.mj_contactForce(m,d,i,reaction)
        for side,body in feet.items():
            if body in bodies and any(int(m.body_rootid[b])!=root for b in bodies):normal[side]+=float(max(0.,reaction[0]))
    item['normal_force_n']=normal
    item['root_up_z']=float(d.xmat[root].reshape(3,3)[2,2])
    item['final_sole_tilt_rad']={s:float(np.arccos(np.clip(-(Rotation.from_matrix(d.xmat[b].reshape(3,3))*targets[s][1].inv()).apply([0.,0.,-1.])[2],-1,1))) for s,b in feet.items()}
    static_pass=bool(item['root_force_residual_normalized']<1e-5 and item['maximum_motor_ratio']<=1
                     and item['hard_violation_rad']<1e-9 and min(normal.values())>10 and item['root_up_z']>.85
                     and min(item['heights'][s] for s in feet)>-.005 and max(item['heights'][s] for s in feet)<.005
                     and item['heights']['nonfoot']>.005 and max(item['final_sole_tilt_rad'].values())<.005)
    item['static_force_feasible']=static_pass;item['strict_geometry_static_feasible']=static_pass
    for name in ['candidate-000.npz','equilibrium.npz']:
        np.savez(args.out/name,qpos=q,qvel=np.zeros(m.nv),required_motor_torque=force[6:],joint_names=np.array(PROFILE['joint_names']))
    (args.out/'initial.npz').write_bytes((args.out/'equilibrium.npz').read_bytes())
    (args.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    (args.out/'result.json').write_text(json.dumps({'best':item,'trials':[item],'approved_for_training':False},indent=2))
    print(json.dumps(item),flush=True)


if __name__=='__main__':main()
