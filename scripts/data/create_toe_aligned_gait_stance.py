"""Adjust an existing X2 standing pose so left and right toes finish level."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new output required')
    spec=json.loads(a.profile.read_text());names=spec['joint_names']
    with np.load(a.source,allow_pickle=False) as z:
        if list(z['joint_names'])!=names:raise ValueError('joint order mismatch')
        root=list(z['body_names']).index('pelvis')
        q=np.r_[z['body_pos_w'][-1,root],z['body_quat_w'][-1,root],z['joint_pos'][-1]].astype(float)
    roll,pitch,_=Rotation.from_quat(q[[4,5,6,3]]).as_euler('xyz')
    q[:2]=0.;q[3:7]=Rotation.from_euler('xyz',[roll,pitch,0]).as_quat()[[3,0,1,2]]
    model=mujoco.MjModel.from_xml_path(str(a.scene));data=mujoco.MjData(model)
    ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name) for name in names]
    if any(i<0 for i in ids):raise ValueError('missing joint')
    adr=[model.jnt_qposadr[i] for i in ids]
    leg=[i for i,name in enumerate(names) if any(x in name for x in ('hip_','knee_','ankle_'))]
    feet=[]
    for side in ('left','right'):
        body=model.body(side+'_ankle_roll_link').id
        spheres=[g for g in range(model.ngeom) if model.geom_bodyid[g]==body and model.geom_type[g]==mujoco.mjtGeom.mjGEOM_SPHERE and model.geom_contype[g]]
        if not spheres:raise ValueError('missing foot spheres')
        feet.append((body,spheres))
    def measure(candidate):
        data.qpos[:7]=candidate[:7];data.qpos[adr]=candidate[7:];mujoco.mj_forward(model,data)
        front=[];gap=[];yaw=[]
        for body,spheres in feet:
            front.append(float(np.max(data.geom_xpos[spheres,0]+model.geom_size[spheres,0])))
            gap.append(float(np.min(data.geom_xpos[spheres,2]-model.geom_size[spheres,0])))
            yaw.append(float(Rotation.from_matrix(data.xmat[body].reshape(3,3)).as_euler('xyz')[2]))
        return np.array(front),np.array(gap),np.array(yaw)
    front0,gap0,yaw0=measure(q);center0=data.xpos[[b for b,_ in feet],:2].copy()
    x0=np.r_[q[2],q[7+np.array(leg)]]
    legids=np.array(ids)[leg];lo=np.maximum(model.jnt_range[legids,0]+1e-6,x0[1:]-.25);hi=np.minimum(model.jnt_range[legids,1]-1e-6,x0[1:]+.25)
    if np.any(lo>=hi):raise ValueError('invalid leg limits')
    def pose(x):
        v=q.copy();v[2]=x[0];v[7+np.array(leg)]=x[1:];return v
    def residual(x):
        front,gap,yaw=measure(pose(x));centers=data.xpos[[b for b,_ in feet],:2]
        return np.r_[1000*(front[0]-front[1]),1000*gap,20*(yaw-yaw0),20*(centers[:,1]-center0[:,1]), .3*(x-x0)]
    fit=least_squares(residual,np.clip(x0,np.r_[q[2]-.04,lo],np.r_[q[2]+.04,hi]),bounds=(np.r_[q[2]-.04,lo],np.r_[q[2]+.04,hi]),max_nfev=150)
    result_q=pose(fit.x);front,gap,yaw=measure(result_q)
    points=np.vstack([data.geom_xpos[gg,:2] for _,gg in feet]);hull=ConvexHull(points);com=data.subtree_com[model.body('pelvis').id][:2]
    margin=float(np.min(-(hull.equations[:,:2]@com+hull.equations[:,2])/np.linalg.norm(hull.equations[:,:2],axis=1)))
    report={'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(), 'initial_toe_offset_m':float(front0[0]-front0[1]),
            'toe_front_positions_m':front.tolist(),'toe_offset_m':float(front[0]-front[1]),'foot_gap_m':gap.tolist(),
            'initial_foot_yaw_deg':np.degrees(yaw0).tolist(),'foot_yaw_deg':np.degrees(yaw).tolist(),
            'max_leg_change_rad':float(np.max(abs(fit.x[1:]-x0[1:]))),'com_hull_margin_m':margin,
            'scope':'kinematic stance candidate; no force or policy validation'}
    if not fit.success or abs(front[0]-front[1])>.003 or max(abs(gap))>.003 or margin<=0:
        raise ValueError('candidate failed geometric screen: '+str(report))
    a.out.mkdir(parents=True);np.savez(a.out/'pose.npz',qpos=result_q)
    (a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
