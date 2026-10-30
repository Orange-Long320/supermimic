"""Construct an exploratory symmetric X2 gait standing pose in official geometry."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True,help='Existing stand clip')
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists(): raise ValueError('new output required')
    spec=json.loads(a.profile.read_text()); names=spec['joint_names']
    with np.load(a.source,allow_pickle=False) as z:
        if list(z['joint_names'])!=names: raise ValueError('joint order mismatch')
        root=list(z['body_names']).index('pelvis')
        q=np.r_[z['body_pos_w'][-1,root],z['body_quat_w'][-1,root],z['joint_pos'][-1]].astype(float)
    before=q.copy(); idx={name:7+i for i,name in enumerate(names)}
    knee=np.mean([q[idx[s+'_knee_joint']] for s in ('left','right')])
    for side in ('left','right'):
        for name in ('hip_roll','hip_yaw','ankle_roll'): q[idx[side+'_'+name+'_joint']]=0.
        q[idx[side+'_hip_pitch_joint']]=-knee/2
        q[idx[side+'_knee_joint']]=knee
        q[idx[side+'_ankle_pitch_joint']]=-knee/2
    for name in ('waist_yaw_joint','waist_pitch_joint','waist_roll_joint'):q[idx[name]]=0.
    q[:2]=0.;q[3:7]=[1.,0.,0.,0.]
    model=mujoco.MjModel.from_xml_path(str(a.scene));data=mujoco.MjData(model)
    ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name) for name in names]
    if any(i<0 for i in ids):raise ValueError('missing official joint')
    adr=[model.jnt_qposadr[i] for i in ids]; feet=[]
    for side in ('left','right'):
        body=model.body(side+'_ankle_roll_link').id
        geom=[i for i in range(model.ngeom) if model.geom_bodyid[i]==body and model.geom_type[i]==mujoco.mjtGeom.mjGEOM_SPHERE and model.geom_contype[i]]
        if not geom:raise ValueError('missing official foot spheres')
        feet.append((body,geom))
    def update():
        data.qpos[:7]=q[:7];data.qpos[adr]=q[7:];mujoco.mj_forward(model,data)
    update();gap=min(float(np.min(data.geom_xpos[gg,2]-model.geom_size[gg,0])) for _,gg in feet)
    q[2]-=gap;update()
    yaws=[float(Rotation.from_matrix(data.xmat[body].reshape(3,3)).as_euler('xyz',degrees=True)[2]) for body,_ in feet]
    gaps=[float(np.min(data.geom_xpos[gg,2]-model.geom_size[gg,0])) for _,gg in feet]
    points=np.vstack([data.geom_xpos[gg,:2] for _,gg in feet]);hull=ConvexHull(points);com=data.subtree_com[model.body('pelvis').id][:2]
    margin=float(np.min(-(hull.equations[:,:2]@com+hull.equations[:,2])/np.linalg.norm(hull.equations[:,:2],axis=1)))
    result={'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(), 'foot_yaw_deg':yaws,
            'foot_gap_m':gaps,'foot_center_xy':data.xpos[[body for body,_ in feet],:2].tolist(),
            'com_xy':com.tolist(),'com_inside_sphere_centres_margin_m':margin,'pelvis_height_m':float(q[2]),
            'changed_joint_names':[name for name in names if abs(q[idx[name]]-before[idx[name]])>1e-8],
            'scope':'kinematic candidate; no dynamic or force validation'}
    if max(abs(x) for x in yaws)>1e-4 or margin<=0 or max(abs(x) for x in gaps)>.003:
        raise ValueError('candidate does not meet geometric stance criteria: '+str(result))
    a.out.mkdir(parents=True); np.savez(a.out/'pose.npz',qpos=q)
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
