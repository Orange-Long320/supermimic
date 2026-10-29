"""Prepare a symmetric, slightly flexed X2 stance for native calibration.

Server only; generates an initial pose, never an approved reference.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--template',type=Path,required=True)
    p.add_argument('--support-height-m',type=float,required=True)
    p.add_argument('--knee-deg',type=float,default=12.)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists() or not 8<=a.knee_deg<=20:p.error('New output and knee flexion 8–20 degrees required')
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra',check_assets=True)
    from supermimic.robots.dimensions import PROFILE
    m=mujoco.MjModel.from_binary_path(str(a.model));d=mujoco.MjData(m)
    q=np.load(a.template)['qpos'].copy();d.qpos[:]=q;mujoco.mj_forward(m,d)
    root=int(m.jnt_bodyid[0]);r=d.xmat[root].reshape(3,3);heading=np.arctan2(r[1,0],r[0,0])
    q[3:7]=Rotation.from_euler('z',heading).as_quat()[[3,0,1,2]]
    names=[m.joint(j).name.rsplit('/',1)[-1] for j in range(1,m.njnt)]
    if names!=PROFILE['joint_names']:raise ValueError('Joint order mismatch')
    for j,name in enumerate(names,1):
        value=PROFILE['default_joint_pos'][name]
        if any(k in name for k in ['hip_','ankle_','waist_']):value=0.
        if 'hip_pitch' in name or 'ankle_pitch' in name:value=-np.radians(a.knee_deg)/2
        if 'knee' in name:value=np.radians(a.knee_deg)
        mid=m.jnt_range[j].mean();half=np.diff(m.jnt_range[j])[0]*.45
        if not mid-half-1e-9<=value<=mid+half+1e-9:raise ValueError(f'Initial {name} outside original reset bounds')
        q[m.jnt_qposadr[j]]=value
    d.qpos[:]=q;mujoco.mj_forward(m,d)
    bottoms=[]
    for side in ['left','right']:
        body=m.body('robot/'+side+'_ankle_roll_link').id
        for g in range(m.ngeom):
            if m.geom_bodyid[g]!=body or not(m.geom_contype[g] or m.geom_conaffinity[g]):continue
            mesh=m.geom_dataid[g];start,n=m.mesh_vertadr[mesh],m.mesh_vertnum[mesh]
            v=m.mesh_vert[start:start+n]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
            bottoms.append(float(v[:,2].min()))
    q[2]+=a.support_height_m-.0005-min(bottoms)
    a.out.mkdir(parents=True)
    np.savez_compressed(a.out/'initial.npz',state=np.r_[q,np.zeros(m.nv)][None,:])
    np.savez_compressed(a.out/'seed.npz',qpos=q)
    (a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    (a.out/'provenance.json').write_text(json.dumps({'configuration':{k:str(v) for k,v in vars(a).items()},
        'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),
        'template_sha256':hashlib.sha256(a.template.read_bytes()).hexdigest(),
        'approved_for_training':False},indent=2)+'\n')


if __name__=='__main__':main()
