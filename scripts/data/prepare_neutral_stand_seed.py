"""Seed a symmetric, arms-down stance for the native contact calibration."""
import argparse,json
from pathlib import Path
import mujoco,numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--standing',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--knee-rad',type=float,default=.35);a=p.parse_args()
    if a.out.exists():p.error('Output exists')
    from supermimic.robots.dimensions import PROFILE
    m=mujoco.MjModel.from_binary_path(str(a.model));d=mujoco.MjData(m)
    with np.load(a.standing,allow_pickle=False) as z:
        root=list(z['body_names']).index('pelvis');q=np.r_[z['body_pos_w'][0,root],z['body_quat_w'][0,root],z['joint_pos'][0]].astype(float)
    names=list(PROFILE['joint_names']);q[:2]=0;q[3:7]=[1,0,0,0]
    for i,n in enumerate(names):
        if any(x in n for x in ('shoulder','elbow','wrist','waist','head')):q[7+i]=PROFILE['default_joint_pos'][n]
        for side in ('left','right'):
            if n==side+'_hip_pitch_joint':q[7+i]=-a.knee_rad/2
            if n==side+'_knee_joint':q[7+i]=a.knee_rad
            if n==side+'_ankle_pitch_joint':q[7+i]=-a.knee_rad/2
            if n in (side+'_hip_roll_joint',side+'_hip_yaw_joint',side+'_ankle_roll_joint'):q[7+i]=0.
    d.qpos[:]=q;mujoco.mj_kinematics(m,d);height=[]
    for side in ('left','right'):
        body=m.body('robot/'+side+'_ankle_roll_link').id;values=[]
        for g in range(m.ngeom):
            if m.geom_bodyid[g]!=body or not (m.geom_contype[g] or m.geom_conaffinity[g]) or m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:continue
            mid=m.geom_dataid[g];v=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
            values.append(float((v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])[:,2].min()))
        height.append(min(values))
    q[2]-=min(height)+.0005
    a.out.mkdir(parents=True);np.savez(a.out/'initial.npz',state=np.r_[q,np.zeros(m.nv)][None],qpos=q,qvel=np.zeros(m.nv))
    (a.out/'seed.json').write_text(json.dumps({'knee_rad':a.knee_rad,'source_stand':str(a.standing),'pose':'symmetric knees and default arms; unvalidated','sole_seed_heights':height,'root_height':q[2]},indent=2)+'\n')
    print((a.out/'seed.json').read_text())
if __name__=='__main__':main()
