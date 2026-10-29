"""Server seed retaining the balanced crouch while aligning both lower legs."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['model','template','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--support-height-m',type=float,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('New output required')
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation
    m=mujoco.MjModel.from_binary_path(str(a.model));d=mujoco.MjData(m)
    original=np.load(a.template)['qpos'].copy();q=original.copy()
    def col(side,joint):return int(m.jnt_qposadr[m.joint('robot/'+side+'_'+joint+'_joint').id])
    for joint in ['hip_pitch','knee']:
        value=float(np.mean([q[col(s,joint)] for s in ['left','right']]))
        for side in ['left','right']:q[col(side,joint)]=value
    for side in ['left','right']:
        for joint in ['hip_roll','hip_yaw','ankle_roll']:q[col(side,joint)]=0.
        q[col(side,'ankle_pitch')]=-q[col(side,'hip_pitch')]-q[col(side,'knee')]
    d.qpos[:]=original;mujoco.mj_forward(m,d);root=int(m.jnt_bodyid[0]);axis=d.xmat[root].reshape(3,3)[:,0]
    q[3:7]=Rotation.from_euler('z',np.arctan2(axis[1],axis[0])).as_quat()[[3,0,1,2]]
    d.qpos[:]=q;mujoco.mj_forward(m,d);bottoms=[]
    for side in ['left','right']:
        body=m.body('robot/'+side+'_ankle_roll_link').id
        for g in range(m.ngeom):
            if m.geom_bodyid[g]!=body or not(m.geom_contype[g] or m.geom_conaffinity[g]):continue
            mesh=m.geom_dataid[g];start,n=m.mesh_vertadr[mesh],m.mesh_vertnum[mesh]
            v=m.mesh_vert[start:start+n]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g];bottoms.append(float(v[:,2].min()))
    q[2]+=a.support_height_m-.0005-min(bottoms)
    a.out.mkdir(parents=True);np.savez_compressed(a.out/'initial.npz',state=np.r_[q,np.zeros(m.nv)][None,:]);np.savez_compressed(a.out/'seed.npz',qpos=q)
    (a.out/'as-run.py').write_bytes(Path(__file__).read_bytes());(a.out/'configuration.json').write_text(json.dumps({k:str(v) for k,v in vars(a).items()},indent=2))


if __name__=='__main__':main()
