"""Kinematic screening of a staged stop; this is not physics validation."""
import argparse,json
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);a=p.parse_args()
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.robots.dimensions import PROFILE
    folder=a.reference;meta=json.loads((folder/'manifest.json').read_text());z=np.load(folder/'natural-stop.npz');q=z['joint_pos'];v=z['joint_vel'];a0=meta['brake_start_frame'];b=meta['standing_frame'];fps=float(z['fps'][0]);idx=list(z['body_names']).index('pelvis');quat=z['body_quat_w'][:,idx];yaw=np.unwrap(Rotation.from_quat(quat[:,[1,2,3,0]]).as_euler('xyz')[:,2]);m=load_g1_model();d=mujoco.MjData(m);h=[]
    for i in range(a0,b+1):
        d.qpos[:]=np.r_[z['body_pos_w'][i,idx],quat[i],q[i]];mujoco.mj_kinematics(m,d);g=[]
        for side in ('left','right'):
            body=m.body(side+'_ankle_roll_link').id;values=[]
            for j in range(m.ngeom):
                if m.geom_bodyid[j]!=body or not(m.geom_contype[j] or m.geom_conaffinity[j]) or m.geom_type[j]!=mujoco.mjtGeom.mjGEOM_MESH:continue
                k=m.geom_dataid[j];verts=m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k]+m.mesh_vertnum[k]];values.append(float((verts@d.geom_xmat[j].reshape(3,3).T+d.geom_xpos[j])[:,2].min()))
            g.append(min(values))
        h.append(g)
    lim=m.jnt_range[1:];ex=np.maximum(np.maximum(lim[:,0]-q,q-lim[:,1]),0);limits=np.array([PROFILE['velocity_limit'][n] for n in PROFILE['joint_names']]);dv=np.linalg.norm(v[a0+1]-v[a0-1])/np.sqrt(q.shape[1]);dq=np.diff(q,axis=0)
    out={'brake_seconds':(b-a0)/fps,'entry_joint_velocity_change_rms':float(dv),'brake_max_speed_ratio':float((abs(v[a0:b+1])/limits).max()),'brake_max_hard_excess_rad':float(ex[a0:b+1].max()),'brake_max_step_rms_rad':float(np.sqrt((dq[a0:b]**2).mean(axis=1)).max()),'brake_min_foot_height_m':np.min(h,axis=0).tolist(),'brake_yaw_range_deg':float(np.degrees(np.ptp(yaw[a0:b+1]))),'max_ankle_target_error_m':meta['max_ankle_target_error_m'],'scope':'Kinematic screen only; no contact force or dynamics proof'}
    with (folder/'speed-audit.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
