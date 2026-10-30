"""Summarize recorded physical rollout, including offline sole clearance proxy."""
import argparse,json
from pathlib import Path
import mujoco,numpy as np
from scipy.spatial.transform import Rotation


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    from supermimic.data.motion_convert import load_g1_model
    m=load_g1_model();d=mujoco.MjData(m);z=np.load(a.run/'trajectory.npz');states=z['states'];ph=z['phases'];frames=z['frames'];goals=z['goals'];r=json.loads((a.run/'result.json').read_text())
    nq=m.nq;assert states.shape[1]==nq+m.nv
    geom=[]
    for side in ('left','right'):
        group=[]
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g])==m.body(side+'_ankle_roll_link').id and (m.geom_contype[g] or m.geom_conaffinity[g]) and m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
                mid=m.geom_dataid[g];group.append((g,m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]].copy()))
        if not group:raise ValueError('Missing foot meshes')
        geom.append(group)
    clear=[]
    for state in states:
        d.qpos[:]=state[:nq];mujoco.mj_kinematics(m,d)
        clear.append([min(float((v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])[:,2].min()) for g,v in group) for group in geom])
    clear=np.asarray(clear);stand=states[ph=='stand'];sc=clear[ph=='stand'];q=states[:,7:nq];lower,upper=m.jnt_range[1:].T
    result={'steps':len(states),'dt_s':.02,'failure':r['failure'],'state_resets':r['state_resets_after_initialization'],
       'max_hard_joint_violation_rad':float(np.maximum(np.maximum(lower-q,q-upper),0).max()),
       'min_sole_height_m':clear.min(axis=0).tolist(),'clearance_scope':'offline geometry at 50 Hz, not actual contact force',
       'stand_seconds':len(stand)*.02}
    if len(stand):
        yaw=np.unwrap(Rotation.from_quat(stand[:,[4,5,6,3]]).as_euler('xyz')[:,2]);
        result.update(stand_mean_planar_speed_m_s=float(np.linalg.norm(stand[:,nq:nq+2],axis=1).mean()),
            stand_mean_abs_yaw_rate_rad_s=float(np.abs(stand[:,nq+5]).mean()),
            stand_joint_velocity_rms_rad_s=float(np.sqrt((stand[:,nq+6:]**2).mean())),
            stand_net_displacement_m=float(np.linalg.norm(stand[-1,:2]-stand[0,:2])),
            stand_max_displacement_m=float(np.linalg.norm(stand[:,:2]-stand[0,:2],axis=1).max()),
            stand_heading_change_rad=float(yaw[-1]-yaw[0]),
            stand_feet_near_ground_fraction=(sc<=.005).mean(axis=0).tolist())
    idx=np.flatnonzero(frames==r['release_frame'])
    if idx.size and idx[0]>0:
        i=idx[0];k=r['future_frames'];size=(goals.shape[1]-9)//k if goals.shape[1]%k else goals.shape[1]//k
        x=goals[i-1,:k*size].reshape(k,size);y=goals[i,:k*size].reshape(k,size)
        result['release_overlapping_future_max_abs_difference']=float(abs(x[1:]-y[:-1]).max())
    out=a.run/'summary.json'
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
