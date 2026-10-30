"""Measure actual and reference foot heading in an official simulator rollout."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial',type=Path,required=True)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new output required')
    rows=[json.loads(s) for s in (a.trial/'frames.jsonl').open()]
    if len(rows)<2:raise ValueError('not enough frames')
    report=json.loads((a.trial/'result.json').read_text());spec=json.loads(a.profile.read_text());names=spec['joint_names'];n=len(names)
    model=mujoco.MjModel.from_xml_path(str(a.scene));data=mujoco.MjData(model)
    ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,k) for k in names]
    if any(i<0 for i in ids):raise ValueError('missing joint')
    adr=[model.jnt_qposadr[i] for i in ids];feet=[model.body(s+'_ankle_roll_link').id for s in ('left','right')]
    time=np.array([r['elapsed'] for r in rows]);poses=np.array([r['root'] for r in rows]);footyaw=[];tilt=[];gaps=[]
    spheres=[[i for i in range(model.ngeom) if model.geom_bodyid[i]==body and model.geom_type[i]==mujoco.mjtGeom.mjGEOM_SPHERE and model.geom_contype[i]] for body in feet]
    for r in rows:
        data.qpos[:7]=np.r_[r['root'],r['q']];data.qpos[adr]=r['joint_pos'];mujoco.mj_kinematics(model,data)
        rot=Rotation.from_quat(np.asarray(r['q'])[[1,2,3,0]]);heading=rot.as_euler('xyz')[2]
        y=[(Rotation.from_matrix(data.xmat[b].reshape(3,3)).as_euler('xyz')[2]-heading+np.pi)%(2*np.pi)-np.pi for b in feet]
        footyaw.append(np.degrees(y));tilt.append(np.degrees(np.arccos(np.clip(rot.as_matrix()[2,2],-1,1))))
        gaps.append([float(np.min(data.geom_xpos[g,2]-model.geom_size[g,0])) for g in spheres])
    footyaw=np.array(footyaw);tilt=np.array(tilt);gaps=np.array(gaps)
    mask=(time>=max(time[0],time[-1]-10)) & np.array([r['mode']=='stand' for r in rows])
    if not mask.any():raise ValueError('no final stand samples')
    fy=footyaw[mask];gt=gaps[mask];idx=np.flatnonzero(mask);stand_xy=poses[idx,:2];delta=np.linalg.norm(stand_xy-stand_xy[0],axis=1)
    modes=np.array([r['mode'] for r in rows]); releases=np.flatnonzero((modes[:-1]=='jog') & (modes[1:]=='stand'))+1
    per_stop=[]
    for i in releases:
        selected=(time>=time[i]+4.) & (time<time[i]+5.) & (modes=='stand')
        if not selected.any():continue
        per_stop.append({'release_s':float(time[i]),'after_4_to_5_s_mean_foot_yaw_deg':footyaw[selected].mean(axis=0).tolist(),
                         'after_4_to_5_s_heading_difference_p95_deg':float(np.percentile(abs(footyaw[selected,0]-footyaw[selected,1]),95)),
                         'after_4_to_5_s_root_tilt_max_deg':float(tilt[selected].max())})
    out={'frames':len(rows),'seconds':float(time[-1]-time[0]),'failure':report['failure'],'window_s':float(time[idx[-1]]-time[idx[0]]),
         'foot_yaw_mean_deg':fy.mean(axis=0).tolist(),'foot_yaw_abs_p95_deg':np.percentile(abs(fy),95,axis=0).tolist(),
         'foot_yaw_difference_abs_p95_deg':float(np.percentile(abs(fy[:,0]-fy[:,1]),95)),
         'root_tilt_p95_deg':float(np.percentile(tilt[mask],95)), 'root_tilt_max_deg':float(tilt[mask].max()),
         'stand_net_displacement_m':float(delta[-1]),'stand_max_displacement_m':float(delta.max()),
         'foot_gap_min_m':gt.min(axis=0).tolist(),'both_feet_near_ground_fraction':float(np.mean(np.all(gt<.005,axis=1))),
         'max_future_overlap_error':report.get('max_future_overlap_error'),
         'stops':per_stop,
         'scope':'one official-simulator rollout; kinematic foot gaps, not measured contact force'}
    a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))


if __name__=='__main__':main()
