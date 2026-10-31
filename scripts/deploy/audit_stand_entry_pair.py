"""Compare official-simulator stand traces; no force/contact claims from geometry."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('direct','post-jog','profile','scene','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--window-seconds',type=float,default=60.)
    a=p.parse_args()
    if a.out.exists() or a.window_seconds<=0:raise ValueError('New output and positive window required')
    spec=json.loads(a.profile.read_text());names=spec['joint_names'];n=len(names)
    model=mujoco.MjModel.from_xml_path(str(a.scene)); data=mujoco.MjData(model)
    joint=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name) for name in names]
    if min(joint)<0:raise ValueError('Missing joint')
    qadr=[model.jnt_qposadr[x] for x in joint]
    feet=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,s+'_ankle_roll_link') for s in ('left','right')]
    geom=[[g for g in range(model.ngeom) if model.geom_bodyid[g]==b and
           model.geom_type[g]==mujoco.mjtGeom.mjGEOM_SPHERE and model.geom_contype[g]] for b in feet]
    if min(feet)<0 or any(not g for g in geom):raise ValueError('Missing official foot geometry')
    motor_by_joint={int(j):i for i,j in enumerate(model.actuator_trnid[:,0])}
    if any(j not in motor_by_joint for j in joint):raise ValueError('Missing motor mapping')
    motors=[motor_by_joint[j] for j in joint]
    limits=np.max(np.abs(model.actuator_ctrlrange[motors]),axis=1)
    kp=np.array([spec['stiffness'][name] for name in names]);kd=np.array([spec['damping'][name] for name in names])
    default=np.array([spec['default_joint_pos'][name] for name in names])
    outputs={}
    for name,path in (('direct',a.direct),('post_jog',a.post_jog)):
        result=json.loads((path/'result.json').read_text())
        rows=[json.loads(line) for line in (path/'frames.jsonl').open()]
        if len(rows)<2:raise ValueError('Too few frames: '+name)
        t=np.array([r['elapsed'] for r in rows]);mode=np.array([r['mode'] for r in rows])
        if np.any(np.diff(t)<=0):raise ValueError('Nonmonotonic time')
        if name=='direct':
            if np.any(mode!='stand'):raise ValueError('Direct run contains movement')
            release=t[0]
        else:
            releases=np.flatnonzero((mode[:-1]=='jog') & (mode[1:]=='stand'))+1
            if len(releases)!=1:raise ValueError('Expected exactly one jog release')
            release=t[releases[0]]
        begin=release+5.;end=min(begin+a.window_seconds,t[-1])
        mask=(t>=begin)&(t<=end)&(mode=='stand');idx=np.flatnonzero(mask)
        if len(idx)<2:raise ValueError('No hold samples: '+name)
        pos=np.array([r['root'] for r in rows]);q=np.array([r['q'] for r in rows]);jpos=np.array([r['joint_pos'] for r in rows]);jvel=np.array([r['joint_vel'] for r in rows]);targets=np.array([r['target'] for r in rows])
        rot=Rotation.from_quat(q[:,[1,2,3,0]]);yaw=np.unwrap(rot.as_euler('xyz')[:,2]);tilt=np.degrees(np.arccos(np.clip(rot.as_matrix()[:,2,2],-1,1)))
        gap=np.empty((len(idx),2));ankle=np.empty((len(idx),2,2));toe=np.empty(len(idx))
        for k,i in enumerate(idx):
            data.qpos[:7]=np.r_[pos[i],q[i]];data.qpos[qadr]=jpos[i];mujoco.mj_kinematics(model,data)
            ankle[k]=data.xpos[feet,:2];forward=np.array([np.cos(yaw[i]),np.sin(yaw[i])]);toes=[]
            for side,g in enumerate(geom):
                gap[k,side]=np.min(data.geom_xpos[g,2]-model.geom_size[g,0])
                toes.append(np.max(data.geom_xpos[g,:2]@forward+model.geom_size[g,0]))
            toe[k]=toes[0]-toes[1]
        speed=np.linalg.norm(np.diff(ankle,axis=0),axis=2)/np.diff(t[idx])[:,None]
        near=(gap[1:]<.005)&(gap[:-1]<.005)
        near_speed=[float(np.percentile(speed[:,s][near[:,s]],95)) if np.any(near[:,s]) else None for s in (0,1)]
        dist=np.linalg.norm(pos[idx,:2]-pos[idx[0],:2],axis=1)
        omega=np.array([r['body_angular_velocity'] for r in rows])[idx]
        estimated=np.clip(kp*(targets[idx]-jpos[idx])-kd*jvel[idx],-limits,limits)
        efforts=[r.get('joint_effort') for r in rows if r.get('joint_effort') is not None]
        goal=np.array([r['goal_current'][:n] for r in rows])[idx]+default
        outputs[name]={'failure':result['failure'],'control_seconds':result['control_seconds'],
                       'resets_after_start':result['resets_after_start'],'release_time_s':float(release),
                       'window_start_s':float(t[idx[0]]),'window_duration_s':float(t[idx[-1]]-t[idx[0]]),
                       'window_requested_s':a.window_seconds,'full_window':bool(t[idx[-1]]>=begin+a.window_seconds-.05),
                       'net_drift_m':float(dist[-1]),'max_drift_m':float(dist.max()),
                       'root_tilt_p95_deg':float(np.percentile(tilt[idx],95)),'root_tilt_max_deg':float(tilt[idx].max()),
                       'root_height_min_m':float(pos[idx,2].min()),
                       'imu_angular_speed_p95_rad_s':float(np.percentile(np.linalg.norm(omega,axis=1),95)),
                       'toe_offset_abs_p95_m':float(np.percentile(np.abs(toe),95)),
                       'near_ground_ankle_speed_p95_m_s':near_speed,
                       'both_feet_geometrically_near_fraction':float(np.mean(np.all(gap<.005,axis=1))),
                       'min_foot_gap_m':gap.min(axis=0).tolist(),
                       'reported_joint_effort_max_abs':float(np.max(np.abs(efforts))) if efforts else None,
                       'estimated_pd_peak_effort_ratio':float(np.max(np.abs(estimated)/limits)),
                       'standing_joint_error_rms_p95_rad':float(np.percentile(np.sqrt(np.mean((jpos[idx]-goal)**2,axis=1)),95)),
                       'max_future_overlap_error':result['max_future_overlap_error']}
    out={'scenarios':outputs,'scope':'official simulator; same initial pose and controller; near-ground is geometric, estimated PD is not measured torque',
         'contact_forces':'not available in recorded official simulator ROS stream',
         'measurement_caveat':'joint effort field may be unpopulated; do not infer measured zero force or load'}
    a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))


if __name__=='__main__':main()
