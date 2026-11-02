"""CPU-only offline sensitivity replay; parameters are hypothetical, not hardware calibration.

Reads one simulator recording, applies one sensor/model error per variant, and
scores against untouched odometry. This is not the ROS executor: no DDS, callback
queues or other joint groups are simulated. Assumes pelvis imu_0, flat terrain,
and the collision-sphere support model; only encoder leg samples are used.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


VARIANTS = {
    'baseline': {},
    'tilt_x_0p5deg': {'tilt_deg': .5},
    'yaw_1deg': {'yaw_deg': 1.},
    'yaw_drift_0p1deg_min': {'yaw_drift_deg_min': .1},
    'accel_bias_xyz_0p01': {'acc_bias': [.01, -.01, .01]},
    'accel_noise_sigma_0p02': {'acc_noise': .02},
    'gyro_bias_xyz_0p005': {'gyro_bias': [.005, -.005, .005]},
    'leg_zero_0p01rad': {'joint_offset': .01},
    'leg_scale_1p01': {'joint_scale': 1.01},
    'lever_error_5mm_xz': {'lever_error': [.005, 0., .005]},
    'mount_roll_error_1deg': {'mount_roll_deg': 1.},
    'leg_stale_5ms': {'joint_delay': .005},
    'imu_drop_every_10': {'drop_every': 10},
    'imu_burst_drop_18': {'burst_drop': 18},
}


def read(path):
    with path.open() as f:
        return [json.loads(s) for s in f]


def stamps(rows):
    return np.array([r['stamp'][0]+r['stamp'][1]*1e-9 for r in rows])


def interp(t, values, points):
    return np.column_stack([np.interp(points,t,values[:,i]) for i in range(values.shape[1])])


def score(rows, truth_t, truth_p, window):
    if len(rows) < 2:
        return {'failure':'no usable output', 'coverage':0.}
    t=np.array([r[0] for r in rows]); p=np.array([r[1] for r in rows]); v=np.array([r[2] for r in rows])
    grid=np.arange(max(t[0],truth_t[0]+.05), min(t[-1],truth_t[-1]-.05), .02)
    if len(grid)<2:
        return {'failure':'insufficient overlap','coverage':0.}
    ix=np.clip(np.searchsorted(t,grid,side='right'),1,len(t)-1)
    grid=grid[(t[ix]-t[ix-1])<=.05]
    if not len(grid):
        return {'failure':'no valid intervals','coverage':0.}
    pred=interp(t,p,grid)-interp(t,p,[grid[0]])
    actual=interp(truth_t,truth_p,grid)-interp(truth_t,truth_p,[grid[0]])
    pv=interp(t,v,grid); tv=(interp(truth_t,truth_p,grid+.05)-interp(truth_t,truth_p,grid-.05))/.1
    pe=pred-actual; ve=pv-tv; rmse=lambda x:float(np.sqrt(np.mean(x*x)))
    return {'coverage':float(len(grid)*.02/window),'scored_samples':len(grid),
            'horizontal_position_rmse_m':rmse(np.linalg.norm(pe[:,:2],axis=1)),
            'vertical_position_rmse_m':rmse(pe[:,2]),
            'horizontal_velocity_rmse_m_s':rmse(np.linalg.norm(ve[:,:2],axis=1)),
            'vertical_velocity_rmse_m_s':rmse(ve[:,2]),
            'max_horizontal_position_error_m':float(np.max(np.linalg.norm(pe[:,:2],axis=1))),
            'max_vertical_position_error_m':float(np.max(np.abs(pe[:,2]))),
            'max_horizontal_velocity_error_m_s':float(np.max(np.linalg.norm(ve[:,:2],axis=1))),
            'max_vertical_velocity_error_m_s':float(np.max(np.abs(ve[:,2]))),
            'final_position_error_m':pe[-1].tolist(),
            'final_velocity_error_m_s':ve[-1].tolist(),
            'max_output_gap_s':float(np.max(np.diff(t)))}


def replay(spec, imu_t, rotations, gyro, acc, jt, joint_pos, joint_vel, names, truth_t, truth_p, xml, support_cls, duration, seed):
    support=support_cls(xml); rng=np.random.default_rng(seed)
    selected=np.arange(len(imu_t))
    if spec.get('drop_every'):
        selected=selected[(selected+1)%spec['drop_every'] != 0]
    if spec.get('burst_drop'):
        first=int(np.searchsorted(imu_t,imu_t[0]+15.))
        selected=selected[(selected<first)|(selected>=first+spec['burst_drop'])]
    tilt=Rotation.from_euler('x',spec.get('tilt_deg',0.),degrees=True).as_matrix()
    yaw0=spec.get('yaw_deg',0.); drift=spec.get('yaw_drift_deg_min',0.)
    if not drift:
        correction=Rotation.from_euler('z',yaw0,degrees=True).as_matrix()@tilt
        measured_rot=np.einsum('ij,njk->nik',correction,rotations)
    else:
        angles=np.deg2rad(yaw0+drift*(imu_t-imu_t[0])/60.)
        correction=Rotation.from_euler('z',angles[:,None]).as_matrix()
        measured_rot=np.einsum('nij,njk->nik',correction,np.einsum('ij,njk->nik',tilt,rotations))
    measured_acc=acc+np.asarray(spec.get('acc_bias',[0.,0.,0.]))
    if spec.get('acc_noise'):
        measured_acc=measured_acc+rng.normal(0.,spec['acc_noise'],size=acc.shape)
    measured_gyro=gyro+np.asarray(spec.get('gyro_bias',[0.,0.,0.]))
    lever=np.array([.0239465,-.000228682,.04171])+np.asarray(spec.get('lever_error',[0.,0.,0.]))
    mount_error=Rotation.from_euler('x',spec.get('mount_roll_deg',0.),degrees=True).as_matrix()
    position=np.zeros(3); velocity=np.zeros(3); origin=None; first_p=None
    last=None; last_try=None; last_accept=None; initialized=False; counts=Counter(); rows=[]
    failure=None; failure_time=None; next_output=None
    for i in selected:
        t=imu_t[i]; rwi=measured_rot[i]; r=rwi@mount_error
        w=mount_error.T@measured_gyro[i]
        world=rwi@measured_acc[i]-np.array([0.,0.,9.81])
        if last is not None:
            dt=t-last
            if not 0.<dt<=.03:
                failure='imu_timestamp_gap'; failure_time=float(t-imu_t[0]); break
            position += velocity*dt+.5*world*dt*dt
            velocity += world*dt
        last=t
        if last_try is None or t-last_try>=.02:
            last_try=t; delay=spec.get('joint_delay',0.); ji=np.searchsorted(jt,t-delay,side='right')-1
            if ji<0:
                counts['missing']+=1
            else:
                lag=t-delay-jt[ji]
                if lag>.01:
                    counts['stale']+=1
                else:
                    scale=spec.get('joint_scale',1.); offset=spec.get('joint_offset',0.)
                    q=(joint_pos[ji]+lag*joint_vel[ji])*scale+offset; qd=joint_vel[ji]*scale
                    qdict=dict(zip(names,q)); qddict=dict(zip(names,qd))
                    pelvis_v,reason=support.observe(t,r,w,qdict,qddict); counts[reason]+=1
                    if pelvis_v is not None:
                        site_v=pelvis_v+r@np.cross(w,lever)
                        site_h=support.accepted_relative_height+(r@lever)[2]
                        if not initialized:
                            position[:]=0.; origin=site_h; initialized=True
                        position[2]=site_h-origin
                        velocity=site_v; last_accept=t
        if next_output is None: next_output=t
        if t>=next_output:
            next_output=t+.02
            if not initialized: counts['waiting']+=1; continue
            if t-last_accept>.1: counts['output_stale']+=1; continue
            pelvis_p=position-r@lever
            if first_p is None: first_p=pelvis_p.copy()
            pelvis_v=velocity-r@np.cross(w,lever)
            rows.append((t,(pelvis_p-first_p).copy(),pelvis_v.copy()))
    result=score(rows,truth_t,truth_p,duration);result.update(counters=dict(counts),failure=failure,
            failure_time_s=failure_time,output_packets=len(rows))
    return result, rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--src',type=Path,required=True)
    p.add_argument('--xml',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--duration',type=float,default=54.);p.add_argument('--seed',type=int,default=20261102)
    p.add_argument('--variants',nargs='+',choices=tuple(VARIANTS),default=list(VARIANTS))
    a=p.parse_args();sys.path.insert(0,str(a.src));from supermimic.deploy.sim_foot_support import SimFootSupport
    a.out.mkdir(parents=True,exist_ok=False)
    imu=read(a.run/'trial-record/imu.jsonl'); joint=[x for x in read(a.run/'trial-record/joints.jsonl') if x['group']=='leg']; truth=read(a.run/'trial-record/odom.jsonl')
    ti,tj,tt=map(stamps,(imu,joint,truth)); truth_p=np.array([x['position'] for x in truth]); start=max(ti[0],tt[0]+.1,tj[0]); end=min(start+a.duration,ti[-1],tt[-1]-.1)
    if end-start<5.: raise ValueError('insufficient recording duration')
    mask=(ti>=start)&(ti<=end); ti=ti[mask]
    rot=Rotation.from_quat(np.array([x['q_xyzw'] for x in imu])[mask]).as_matrix(); w=np.array([x['w'] for x in imu])[mask]; acc=np.array([x['a'] for x in imu])[mask]
    names=[x[0] for x in joint[0]['joints']]
    if any([x[0] for x in row['joints']]!=names for row in joint):raise ValueError('joint order changes')
    q=np.array([[x[1] for x in row['joints']] for row in joint]);qd=np.array([[x[2] for x in row['joints']] for row in joint])
    manifest={'scope':__doc__,'source':str(a.run),'duration_s':float(end-start),'seed':a.seed,
              'variants':{k:VARIANTS[k] for k in a.variants},'xml_sha256':hashlib.sha256(a.xml.read_bytes()).hexdigest(),
              'noise_note':'Adds noise to recorded IMU; existing MuJoCo XML declares gyro and accelerometer noise 0.001, orientation framequat 0.',
              'orientation_note':'Synthetic world-frame attitude output errors; gyro and accelerometer measurements remain as recorded unless separately perturbed. Mount rotation is a separate estimator calibration error.',
              'timing_note':'Joint delay makes old samples appear 5 ms newer; simulated header semantics, not physical timestamp proof.',
              'source_hashes':{name:hashlib.sha256((a.run/'trial-record'/name).read_bytes()).hexdigest() for name in ('imu.jsonl','joints.jsonl','odom.jsonl')}}
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    results={}
    for name in a.variants:
        print('starting',name,flush=True)
        result,rows=replay(VARIANTS[name],ti,rot,w,acc,tj,q,qd,names,tt,truth_p,a.xml,SimFootSupport,end-start,a.seed)
        results[name]=result
        np.savez_compressed(a.out/(name+'.npz'),t=np.array([r[0] for r in rows]),p=np.array([r[1] for r in rows]),v=np.array([r[2] for r in rows]))
        (a.out/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
        print(name,json.dumps(result),flush=True)
    (a.out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__': main()
