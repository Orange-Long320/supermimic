"""Offline support-foot kinematic odometry diagnostic against simulator truth.

Foot support is guessed from relative height. This is not a real contact sensor,
not a robot state estimator, and must never feed a hardware controller.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('trace','profile','scene','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--hysteresis-m',type=float,default=.015)
    a=p.parse_args()
    if a.out.exists() or a.hysteresis_m<0:raise ValueError('Expected new output and nonnegative hysteresis')
    rows=[json.loads(line) for line in a.trace.open()]
    if len(rows)<2:raise ValueError('Trace too short')
    model=mujoco.MjModel.from_xml_path(str(a.scene)); data=mujoco.MjData(model)
    names=json.loads(a.profile.read_text())['joint_names']
    ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name) for name in names]
    feet=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,side+'_ankle_roll_link') for side in ('left','right')]
    if min(*ids,*feet)<0:raise ValueError('Joint or ankle missing')
    address=[model.jnt_qposadr[i] for i in ids]; times=np.array([r['elapsed'] for r in rows])
    truth=np.array([r['root'][:2] for r in rows]);positions=np.empty((len(rows),2,3))
    if np.any(np.diff(times)<=0):raise ValueError('Nonmonotonic trace')
    for i,row in enumerate(rows):
        data.qpos[:7]=np.r_[[0.,0.,0.],row['q']];data.qpos[address]=row['joint_pos']
        mujoco.mj_kinematics(model,data);positions[i]=data.xpos[feet]
    estimate=np.zeros((len(rows),2));chosen=np.empty(len(rows),dtype=int)
    chosen[0]=int(np.argmin(positions[0,:,2]))
    for i in range(1,len(rows)):
        prev=chosen[i-1];other=1-prev
        chosen[i]=other if positions[i,other,2] < positions[i,prev,2]-a.hysteresis_m else prev
        side=chosen[i];estimate[i]=estimate[i-1]-(positions[i,side,:2]-positions[i-1,side,:2])
    err=np.linalg.norm(estimate-(truth-truth[0]),axis=1)
    stops=[];mode=[r['mode'] for r in rows]
    for i in range(1,len(rows)):
        if mode[i-1]!='stand' and mode[i]=='stand':
            j=min(int(np.searchsorted(times,times[i]+4)),len(rows)-1)
            measured=(estimate[j]-estimate[i])-(truth[j]-truth[i])
            stops.append({'motion':mode[i-1],'time_s':float(times[i]),'window_s':float(times[j]-times[i]),
                          'relative_4s_error_m':float(np.linalg.norm(measured))})
    report={'frames':len(rows),'seconds':float(times[-1]-times[0]),'support_switches':int(np.sum(chosen[1:]!=chosen[:-1])),
            'xy_error_p50_m':float(np.percentile(err,50)),'xy_error_p95_m':float(np.percentile(err,95)),
            'xy_error_max_m':float(err.max()),'xy_error_final_m':float(err[-1]),'stop_windows':stops,
            'scope':'offline simulator trace comparison; orientation and joint data taken from simulation; foot support inferred solely by relative height, not measured contact',
            'limitations':'Neither physical contact nor pelvis height estimated; joint/IMU timing and real mounting not represented; not for hardware control.'}
    a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
