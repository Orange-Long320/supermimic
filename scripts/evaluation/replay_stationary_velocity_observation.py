"""Causal rest-only evaluation. Never enables foot support for a suspended robot.

The independent declaration applies throughout this capture. This is explicitly
not an estimator for walking: constant velocity cannot be inferred from quiet IMU.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

# Sibling audit owns the raw wire format and initial calibration checks.
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_native_stationary_imu import DTYPE
from supermimic.deploy.imu_dead_reckoning import ImuDeadReckoning
from supermimic.deploy.velocity_observation import DeclaredRestGyroBias,VelocityObservationCorrection


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stationary-for-entire-capture',action='store_true')
    a=p.parse_args()
    if not a.stationary_for_entire_capture:p.error('An independent whole-capture stationary declaration is required')
    initial=json.loads((a.capture/'audit.json').read_text());meta=json.loads((a.capture/'capture.json').read_text())
    if not meta.get('operator_declared_stationary') or initial['calibration'] is None:raise ValueError('No valid independent rest calibration')
    dtype=DTYPE if meta['wire_size']==832 else np.dtype([(n,DTYPE.fields[n][0]) for n in DTYPE.names[:12]])
    data=np.fromfile(a.capture/'raw.bin',dtype=dtype);t=(data['stamp']-data['stamp'][0])*1e-9
    anchor=int(np.searchsorted(t,initial['calibration_end_s']));start=anchor-int(initial['calibration']['samples'])+1
    variants={}
    for name,adaptive,observe in [('baseline',False,False),('rest_gyro_mean',True,False),('rest_velocity',False,True),('rest_gyro_and_velocity',True,True)]:
        f=ImuDeadReckoning(point_trapezoid=True,max_gap_seconds=.04);f.initialize(initial['calibration'])
        bias=DeclaredRestGyroBias();correction=VelocityObservationCorrection();last=None;points=[];reasons=Counter()
        for i in range(start,anchor):bias.update(float(t[i]),data['imu'][i,:3],stationary_declared=True)
        for i in range(anchor,len(data)):
            # Motion thresholds only veto the external rest declaration.
            quiet=(np.linalg.norm(data['imu'][i,:3])<=.08 and abs(np.linalg.norm(data['imu'][i,3:])-9.81)<=.5 and np.max(np.abs(data['qd'][i]))<=.03 and np.max(np.abs(data['q'][i]-data['q'][start]))<=.015)
            b=bias.update(float(t[i]),data['imu'][i,:3],stationary_declared=bool(quiet))
            if adaptive and b is not None:
                delta=b-f.gyro_bias;f.gyro_bias=b
                if f.previous_gyro is not None:f.previous_gyro-=delta
            f.update(float(t[i]-t[anchor]),data['imu'][i,:3],data['imu'][i,3:])
            if last is not None and t[i]-last<.020:continue
            last=t[i]
            z=np.zeros(2) if observe and quiet and b is not None else None
            x=correction.update(float(t[i]),f.x,z)
            reasons['rest-velocity' if z is not None else 'inertial-only']+=1
            points.append([float(t[i]-t[anchor]),*x])
        trace=np.asarray(points)
        variants[name]=dict(max_xy_displacement_m=float(np.linalg.norm(trace[:,1:3],axis=1).max()),final_xy_m=trace[-1,1:3].tolist(),final_velocity_m_s=trace[-1,3:5].tolist(),final_gyro_bias_rad_s=f.gyro_bias.tolist(),observation_counts=dict(reasons),position_resets=0)
    report=dict(scope='offline conditional rest-only observation, no truth pose or policy input; foot support DISABLED',capture=str(a.capture),evaluation_seconds=float(t[-1]-t[anchor]),variants=variants,moving_accuracy_validated=False,motor_readiness_approved=False)
    with a.out.open('x') as out:json.dump(report,out,indent=2);out.write('\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
