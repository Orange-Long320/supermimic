"""Offline audit of native sensor frames from independently declared physical rest.

No ROS, command publishers or truth pose. Calibration uses one initial window;
subsequent samples evaluate drift without resetting position, velocity or tilt.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from supermimic.deploy.imu_dead_reckoning import ImuDeadReckoning, fit_static_initialization
from supermimic.deploy.native_imu_reader import FRAME, MAGIC

DTYPE = np.dtype([
    ('magic','<u8'),('stamp','<i8'),('received','<i8'),('imu','<f8',(6,)),('frame','S64'),
    ('q','<f8',(31,)),('qd','<f8',(31,)),('joint_stamps','<i8',(4,)),
    ('joint_received','<i8',(4,)),('domains','<i4',(4,)),('groups','<u4'),('joints','<u4'),
    ('quat','<f8',(4,)),('state','<f8',(4,)),('gyro','<f8',(3,)),('samples','<u8'),
    ('max_gap','<f8'),('integrated','<u8')])


def audit(capture):
    metadata=json.loads((capture/'capture.json').read_text())
    if not metadata.get('operator_declared_stationary'):
        raise ValueError('Independent physical rest declaration required')
    size=(capture/'raw.bin').stat().st_size
    wire_size=metadata['wire_size']
    dtype=DTYPE if wire_size==832 else np.dtype([(n,DTYPE.fields[n][0]) for n in DTYPE.names[:12]])
    if wire_size not in (720,832) or dtype.itemsize!=wire_size or not size or size%wire_size:
        raise ValueError('Empty or incomplete native capture')
    data=np.fromfile(capture/'raw.bin',dtype=dtype)
    if np.any(data['magic']!=(MAGIC if wire_size==832 else MAGIC-1)) or (wire_size==832 and np.any(data['integrated'])):
        raise ValueError('Expected full-rate unintegrated raw sensor frames')
    stamps=data['stamp'];time=(stamps-stamps[0])*1e-9;dt=np.diff(stamps)*1e-9
    if np.any(dt<=0) or not np.isfinite(data['imu']).all():
        raise ValueError('Invalid raw timestamps or measurements')
    gyro=data['imu'][:,:3];acc=data['imu'][:,3:]
    aligned=(data['groups']==15)&(data['joints']==0x7fffffff)&np.all(data['domains']==0,axis=1)
    aligned &= np.all((stamps[:,None]-data['joint_stamps']>=0)&(stamps[:,None]-data['joint_stamps']<=25000000),axis=1)
    report=dict(scope='stationary sensor-only measurement; not motion accuracy acceptance',
                capture=str(capture),samples=len(data),sensor_duration_s=float(time[-1]),
                header_hz=float(1/dt.mean()),header_gap_ms=np.percentile(dt*1000,[0,50,95,99,100]).tolist(),
                receive_age_ms=np.percentile((data['received']-stamps)*1e-6,[0,50,95,99,100]).tolist(),
                receive_gap_ms=np.percentile(np.diff(data['received'])*1e-6,[0,50,95,99,100]).tolist(),
                gaps_above_3ms=int(np.sum(dt>.003)),aligned_samples=int(aligned.sum()),
                gyro_mean_rad_s=gyro.mean(0).tolist(),gyro_std_rad_s=gyro.std(0).tolist(),
                acceleration_mean_m_s2=acc.mean(0).tolist(),acceleration_std_m_s2=acc.std(0).tolist(),
                encoder_max_speed_rad_s=float(np.abs(data['qd'][aligned]).max()) if aligned.any() else None,
                encoder_max_excursion_rad=float(np.ptp(data['q'][aligned],axis=0).max()) if aligned.any() else None,
                motor_output_started=False)
    calibration=None;rejections=[];anchor=None
    # Earliest complete three-second initial window after one second discovery.
    for start in np.arange(1.,min(10.,time[-1]-3.),.5):
        lo=int(np.searchsorted(time,start));hi=int(np.searchsorted(time,time[lo]+3.,side='left'))
        if hi>=len(data) or not aligned[lo:hi+1].all():continue
        window=[dict(time_s=float(time[i]),gyro=gyro[i].tolist(),acceleration=acc[i].tolist(),
                     joint_pos=data['q'][i].tolist(),joint_vel=data['qd'][i].tolist()) for i in range(lo,hi+1)]
        try:calibration=fit_static_initialization(window,declared_stationary=True,max_gap_seconds=.04)
        except ValueError as exc:rejections.append(dict(start_s=float(time[lo]),reason=str(exc)));continue
        anchor=hi;break
    report.update(calibration=calibration,rejected_initial_windows=rejections)
    if calibration is None:return report
    report['calibration_end_s']=float(time[anchor]);report['evaluation_seconds']=float(time[-1]-time[anchor])
    report['gyro_mean_after_calibration_rad_s']=gyro[anchor:].mean(0).tolist()
    report['gyro_bias_change_rad_s']=(gyro[anchor:].mean(0)-calibration['gyro_bias']).tolist()
    report['gyro_10s_blocks']=[dict(start_s=float(time[i]),mean_rad_s=gyro[i:i+5000].mean(0).tolist(),
                                       std_rad_s=gyro[i:i+5000].std(0).tolist()) for i in range(anchor,len(data),5000)]
    variants={}
    for name,use_bias in [('calibrated',True),('without_gyro_bias_subtraction',False)]:
        f=ImuDeadReckoning(point_trapezoid=True,max_gap_seconds=.04)
        c=dict(calibration)
        if not use_bias:c['gyro_bias']=[0.,0.,0.]
        f.initialize(c);positions=[];world=[]
        for i in range(anchor,len(data)):
            f.update(float(time[i]-time[anchor]),gyro[i],acc[i])
            positions.append(f.x[:2].copy());world.append(f.last_world_acceleration.copy())
        world=np.asarray(world);positions=np.asarray(positions)
        relative=f.initial_rotation.inv()*f.rotation
        variants[name]=dict(final_xy_displacement_m=f.x[:2].tolist(),max_xy_displacement_m=float(np.linalg.norm(positions,axis=1).max()),
                            final_xy_velocity_m_s=f.x[2:].tolist(),mean_world_acceleration_m_s2=world.mean(0).tolist(),
                            std_world_acceleration_m_s2=world.std(0).tolist(),final_relative_rotation_deg=relative.as_rotvec(degrees=True).tolist())
    report['variants']=variants
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('capture',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();report=audit(a.capture)
    with a.out.open('x') as out:json.dump(report,out,indent=2);out.write('\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
