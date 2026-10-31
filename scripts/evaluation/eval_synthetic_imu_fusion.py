"""Offline IMU/leg fusion screen with synthetic acceleration from saved poses.

Never uses hardware or produces actuation. The acceleration is reconstructed
from simulator truth; recorded orientation is used without inertial drift.
Results are not independent validation and are optimistic by construction.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('trace', 'profile', 'scene', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--seed', type=int, default=217)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('Output already exists')
    rows = [json.loads(s) for s in a.trace.open()]
    if len(rows) < 100:
        raise ValueError('Trace too short')
    ts = np.array([r['elapsed'] for r in rows]); dt = .02
    if np.any(np.diff(ts) <= 0):
        raise ValueError('Nonmonotonic trace')
    grid = np.arange(ts[0]+.2, ts[-1]-.2, dt)
    if len(grid) < 100:
        raise ValueError('Trace too short after cropping')
    def interp(values):
        return np.column_stack([np.interp(grid, ts, values[:,i]) for i in range(values.shape[1])])
    truth = interp(np.array([r['root'] for r in rows]))
    q = interp(np.array([r['joint_pos'] for r in rows]))
    q_imu = np.array([r.get('imu_orientation_wxyz', r['q']) for r in rows])
    rot = Slerp(ts, Rotation.from_quat(q_imu[:,[1,2,3,0]]))(grid)
    # Construct synthetic specific force from smoothed truth. This leaks truth
    # into the test and omits IMU lever-arm acceleration.
    win = 15
    acc = savgol_filter(truth, win, 3, deriv=2, delta=dt, axis=0, mode='interp')
    gravity = np.array([0.,0.,-9.81]); rng = np.random.default_rng(a.seed)
    bias_world = np.array([.02,-.015,0.]); noise = rng.normal(0.,.04,acc.shape)
    specific_body = rot.inv().apply(acc-gravity+bias_world)+noise
    measured_world = rot.apply(specific_body)+gravity
    m = mujoco.MjModel.from_xml_path(str(a.scene)); d = mujoco.MjData(m)
    names = json.loads(a.profile.read_text())['joint_names']
    ids = [mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]
    feet = [mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,s+'_ankle_roll_link') for s in ('left','right')]
    if min(*ids,*feet) < 0:
        raise ValueError('Missing model joint or foot')
    addr = [m.jnt_qposadr[i] for i in ids]
    rel = np.zeros((len(grid),2,3)); quats = rot.as_quat()[:,[3,0,1,2]]
    for i in range(len(grid)):
        d.qpos[:7] = np.r_[[0.,0.,0.],quats[i]]; d.qpos[addr] = q[i]
        mujoco.mj_kinematics(m,d); rel[i] = d.xpos[feet]
    foot_v = savgol_filter(rel,win,3,deriv=1,delta=dt,axis=0,mode='interp')
    # State: horizontal position, velocity and world-frame acceleration bias.
    x = np.zeros(6); P = np.diag([.01,.01,.1,.1,.01,.01])
    imu = np.zeros((len(grid),4)); est = np.zeros((len(grid),6))
    F = np.eye(6); F[0,2]=F[1,3]=dt; F[0,4]=F[1,5]=-.5*dt**2; F[2,4]=F[3,5]=-dt
    B = np.zeros((6,2)); B[0,0]=B[1,1]=.5*dt**2; B[2,0]=B[3,1]=dt
    Q = np.diag([1e-7,1e-7,4e-5,4e-5,1e-7,1e-7])
    H = np.zeros((2,6)); H[0,2]=H[1,3]=1
    R = np.eye(2)*(.10**2); I = np.eye(6); accepted = switches = 0; last = None
    for i in range(1,len(grid)):
        u = measured_world[i,:2]; x = F@x+B@u; P = F@P@F.T+Q
        imu[i,:2] = imu[i-1,:2]+imu[i-1,2:]*dt+.5*u*dt**2
        imu[i,2:] = imu[i-1,2:]+u*dt
        low = np.min(rel[i,:,2]); candidates = []
        for foot in range(2):
            measurement = -foot_v[i,foot,:2]
            if rel[i,foot,2]-low <= .015 and abs(foot_v[i,foot,2]) <= .12 and np.linalg.norm(measurement) <= 1.5:
                candidates.append((np.linalg.norm(measurement-H@x),foot,measurement))
        if candidates:
            diff,foot,z = min(candidates,key=lambda c:c[0])
            if diff < .6:
                S = H@P@H.T+R; K = np.linalg.solve(S,(P@H.T).T).T
                x = x+K@(z-H@x); P = (I-K@H)@P@(I-K@H).T+K@R@K.T
                accepted += 1
                if last is not None and foot != last: switches += 1
                last = foot
        est[i] = x
    xy = truth[:,:2]-truth[0,:2]
    def metrics(estimate):
        error = np.linalg.norm(estimate-xy,axis=1)
        return {'p50_m':float(np.percentile(error,50)),'p95_m':float(np.percentile(error,95)),
                'max_m':float(error.max()),'final_m':float(error[-1])}
    modes = [r['mode'] for r in rows]; stop = []
    for i in range(1,len(rows)):
        if modes[i-1] != 'stand' and modes[i] == 'stand' and grid[0] <= ts[i] and ts[i]+4 <= grid[-1]:
            j=int(np.searchsorted(grid,ts[i])); k=int(np.searchsorted(grid,ts[i]+4)); actual=xy[k]-xy[j]
            stop.append({'skill':modes[i-1],'time_s':float(ts[i]),
                         'imu_only_relative_4s_error_m':float(np.linalg.norm((imu[k,:2]-imu[j,:2])-actual)),
                         'fusion_relative_4s_error_m':float(np.linalg.norm((est[k,:2]-est[j,:2])-actual))})
    report = {'seconds':float(grid[-1]-grid[0]),'samples':len(grid),'seed':a.seed,
              'imu_only_xy_error':metrics(imu[:,:2]),'fusion_xy_error':metrics(est[:,:2]),
              'foot_updates':accepted,'foot_switches':switches,'stop_windows':stop,
              'simulation':'specific force reconstructed from smoothed true root position with synthetic noise and fixed bias; true orientation supplied; ideal timestamps, noncausal smoothing and no lever arm',
              'limitations':'Not an independent sensor test; guessed foot support and ideal orientation, no measured contact, no hardware validation, no control integration.'}
    a.out.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
