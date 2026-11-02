"""Same-sample offline timing, initial bias and waist-kinematic ablations."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from eval_walking_imu import kalman
from supermimic.deploy.pelvis_state import UrdfPelvisMount, pelvis_angular_velocity, pelvis_velocity_from_imu


def rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--record', type=Path, required=True)
    p.add_argument('--mount-urdf', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    imu, odom, joints = (rows(a.record/n) for n in ('imu-chest.jsonl', 'odom.jsonl', 'joints.jsonl'))
    joints = [x for x in joints if x['group'] == 'waist']
    origin = imu[0]['stamp'][0]*10**9+imu[0]['stamp'][1]
    def stamps(data):
        return np.array([(r['stamp'][0]*10**9+r['stamp'][1]-origin)*1e-9 for r in data])
    t, ot, jt = stamps(imu), stamps(odom), stamps(joints)
    if any(np.any(np.diff(x) <= 0) for x in (t, ot, jt)):
        raise ValueError('Nonmonotonic input')
    accel, gyro = np.array([r['a'] for r in imu]), np.array([r['w'] for r in imu])
    rot = Rotation.from_quat([r['q_xyzw'] for r in imu])
    world = rot.apply(accel)
    rec = np.array([r['recv'] for r in imu]); rec -= rec[0]
    window = t <= 2.
    still = (abs(np.linalg.norm(accel[window], axis=1)-9.81) <= .5) & (np.linalg.norm(gyro[window], axis=1) <= .08)
    biased = world[window, :2].mean(axis=0)
    variants = {'receive_zero_bias': kalman(rec, accel, gyro, world, np.zeros(2)),
                'header_zero_bias': kalman(t, accel, gyro, world, np.zeros(2)),
                'header_moving_window_bias_diagnostic': kalman(t, accel, gyro, world, biased)}
    names = ('waist_yaw_joint', 'waist_pitch_joint', 'waist_roll_joint')
    mount = UrdfPelvisMount(a.mount_urdf, names, [-.0248787, .00198528, .105938], np.eye(3))
    indices = np.arange(0, len(t), 10)
    indices = indices[(t[indices] >= max(ot[0]+.05, jt[0])) & (t[indices] <= min(ot[-1]-.05, jt[-1]))]
    pos, vel, valid = [], [], []
    for i in indices:
        j = np.searchsorted(jt, t[i], side='right')-1
        dt = t[i]-jt[j]
        if not 0 <= dt <= .005:
            continue
        d = {x[0]: x[1:] for x in joints[j]['joints']}
        qd = np.array([d[n][1] for n in names])
        q = np.array([d[n][0] for n in names])+dt*qd
        pp, rr, wm, vm = mount.mount_with_velocity(q, qd)
        rwp = rot[i].as_matrix() @ rr.T
        omega = pelvis_angular_velocity(gyro[i], rr, wm)
        p0, v0, _ = variants['header_zero_bias']
        pos.append(p0[i]-(rwp@pp)[:2])
        vel.append(pelvis_velocity_from_imu([*v0[i], 0.], rwp, pp, omega, vm)[:2])
        valid.append(i)
    valid = np.asarray(valid)
    if len(valid) < 250:
        raise ValueError('Insufficient waist samples')
    tv = t[valid]
    truth = np.array([r['position'][:2] for r in odom])
    def interp(times, values, at):
        return np.column_stack([np.interp(at, times, values[:, k]) for k in range(values.shape[1])])
    actual = interp(ot, truth, tv); actual -= actual[0]
    av = (interp(ot, truth, tv+.05)-interp(ot, truth, tv-.05))/.1
    output = {'calibration_first_2s': {'still_sample_fraction': float(np.mean(still)),
             'horizontal_world_mean': biased.tolist(), 'valid_static_calibration': bool(np.all(still))},
             'duration_s': float(tv[-1]-tv[0]), 'variants': {}}
    scored = {k:(v[0][valid], v[1][valid]) for k,v in variants.items()}
    scored['header_zero_bias_waist_compensated'] = (np.array(pos), np.array(vel))
    for name, (position, velocity) in scored.items():
        error = np.linalg.norm(position-position[0]-actual, axis=1)
        verr = np.linalg.norm(velocity-av, axis=1)
        output['variants'][name] = {'position_rmse_m': float(np.sqrt(np.mean(error**2))),
            'velocity_rmse_m_s': float(np.sqrt(np.mean(verr**2))),
            'final_position_error_m': float(error[-1]), 'final_velocity_error_m_s': float(verr[-1]),
            'checkpoints': [{'seconds': k, 'position_error_m': float(error[np.searchsorted(tv,tv[0]+k)]),
                             'velocity_error_m_s': float(verr[np.searchsorted(tv,tv[0]+k)])}
                            for k in (5, 10, 20, 30, 40, 50) if tv[0]+k < tv[-1]]}
    a.out.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
