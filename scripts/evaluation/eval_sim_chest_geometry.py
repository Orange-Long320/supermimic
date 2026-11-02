"""Check captured chest orientation/gyro against simulator pelvis truth and waist FK."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from supermimic.deploy.pelvis_state import UrdfPelvisMount, pelvis_angular_velocity


def rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--record', type=Path, required=True)
    p.add_argument('--mount-urdf', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    chest, odom, joint = (rows(a.record/n) for n in ('imu-chest.jsonl', 'odom.jsonl', 'joints.jsonl'))
    joint = [r for r in joint if r['group'] == 'waist']
    origin = chest[0]['stamp'][0]*10**9+chest[0]['stamp'][1]
    def times(data):
        return np.array([(r['stamp'][0]*10**9+r['stamp'][1]-origin)*1e-9 for r in data])
    ct, ot, jt = times(chest), times(odom), times(joint)
    if any(np.any(np.diff(t) <= 0) for t in (ct, ot, jt)):
        raise ValueError('Nonmonotonic acquisition timestamps')
    oq = Rotation.from_quat([r['q_xyzw'] for r in odom])
    ow = np.array([r['angular'] for r in odom])
    slerp = Slerp(ot, oq)
    names = ('waist_yaw_joint', 'waist_pitch_joint', 'waist_roll_joint')
    mount = UrdfPelvisMount(a.mount_urdf, names, [-.0248787, .00198528, .105938], np.eye(3))
    max_t = min(ct[-1], ot[-1], jt[-1]); min_t = max(ct[0], ot[0], jt[0])
    indices = np.nonzero((ct >= min_t) & (ct <= max_t))[0][::10]
    angles, angular_errors, skews, waist_speed = [], [], [], []
    for i in indices:
        j = np.searchsorted(jt, ct[i], side='right')-1
        delta = ct[i]-jt[j]
        if not 0 <= delta <= .005:
            continue
        data = {x[0]: x[1:] for x in joint[j]['joints']}
        qd = np.array([data[n][1] for n in names])
        q = np.array([data[n][0] for n in names])+delta*qd
        _, rpi, wm, _ = mount.mount_with_velocity(q, qd)
        rwi = Rotation.from_quat(chest[i]['q_xyzw']).as_matrix()
        rp = rwi@rpi.T
        truth = slerp([ct[i]]).as_matrix()[0]
        angles.append(np.linalg.norm(Rotation.from_matrix(truth.T@rp).as_rotvec()))
        # Odometry angular velocity is checked independently below; it is treated as body-frame.
        truth_w = np.array([np.interp(ct[i], ot, ow[:, k]) for k in range(3)])
        angular_errors.append(np.linalg.norm(pelvis_angular_velocity(chest[i]['w'], rpi, wm)-truth_w))
        skews.append(delta); waist_speed.append(np.linalg.norm(qd))
    if not angles:
        raise ValueError('No synchronized samples')
    report = {'samples': len(angles), 'max_joint_age_ms': float(max(skews)*1000),
              'waist_speed_p95_rad_s': float(np.percentile(waist_speed, 95)),
              'orientation_error_rad': {'median': float(np.median(angles)), 'p95': float(np.percentile(angles, 95)), 'max': float(max(angles))},
              'angular_velocity_error_rad_s': {'median': float(np.median(angular_errors)), 'p95': float(np.percentile(angular_errors, 95)), 'max': float(max(angular_errors))},
              'scope': 'Simulated chest and waist samples; odometry used only for scoring; 5 ms maximum causal joint extrapolation'}
    a.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
