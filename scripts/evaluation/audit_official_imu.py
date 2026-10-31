"""Compare official simulator IMU orientation with FK; not a sensor calibration."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def rotation(wxyz):
    return Rotation.from_quat(np.asarray(wxyz)[[1, 2, 3, 0]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('frames', 'profile', 'scene', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Output already exists')
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    names = json.loads(args.profile.read_text())['joint_names']
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso_link')
    if min(*joints, torso) < 0:
        raise ValueError('Missing joint or torso in model')
    qaddr = [model.jnt_qposadr[j] for j in joints]
    imu_to_torso, imu_to_pelvis = [], []
    with args.frames.open() as source:
        for line in source:
            row = json.loads(line)
            data.qpos[:7] = np.r_[row['root'], row['q']]
            data.qpos[qaddr] = row['joint_pos']
            mujoco.mj_kinematics(model, data)
            imu = rotation(row['imu_orientation_wxyz'])
            imu_to_torso.append(np.degrees((imu.inv()*rotation(data.xquat[torso])).magnitude()))
            imu_to_pelvis.append(np.degrees((imu.inv()*rotation(row['q'])).magnitude()))
    if not imu_to_torso:
        raise ValueError('Empty frame file')
    def quantiles(v):
        return dict(zip(('p50', 'p95', 'max'), map(float, np.percentile(v, (50, 95, 100)))))
    report = {'frames': len(imu_to_torso), 'imu_to_fk_torso_degrees': quantiles(imu_to_torso),
              'imu_to_odom_pelvis_degrees': quantiles(imu_to_pelvis),
              'scope': 'simulator messages and model FK; asynchronous message snapshots, no hardware calibration'}
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
