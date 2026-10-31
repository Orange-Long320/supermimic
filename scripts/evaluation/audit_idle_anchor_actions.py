"""Recompute frozen-policy actions and isolate horizontal anchor feedback.

Counterfactual actions are diagnostic only. They do not prove closed-loop safety
and do not change a recorded observation or the original trajectory.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'diagnosis', 'profile', 'policy', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    torch.set_num_threads(2)
    model = torch.jit.load(str(a.policy), map_location='cpu').eval()
    profile = json.loads(a.profile.read_text())
    names = profile['joint_names']
    default = np.array([profile['default_joint_pos'][n] for n in names])
    scale = np.array([profile['action_scale'][n] for n in names])
    xml = Path(profile['xml_path'])
    kin = mujoco.MjModel.from_xml_path(str(xml if xml.is_absolute() else a.profile.parent/xml))
    data = mujoco.MjData(kin)
    adr = [kin.jnt_qposadr[kin.joint(n).id] for n in names]
    anchor = kin.body(profile['anchor_body']).id
    source = json.loads((a.diagnosis/'result.json').read_text())['trials']
    output = {}
    for case in source:
        frames = a.run/case/'policy/frames.jsonl'
        if hashlib.sha256(frames.read_bytes()).hexdigest() != source[case]['source_frames_sha256']:
            raise ValueError('Diagnosis and policy trace mismatch')
        rows = [json.loads(line) for line in frames.open()]
        with np.load(a.diagnosis/(case+'.npz')) as z:
            reference = z['reference_position'].copy()
        times = np.array([r['elapsed'] for r in rows])
        mode = np.array([r['mode'] for r in rows])
        release = np.flatnonzero((mode[:-1] == 'walk') & (mode[1:] == 'stand'))[-1]+1
        ids = np.flatnonzero((times >= times[release]+1.5) & (times < times[release]+15))[::5]
        records = []
        for i in ids:
            history = []
            for j in range(i-9, i+1):
                r = rows[j]
                w,x,y,z = r['imu_orientation_wxyz']
                gravity = [2*(w*y-x*z), -2*(w*x+y*z), -1+2*(x*x+y*y)]
                previous = (np.array(rows[j-1]['target'])-default)/scale
                history.append([np.array(r['joint_pos'])-default, r['joint_vel'],
                                r['body_angular_velocity'], gravity, previous])
            obs = np.concatenate([np.asarray(history[j][term]) for term in range(5) for j in range(10)])
            goal = np.r_[np.tile(rows[i]['goal_current'], 10), rows[i]['anchor']]
            data.qpos[:7] = np.r_[rows[i]['root'], rows[i]['q']]
            data.qpos[adr] = rows[i]['joint_pos']
            mujoco.mj_kinematics(kin, data)
            actual = Rotation.from_quat(data.xquat[anchor][[1,2,3,0]])
            difference = np.r_[reference[i,:2]-np.array(rows[i]['root'])[:2], 0.]
            alternative = goal.copy()
            alternative[-9:-6] -= actual.inv().apply(difference)
            xx = torch.tensor(obs[None], dtype=torch.float32)
            with torch.inference_mode():
                original = model(xx, torch.tensor(goal[None], dtype=torch.float32))[0].numpy()
                changed = model(xx, torch.tensor(alternative[None], dtype=torch.float32))[0].numpy()
            recorded = (np.array(rows[i]['target'])-default)/scale
            error = float(np.max(np.abs(original-recorded)))
            if error > 2e-5:
                raise ValueError('Original policy action does not reproduce: '+str(error))
            delta = (changed-original)*scale
            major = np.argsort(np.abs(delta))[-4:][::-1]
            records.append(dict(after_release_s=float(times[i]-times[release]),
                observed_joint_speed_rms_rad_s=float(np.sqrt(np.mean(np.array(rows[i]['joint_vel'])**2))),
                source_action_max_error=error, horizontal_root_error_m=difference[:2].tolist(),
                counterfactual_target_delta_rms_rad=float(np.sqrt(np.mean(delta**2))),
                largest_target_deltas_rad={names[j]:float(delta[j]) for j in major}))
        output[case] = dict(samples=records,
            max_source_action_error=max(r['source_action_max_error'] for r in records),
            max_counterfactual_joint_target_delta_rms_rad=max(r['counterfactual_target_delta_rms_rad'] for r in records))
    a.out.write_text(json.dumps(dict(trials=output, scope='Same-state action attribution; horizontal anchor error removed only for a counterfactual forward pass, not a controller or acceptance'), indent=2)+'\n')
    print(json.dumps({k:{p:v for p,v in val.items() if p != 'samples'} for k,val in output.items()}, indent=2))


if __name__ == '__main__':
    main()
