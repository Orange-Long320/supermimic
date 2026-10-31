"""Add a stationary standing reference to a verified five-gait dataset."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--iteration', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--frames', type=int, default=500)
    a = p.parse_args()
    if a.frames < 100 or a.out.exists():
        raise ValueError('Use at least 100 frames and a new output directory')
    manifest = json.loads((a.source / 'manifest.json').read_text())
    clips = manifest['clips']
    expected = {'walk', 'backward', 'turn_left_inplace', 'turn_right_inplace', 'jog'}
    names = {c['name'] for c in clips}
    if names != expected | {n + '_stop' for n in expected}:
        raise ValueError('Expected exactly five gait and five stop clips')
    for c in clips:
        if sha(a.source / (c['name'] + '.npz')) != c['sha256']:
            raise ValueError('Reference SHA mismatch: ' + c['name'])
    source = a.source / 'walk_stop.npz'
    with np.load(source, allow_pickle=False) as z:
        pose = {k: z[k].copy() for k in z.files}
    if float(pose['fps'][0]) != 50.:
        raise ValueError('Expected 50 Hz source')
    pelvis = list(pose['body_names']).index('pelvis')
    with np.load(a.source / 'walk.npz', allow_pickle=False) as walk:
        initial_q_error = float(np.max(np.abs(walk['joint_pos'][0] - pose['joint_pos'][-1])))
    checks = {}
    def gravity(quat):
        return Rotation.from_quat(quat[[1, 2, 3, 0]]).inv().apply([0., 0., -1.])
    stand_gravity = gravity(pose['body_quat_w'][-1, pelvis])
    for name in sorted(expected):
        with np.load(a.source / (name + '_stop.npz'), allow_pickle=False) as z:
            if list(z['joint_names']) != list(pose['joint_names']):
                raise ValueError('Joint order mismatch: ' + name)
            idx = list(z['body_names']).index('pelvis')
            error = float(np.max(np.abs(z['joint_pos'][-1] - pose['joint_pos'][-1])))
            height_error = float(abs(z['body_pos_w'][-1, idx, 2] - pose['body_pos_w'][-1, pelvis, 2]))
            gravity_error = float(np.max(np.abs(gravity(z['body_quat_w'][-1, idx]) - stand_gravity)))
            terminal_velocity = max(float(np.max(np.abs(z[k][-1]))) for k in
                                    ('joint_vel', 'body_lin_vel_w', 'body_ang_vel_w'))
            if error > 1e-5 or height_error > 1e-5 or gravity_error > 1e-5 or terminal_velocity > 1e-5:
                raise ValueError('Stop endpoint mismatch: ' + name)
            checks[name] = {'joint_max_abs_rad': error, 'pelvis_height_abs_m': height_error,
                            'projected_gravity_max_abs': gravity_error, 'terminal_velocity_max_abs': terminal_velocity}
    if initial_q_error > 1e-5:
        raise ValueError('Standing pose differs from first walking frame')
    for key in ('joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w'):
        pose[key] = np.repeat(pose[key][-1:], a.frames, axis=0)
    for key in ('joint_vel', 'body_lin_vel_w', 'body_ang_vel_w'):
        pose[key][:] = 0.
    # Place the pelvis at the world origin in XY without changing relative geometry.
    pose['body_pos_w'][:, :, :2] -= pose['body_pos_w'][:, pelvis:pelvis+1, :2].copy()
    pose['source'] = np.array('standing hold from verified walk_stop terminal frame')
    pose['velocity_source'] = np.array('stationary hold: zero reference velocities')
    a.out.mkdir(parents=True)
    for c in clips:
        shutil.copy2(a.source / (c['name'] + '.npz'), a.out / (c['name'] + '.npz'))
    out = a.out / 'stand.npz'
    np.savez(out, **pose)
    new = dict(manifest)
    new['clips'] = clips + [{'name': 'stand', 'type': 'stationary_stand',
                             'source': str(source), 'source_sha256': sha(source),
                             'sha256': sha(out), 'frames': a.frames}]
    new['initial_checkpoint_sha256'] = sha(a.checkpoint)
    new['initial_checkpoint_iteration'] = a.iteration
    new['scope'] = 'Reference geometry only; physical contact and policy tracking are unverified'
    new['stand_endpoint_audit'] = {'initial_walk_joint_error_rad': initial_q_error, 'stops': checks}
    (a.out / 'manifest.json').write_text(json.dumps(new, indent=2) + '\n')
    print(json.dumps(new['stand_endpoint_audit'], indent=2))


if __name__ == '__main__':
    main()
