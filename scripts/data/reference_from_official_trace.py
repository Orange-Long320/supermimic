"""Extract an exploratory kinematic reference from an official simulation trace."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('trace', 'profile', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--start', type=float, required=True)
    p.add_argument('--end', type=float, required=True)
    p.add_argument('--release', type=float, required=True)
    p.add_argument('--fps', type=float, default=50.)
    a = p.parse_args()
    if a.out.exists() or not a.start < a.release < a.end or a.fps <= 0:
        raise ValueError('Expected new output and valid time interval')
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.data.physics_motion import export_qpos_motion
    model = load_g1_model()
    names = json.loads(a.profile.read_text())['joint_names']
    model_names = [model.joint(i).name for i in range(1, model.njnt)]
    if set(model_names) != set(names) or len(set(names)) != len(names):
        raise ValueError('Joint mapping mismatch')
    rows = [json.loads(line) for line in a.trace.open()]
    t = np.asarray([r['elapsed'] for r in rows])
    if np.any(np.diff(t) <= 0) or a.start < t[0] or a.end > t[-1]:
        raise ValueError('Nonmonotonic trace or interval not recorded')
    grid = a.start + np.arange(int(np.floor((a.end-a.start)*a.fps))+1)/a.fps
    positions = np.asarray([r['root'] for r in rows])
    joints = np.asarray([r['joint_pos'] for r in rows])[:, [names.index(n) for n in model_names]]
    rot = Rotation.from_quat(np.asarray([r['q'] for r in rows])[:, [1,2,3,0]])
    interpolate = lambda v: np.column_stack([np.interp(grid, t, v[:,i]) for i in range(v.shape[1])])
    q = np.c_[interpolate(positions), Slerp(t, rot)(grid).as_quat()[:, [3,0,1,2]], interpolate(joints)]
    a.out.mkdir(parents=True)
    dest = a.out/'reference.npz'
    export_qpos_motion(model, q, dest, a.fps, source=str(a.trace))
    nearest = np.clip(np.searchsorted(t, grid), 1, len(t)-1)
    max_gap = float(np.max(t[nearest]-t[nearest-1]))
    meta = {'source':str(a.trace), 'source_sha256':sha(a.trace), 'reference_sha256':sha(dest),
            'start_s':a.start, 'end_s':a.end, 'release_s':a.release, 'release_in_clip_s':a.release-a.start,
            'frames':len(q), 'fps':a.fps, 'maximum_source_gap_s':max_gap,
            'scope':'Interpolated kinematic reference from previously recorded policy physics; reference playback is not a new physics test or an improved controller',
            'limitations':'Original stopping distance and pose deviations remain; contact loads not recorded; differentiated velocities are estimates.'}
    (a.out/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))


if __name__ == '__main__':
    main()
