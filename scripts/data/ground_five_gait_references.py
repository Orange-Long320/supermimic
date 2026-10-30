"""Ground five gait references smoothly and preserve the walk first-frame pose."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    import mujoco
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.data.physics_motion import export_qpos_motion
    model = load_g1_model()
    data = mujoco.MjData(model)
    feet = []
    for side in ('left', 'right'):
        body = model.body(side + '_ankle_roll_link').id
        meshes = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != body or not (model.geom_contype[g] or model.geom_conaffinity[g]) or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            k = model.geom_dataid[g]
            meshes.append((g, model.mesh_vert[model.mesh_vertadr[k]:model.mesh_vertadr[k]+model.mesh_vertnum[k]].copy()))
        if not meshes:
            raise ValueError('Missing foot collision mesh')
        feet.append(meshes)
    original = json.loads((a.source / 'manifest.json').read_text())
    a.out.mkdir(parents=True)
    clips = []
    for item in original['clips']:
        if item['type'] != 'prior_gait':
            continue
        source = a.source / (item['name'] + '.npz')
        if sha(source) != item['sha256']:
            raise ValueError('Source hash mismatch')
        with np.load(source, allow_pickle=False) as z:
            root = list(z['body_names']).index('pelvis')
            q = np.c_[z['body_pos_w'][:, root], z['body_quat_w'][:, root], z['joint_pos']].astype(float)
            fps = float(z['fps'][0])
        heights = []
        for row in q:
            data.qpos[:] = row
            mujoco.mj_kinematics(model, data)
            heights.append([min(float((v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])[:, 2].min()) for g, v in meshes) for meshes in feet])
        heights = np.asarray(heights)
        support = heights.min(axis=1)
        shift = gaussian_filter1d(support, sigma=3., mode='nearest')
        q[:, 2] -= shift
        if item['name'] == 'walk':
            pose = q[0].copy()
            pose[2] += shift[0] - support[0]
            np.savez(a.out / 'initial-stance.npz', qpos=pose)
        path = a.out / (item['name'] + '.npz')
        export_qpos_motion(model, q, path, fps, source=str(source))
        clips.append({'name': item['name'], 'type': 'prior_gait', 'sha256': sha(path), 'source': str(source), 'source_sha256': sha(source),
                      'ground_correction_range_m': [float(shift.min()), float(shift.max())],
                      'grounded_min_foot_height_m': float((heights-shift[:, None]).min()),
                      'max_correction_step_m': float(np.max(np.abs(np.diff(shift))))})
    if len(clips) != 5:
        raise ValueError('Expected five gait clips')
    (a.out / 'manifest.json').write_text(json.dumps({'clips': clips, 'scope': 'Smoothed kinematic ground correction, not physical stability proof'}, indent=2) + '\n')
    print(json.dumps(clips, indent=2))


if __name__ == '__main__':
    main()
