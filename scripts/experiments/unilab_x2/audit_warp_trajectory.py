"""Replay identical standing controls on one specified GPU, without learning."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import mujoco
import numpy as np
import warp as wp
import mujoco_warp as mw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', required=True)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--initial', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=200)
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != a.gpu:
        raise ValueError('Explicit GPU mask must match --gpu')
    if a.out.exists():
        raise FileExistsError(a.out)
    m = mujoco.MjModel.from_xml_path(str(a.scene.resolve()))
    initial = np.load(a.initial)
    d = mujoco.MjData(m)
    d.qpos[:] = initial['qpos']
    d.qvel[:] = 0
    d.ctrl[:] = initial['ctrl']
    mujoco.mj_forward(m, d)
    wp.init()
    model = mw.put_model(m)
    state = mw.put_data(m, d, nworld=1, nconmax=256, njmax=1024)
    with wp.ScopedCapture() as capture:
        mw.step(model, state)
    qpos, qvel = [state.qpos.numpy()[0].copy()], [state.qvel.numpy()[0].copy()]
    for _ in range(a.steps):
        wp.capture_launch(capture.graph)
        qpos.append(state.qpos.numpy()[0].copy())
        qvel.append(state.qvel.numpy()[0].copy())
    qpos, qvel = np.asarray(qpos), np.asarray(qvel)
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError('Nonfinite trajectory')
    np.savez_compressed(a.out, qpos=qpos, qvel=qvel)
    import importlib.metadata as metadata
    a.out.with_suffix('.json').write_text(json.dumps({
        'mujoco': mujoco.__version__, 'mujoco_warp': metadata.version('mujoco-warp'),
        'gpu': a.gpu, 'steps': a.steps, 'timestep': m.opt.timestep,
        'initial_sha256': hashlib.sha256(a.initial.read_bytes()).hexdigest(),
        'scene_sha256': hashlib.sha256(a.scene.read_bytes()).hexdigest(),
        'nconmax': 256, 'njmax': 1024,
        'scope': 'fixed standing control; physics validation only, not a performance test'
    }, indent=2) + '\n')
    print(f'{mujoco.__version__}: {a.steps} Warp steps completed', flush=True)


if __name__ == '__main__':
    main()
