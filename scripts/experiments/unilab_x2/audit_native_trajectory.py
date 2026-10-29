"""Replay the same initial state and bounded control in native MuJoCo."""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--initial', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=200)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    model = mujoco.MjModel.from_xml_path(str(a.scene.resolve()))
    initial = np.load(a.initial)
    data = mujoco.MjData(model)
    data.qpos[:] = initial['qpos']
    data.qvel[:] = 0
    data.ctrl[:] = initial['ctrl']
    mujoco.mj_forward(model, data)
    qpos, qvel = [data.qpos.copy()], [data.qvel.copy()]
    for _ in range(a.steps):
        mujoco.mj_step(model, data)
        qpos.append(data.qpos.copy())
        qvel.append(data.qvel.copy())
    qpos, qvel = np.asarray(qpos), np.asarray(qvel)
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError('Nonfinite trajectory')
    np.savez_compressed(a.out, qpos=qpos, qvel=qvel)
    a.out.with_suffix('.json').write_text(json.dumps({
        'mujoco': mujoco.__version__, 'steps': a.steps, 'timestep': model.opt.timestep,
        'initial_sha256': hashlib.sha256(a.initial.read_bytes()).hexdigest(),
        'scene_sha256': hashlib.sha256(a.scene.read_bytes()).hexdigest(),
        'control': 'constant default joint target; not a learned policy'}, indent=2) + '\n')
    print(f'{mujoco.__version__}: {a.steps} native steps completed', flush=True)


if __name__ == '__main__':
    main()
