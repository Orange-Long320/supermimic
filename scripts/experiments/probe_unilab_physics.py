"""Bounded X2 physics probe; not a UniLab task or training benchmark.

Export the existing mjlab scene, then run the identical XML and nominal standing
control with UniLab's pinned mjbatch or the project's MuJoCo Warp backend.
Different MuJoCo versions and omitted task/learner work limit comparisons.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['export', 'cpu', 'gpu'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--envs', type=int, default=256)
    p.add_argument('--threads', type=int, default=8)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    import mujoco
    import numpy as np
    xml = a.out / 'scene.xml'
    if a.mode == 'export':
        from supermimic.robots.paths import select_robot
        select_robot('x2_ultra', check_assets=True)
        from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg
        from supermimic.robots.dimensions import PROFILE
        from mjlab.scene import Scene
        cfg = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=[], num_envs=1))
        scene = Scene(cfg.scene, device='cpu')
        model = scene.compile()
        cfg.sim.mujoco.apply(model)
        d = mujoco.MjData(model)
        d.qpos[2] = PROFILE['standing_height']
        for name, pos in PROFILE['default_joint_pos'].items():
            matches = [i for i in range(model.njnt)
                       if model.joint(i).name.split('/')[-1] == name]
            if len(matches) != 1:
                raise ValueError((name, matches))
            d.qpos[model.jnt_qposadr[matches[0]]] = pos
        for i in range(model.nu):
            d.ctrl[i] = d.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]]
        mujoco.mj_forward(model, d)
        if xml.exists():
            raise FileExistsError(xml)
        mujoco.mj_saveLastXML(str(xml), model)
        np.savez(a.out / 'initial.npz', qpos=d.qpos, ctrl=d.ctrl)
        print(json.dumps({'nq': model.nq, 'nv': model.nv, 'nu': model.nu,
                          'timestep': model.opt.timestep}), flush=True)
        return
    model = mujoco.MjModel.from_xml_path(str(xml))
    initial = np.load(a.out / 'initial.npz')
    d = mujoco.MjData(model)
    d.qpos[:] = initial['qpos']
    d.ctrl[:] = initial['ctrl']
    mujoco.mj_forward(model, d)
    if a.mode == 'cpu':
        import mjbatch
        batch = mjbatch.Batch(model, a.envs, num_threads=a.threads)
        q = batch.bind('qpos')
        v = batch.bind('qvel')
        ctrl = batch.bind('ctrl')
        def reset():
            q[:] = initial['qpos']; v[:] = 0; ctrl[:] = initial['ctrl']
        def step():
            batch.step(nstep=10)
        def sync():
            pass
        def final_q():
            return q.copy()
    else:
        import warp as wp
        import mujoco_warp as mw
        wp.init()
        m = mw.put_model(model)
        wd = mw.put_data(model, d, nworld=a.envs)
        q0 = wp.array(np.repeat(initial['qpos'][None], a.envs, axis=0), dtype=wp.float32, device='cuda:0')
        def reset():
            wp.copy(wd.qpos, q0); wd.qvel.zero_()
        with wp.ScopedCapture() as cap:
            for _ in range(10):
                mw.step(m, wd)
        def step():
            wp.capture_launch(cap.graph)
        def sync():
            wp.synchronize()
        def final_q():
            return wd.qpos.numpy()
    reset(); step(); sync()
    timings = []
    for _ in range(10):
        reset(); sync()
        start = time.perf_counter()
        step(); sync()
        timings.append(time.perf_counter() - start)
    state = final_q()
    if not np.isfinite(state).all():
        raise ValueError('Nonfinite simulation state')
    result = dict(mode=a.mode, envs=a.envs, threads=a.threads if a.mode == 'cpu' else None,
                  mujoco=mujoco.__version__, xml_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),
                  nq=model.nq, nv=model.nv, nu=model.nu, timestep=model.opt.timestep,
                  samples_s=timings, median_s=float(np.median(timings)),
                  physics_steps_per_s=float(a.envs * 10 / np.median(timings)),
                  finite=True, final_root_z=float(state[0, 2]),
                  scope='10 physics substeps from nominal standing; no policy, reward, reset or learning cost')
    package = 'mjbatch-uni' if a.mode == 'cpu' else 'mujoco-warp'
    result[package] = importlib.metadata.version(package)
    target = a.out / f'{a.mode}-{a.envs}-{a.threads if a.mode == "cpu" else 0}.json'
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
