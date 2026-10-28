"""Server-only, audited IMG4868 fixed-box training: capacity smoke, then four hours.

Input audits must describe the exact dense reference and native execution. This
entry refuses incomplete references before allocating any GPU resources.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_reference(reference, geometry_path, force_path, climb_path):
    reference_hash = digest(reference)
    geometry = json.loads(geometry_path.read_text())
    if geometry.get('source_sha256') != reference_hash or not geometry.get('geometry_gates_passed'):
        raise ValueError('Exact reference has not passed independent geometry checks')
    force = json.loads(force_path.read_text())
    if force.get('source_sha256') != reference_hash or not force['summary']['1.0']['passed']:
        raise ValueError('Dense reference has not passed all sampled force/motor checks at its own timing')
    climb = json.loads(climb_path.read_text())
    if not climb.get('continuous_climb_passed'):
        raise ValueError('Continuous climb and terminal hold have not passed')
    trace = Path(climb['trace'])
    if digest(trace / 'trajectory.npz') != climb['trajectory_sha256']:
        raise ValueError('Continuous trace changed since its audit')
    report = json.loads((trace / 'result.json').read_text())
    native_model = Path(report['configuration']['model'])
    if digest(native_model) != climb['model_sha256'] or force['model_sha256'] != climb['model_sha256']:
        raise ValueError('Force and continuous audits used different native models')
    import numpy as np
    with np.load(reference) as selected, np.load(report['configuration']['reference']) as executed:
        np.testing.assert_array_equal(selected['qpos'], executed['qpos'])
        np.testing.assert_allclose(float(selected['fps'].reshape(-1)[0]), float(executed['fps'].reshape(-1)[0]), rtol=0, atol=0)
    return geometry, native_model, reference_hash


def require_idle_gpu(gpu):
    pids = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    if pids:
        raise RuntimeError(f'GPU {gpu} is occupied; no training launched (PIDs {pids})')


def main():
    # An isolated deployment must import its own source, not the editable install
    # belonging to another running experiment in the shared checkout.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--geometry-audit', type=Path, required=True)
    p.add_argument('--force-scan', type=Path, required=True)
    p.add_argument('--climb-audit', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--run-name', required=True)
    p.add_argument('--phase', choices=['smoke', 'train'], required=True)
    p.add_argument('--num-envs', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--capacity-proof', type=Path)
    p.add_argument('--sole-profile', type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--validate-only', action='store_true')
    a = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', a.run_name) or min(a.num_envs, a.batch_size) < 1 or a.gpu < 0:
        p.error('Invalid run name, capacity or GPU')
    geometry, native_path, reference_hash = validate_reference(a.reference, a.geometry_audit, a.force_scan, a.climb_audit)
    if a.validate_only:
        print(json.dumps({'reference_sha256': reference_hash, 'reference_gates_passed': True}))
        return
    if Path(sys.prefix).resolve() != Path('/home/2025-Long.xh/HHD/venvs/supermimic').resolve():
        raise RuntimeError('Run with the dedicated server supermimic environment')
    sole_hash = digest(a.sole_profile)
    proof = None
    if a.phase == 'train':
        if a.capacity_proof is None:
            p.error('Four-hour training requires a completed capacity smoke')
        proof = json.loads(a.capacity_proof.read_text())
        expected = {'reference_sha256': reference_hash, 'sole_sha256': sole_hash,
                    'source_commit': a.source_commit, 'num_envs': a.num_envs, 'batch_size': a.batch_size}
        if proof.get('status') != 'completed' or proof.get('phase') != 'smoke' or any(proof.get(k) != v for k, v in expected.items()):
            p.error('Capacity smoke does not match this source, reference and configuration')
        if proof.get('updates', 0) < 1 or digest(proof['checkpoint']) != proof['checkpoint_sha256']:
            p.error('Capacity smoke has no verified updated checkpoint')
    require_idle_gpu(a.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(a.gpu)
    os.environ['SM_SOLE_BOX_CONFIG'] = str(a.sole_profile.resolve())
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra', check_assets=True)
    import mujoco
    import numpy as np
    import torch
    from mjlab.scene import Scene
    from supermimic.config import training_config
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.data.physics_motion import export_qpos_motion
    from supermimic.envs.box_scene import FixedBox
    from supermimic.envs.tracking_env_cfg import SuperMimicEnvCfgArgs, make_env_cfg
    from supermimic.training.loop import Trainer

    box = FixedBox(**geometry['box'])
    cfg = training_config('smoke')
    cfg.robot = 'x2_ultra'
    cfg.run_name = a.run_name
    cfg.env.box = box
    cfg.env.sole_profile = a.sole_profile.resolve()
    cfg.env.num_envs = a.num_envs
    cfg.env.anchor_obs = True
    cfg.env.domain_randomization = cfg.env.obs_noise = cfg.env.reset_noise = False
    cfg.env.action_noise_std = 0
    cfg.env.replay_capacity = 512
    cfg.wm_trainer.batch_size = cfg.policy_trainer.batch_size = a.batch_size
    cfg.policy_trainer.obs_noise = False
    cfg.schedule.lr_final_ratio = 1
    cfg.schedule.checkpoint_interval = 500 if a.phase == 'train' else 100
    cfg.schedule.total_iterations = 10_000_000 if a.phase == 'train' else 100
    cfg.schedule.max_hours = 4 if a.phase == 'train' else 1/6
    # Compare the actual training scene to the independently executed model.
    env_cfg = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=[], box=box, sole_profile=cfg.env.sole_profile))
    model = Scene(env_cfg.scene, device='cpu').compile()
    env_cfg.sim.mujoco.apply(model)
    native = mujoco.MjModel.from_binary_path(str(native_path))
    for field in ('nq', 'nv', 'nu', 'ngeom'):
        if getattr(model, field) != getattr(native, field):
            raise ValueError(f'Native/training model differs: {field}')
    for field in ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'jnt_range', 'dof_armature',
                  'actuator_gainprm', 'actuator_biasprm', 'actuator_forcerange', 'actuator_ctrlrange',
                  'geom_type', 'geom_bodyid', 'geom_size', 'geom_pos', 'geom_quat', 'geom_friction',
                  'geom_contype', 'geom_conaffinity', 'geom_condim', 'geom_solref', 'geom_solimp'):
        np.testing.assert_allclose(getattr(model, field), getattr(native, field), rtol=0, atol=1e-12, err_msg=field)
    for field in ('timestep', 'integrator', 'iterations', 'ls_iterations', 'cone', 'solver'):
        if getattr(model.opt, field) != getattr(native.opt, field):
            raise ValueError(f'Native/training physics differs: {field}')

    run = Path(cfg.log_root) / a.run_name
    run.mkdir(parents=True, exist_ok=False)
    (run/'sole-profile.json').write_bytes(a.sole_profile.read_bytes())
    cfg.env.sole_profile = (run/'sole-profile.json').resolve()
    for label, path in (('geometry-audit', a.geometry_audit), ('force-scan', a.force_scan), ('climb-audit', a.climb_audit)):
        (run/(label+'.json')).write_bytes(path.read_bytes())
    source_root = Path(__file__).resolve().parents[2]
    source_hashes = {str(path.relative_to(source_root)): digest(path) for path in sorted((source_root/'src').rglob('*.py'))}
    source_hashes[str(Path(__file__).relative_to(source_root))] = digest(__file__)
    (run/'source-sha256.json').write_text(json.dumps(source_hashes, indent=2)+'\n')
    with np.load(a.reference) as motion:
        poses, fps = motion['qpos'].copy(), float(motion['fps'].reshape(-1)[0])
    cfg.env.episode_length_s = (len(poses)-1)/fps + 2
    cfg.data.npz_dir = run / 'input'
    cfg.data.clips = ['box45']
    export_qpos_motion(load_g1_model(), poses, cfg.data.npz_dir/'box45.npz', fps, source=a.reference)
    if proof:
        cfg.resume = Path(proof['checkpoint'])
    record = {'phase': a.phase, 'status': 'running', 'gpu': a.gpu, 'pid': os.getpid(),
              'source_commit': a.source_commit, 'reference_sha256': reference_hash,
              'sole_sha256': sole_hash, 'num_envs': a.num_envs, 'batch_size': a.batch_size,
              'argv': sys.argv, 'python': sys.version, 'executable': str(Path(sys.executable).resolve()),
              'packages': {name: importlib.metadata.version(name) for name in ('torch', 'mujoco', 'mujoco-warp', 'warp-lang')},
              'config': asdict(cfg)}
    (run/'launch.json').write_text(json.dumps(record, indent=2, default=str)+'\n')
    require_idle_gpu(a.gpu)
    started = time.monotonic()
    stop = threading.Event()
    device_samples = []
    def sample_device():
        while not stop.is_set():
            try:
                value = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True, timeout=5)
                device_samples.append(int(value.strip()))
            except (subprocess.SubprocessError, ValueError):
                pass
            stop.wait(1)
    monitor = threading.Thread(target=sample_device, daemon=True)
    monitor.start()
    torch.cuda.reset_peak_memory_stats()
    trainer = None
    try:
        trainer = Trainer(cfg)
        trainer.run()
        checkpoint = run/'ckpt_latest.pt'
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert all(torch.isfinite(v).all() for name in ('wm', 'policy', 'critic') for v in ck[name].values() if torch.is_tensor(v))
        record.update(status='completed', checkpoint=str(checkpoint.resolve()), checkpoint_sha256=digest(checkpoint),
                      iteration=ck['iteration'], updates=ck['num_updates'])
    finally:
        stop.set()
        monitor.join(timeout=6)
        if trainer is not None:
            trainer.env.close()
            trainer.writer.close()
        if record['status'] != 'completed':
            record['status'] = 'failed'
        record.update(elapsed_s=time.monotonic()-started, torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                      torch_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                      device_peak_sampled_mib=max(device_samples, default=None))
        (run/'capacity.json').write_text(json.dumps(record, indent=2, default=str)+'\n')


if __name__ == '__main__':
    main()
