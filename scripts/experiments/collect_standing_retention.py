"""Collect a gated real-physics standing retention set, without training."""

from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'dataset', 'reference', 'standing-pose', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--seconds', type=float, default=20.)
    p.add_argument('--num-envs', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    python = Path('/home/2025-Long.xh/HHD/venvs/supermimic/bin/python')
    if Path(sys.prefix).resolve() != python.parent.parent.resolve() or os.path.realpath(sys.executable) != os.path.realpath(python):
        p.error(f'Use the dedicated server interpreter: {python}')
    if a.gpu < 0 or a.seconds <= 1 or a.num_envs <= 0:
        p.error('GPU, duration and environment count are invalid')
    if a.out.exists():
        p.error('Output directory already exists')
    for path in (a.checkpoint, a.reference, a.standing_pose, a.dataset / 'walk.npz'):
        if not path.is_file():
            p.error(f'Required input missing: {path}')
    a.out.mkdir(parents=True, exist_ok=False)
    record = {'status': 'initializing', 'argv': sys.argv, 'started_epoch': time.time(), 'gpu': a.gpu,
              'checkpoint': str(a.checkpoint.resolve()), 'checkpoint_sha256': sha256(a.checkpoint),
              'reference_sha256': sha256(a.reference), 'standing_pose_sha256': sha256(a.standing_pose),
              'walk_sha256': sha256(a.dataset / 'walk.npz'), 'entry_sha256': sha256(Path(__file__)),
              'scope': 'zero-command physical standing retention only; full control acceptance remains pending'}
    def report():
        tmp = a.out / 'report.json.tmp'
        tmp.write_text(json.dumps(record, indent=2) + '\n')
        tmp.replace(a.out / 'report.json')
    report()
    env = None
    raw = {'observation': [], 'feedback': [], 'action': [], 'state': [], 'age_s': [], 'terminated': [], 'timeout': []}
    try:
        processes = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        gpu_info = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-gpu=index,name,uuid,driver_version,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
        record.update(gpu_processes=processes, gpu_info=gpu_info)
        if processes:
            raise RuntimeError('Selected GPU has active compute processes')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
        os.environ['SM_ROBOT'] = 'x2_ultra'
        os.environ['MUJOCO_GL'] = 'egl'
        for key in ('PYTHONHOME', 'SM_ROBOT_SPEC', 'SM_SOLE_BOX_CONFIG', 'SM_GPU_PINNED'):
            os.environ.pop(key, None)
        import numpy as np
        import torch
        from supermimic.envs import SuperMimicEnvCfgArgs, make_env
        from supermimic.eval.benchmark import load_policy_from_checkpoint
        from supermimic.kinematics.rotations import heading_yaw
        from supermimic.locomotion.config import VelocityCfg
        from supermimic.locomotion.task import measured_velocity
        from supermimic.policy.standing import StandingContext
        from supermimic.robots.dimensions import PROFILE, NQ

        torch.set_num_threads(2)
        torch.manual_seed(a.seed)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Expected exactly one visible CUDA device')
        record.update(interpreter=sys.executable, python=sys.version, torch=torch.__version__, cuda_runtime=torch.version.cuda,
                      cuda_device=torch.cuda.get_device_name(0))
        policy, saved = load_policy_from_checkpoint(a.checkpoint, 'cuda:0')
        if not hasattr(policy, 'standing_skill') or policy.standing_skill.feedback_dim != 6:
            raise ValueError('Checkpoint must contain a six-input standing expert')
        reference = np.load(a.reference)
        if 'joint_pos' not in reference or not len(reference['joint_pos']):
            raise ValueError('Standing reference lacks joint positions')
        pose = json.loads(a.standing_pose.read_text())['joint_positions_rad']
        arm_ids = [i for i, name in enumerate(PROFILE['joint_names']) if any(k in name for k in ('shoulder', 'elbow', 'wrist'))]
        if not arm_ids:
            raise ValueError('No arm joints found in robot profile')
        arm_target = torch.tensor([pose[PROFILE['joint_names'][i]] for i in arm_ids], device='cuda')
        cfg = SuperMimicEnvCfgArgs(npz_files=[str(a.dataset / 'walk.npz')], num_envs=a.num_envs, seed=a.seed,
                                   future_frames=saved['env']['future_frames'], episode_length_s=a.seconds + 5,
                                   action_noise_std=0., obs_noise=False, reset_noise=False, domain_randomization=False,
                                   record_transitions=False, anchor_obs=saved['env']['anchor_obs'],
                                   velocity=VelocityCfg(**saved['velocity']))
        env = make_env(cfg, device='cuda:0')
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
        torch.backends.cudnn.fp32_precision = 'ieee'
        record['matmul_precision'] = torch.backends.cuda.matmul.fp32_precision
        env.cfg.auto_reset = False
        cmd = env.command_manager.get_term('motion')
        cmd.is_velocity[:] = True
        cmd.force_standing_reset = True
        cmd.set_manual_command((0., 0., 0.))
        observations, _ = env.reset()
        context = StandingContext(cmd, env)
        zero = torch.zeros(a.num_envs, 3, device='cuda')
        steps = round(a.seconds / env.step_dt)
        if steps * env.step_dt < a.seconds - 1e-6:
            raise ValueError('Duration must be an integer multiple of the simulation step')
        speed_sum = torch.zeros(a.num_envs, device='cuda')
        yaw_sum = torch.zeros_like(speed_sum)
        arm_square_sum = torch.zeros_like(speed_sum)
        arm_pose_square_sum = torch.zeros_like(speed_sum)
        max_dist = torch.zeros_like(speed_sum)
        max_heading = torch.zeros_like(speed_sum)
        origin = origin_yaw = None
        net_dist = torch.zeros_like(speed_sum)
        net_heading = torch.zeros_like(speed_sum)
        counts = 0
        any_terminated = torch.zeros(a.num_envs, device='cuda', dtype=torch.bool)
        any_timeout = torch.zeros_like(any_terminated)
        record.update(status='collecting', dt=env.step_dt, expected_steps=steps, seed=a.seed, num_envs=a.num_envs,
                      action_override=False, observation_noise=False, reset_noise=False, domain_randomization=False,
                      auto_reset=False, reference_arm_target='standing-pose JSON',
                      arm_joint_names=[PROFILE['joint_names'][i] for i in arm_ids])
        report()
        with torch.no_grad():
            for step in range(steps):
                age = step * env.step_dt
                cmd.set_manual_command((0., 0., 0.))
                feedback = context.features(zero).clone()
                obs = observations['policy'].clone()
                state_before = cmd.state_io.read().clone()
                action = policy.velocity_action(obs, zero, feedback)
                if not all(torch.isfinite(t).all() for t in (obs, feedback, state_before, action)):
                    raise FloatingPointError('Non-finite observation, feedback, state or action')
                observations, _, terminated, timeout, _ = env.step(action)
                terminated, timeout = terminated.clone(), timeout.clone()
                state = cmd.state_io.read().clone()
                any_terminated |= terminated
                any_timeout |= timeout
                raw['observation'].append(obs.detach().cpu().clone())
                raw['feedback'].append(feedback.detach().cpu().clone())
                raw['action'].append(action.detach().cpu().clone())
                raw['state'].append(state_before.detach().cpu())
                raw['age_s'].append(age)
                raw['terminated'].append(terminated.cpu())
                raw['timeout'].append(timeout.cpu())
                if not torch.isfinite(state).all():
                    raise FloatingPointError('Non-finite physical state')
                if (step + 1) * env.step_dt >= 1.:
                    velocity = measured_velocity(state)
                    speed_sum += velocity[:, :2].norm(dim=-1)
                    yaw_sum += velocity[:, 2].abs()
                    arm_square_sum += state[:, NQ + 6:][:, arm_ids].square().mean(-1)
                    arm_pose_square_sum += (state[:, 7:NQ][:, arm_ids] - arm_target).square().mean(-1)
                    yaw = heading_yaw(state[:, 3:7])
                    if origin is None:
                        origin, origin_yaw = state[:, :2].clone(), yaw.clone()
                    net_dist = (state[:, :2] - origin).norm(dim=-1)
                    net_heading = torch.atan2(torch.sin(yaw - origin_yaw), torch.cos(yaw - origin_yaw)).abs()
                    max_dist = torch.maximum(max_dist, net_dist)
                    max_heading = torch.maximum(max_heading, net_heading)
                    counts += 1
                if (terminated | timeout).any():
                    break
        if counts == 0:
            raise RuntimeError('No physical samples at or after one second')
        metrics = {'mean_planar_speed_mps': (speed_sum / counts).cpu().tolist(),
                   'mean_absolute_yaw_rad_s': (yaw_sum / counts).cpu().tolist(),
                   'arm_velocity_rms_rad_s': (arm_square_sum / counts).sqrt().cpu().tolist(),
                   'arm_pose_rmse_rad': (arm_pose_square_sum / counts).sqrt().cpu().tolist(),
                   'net_displacement_m': net_dist.cpu().tolist(), 'max_displacement_m': max_dist.cpu().tolist(),
                   'net_heading_rad': net_heading.cpu().tolist(), 'max_heading_rad': max_heading.cpu().tolist(),
                   'terminated': any_terminated.cpu().tolist(), 'timeout': any_timeout.cpu().tolist(),
                   'physical_samples_per_env': counts}
        limits = {'mean_planar_speed_mps': .06, 'mean_absolute_yaw_rad_s': .08,
                  'arm_velocity_rms_rad_s': .12, 'arm_pose_rmse_rad': .2, 'net_displacement_m': .05,
                  'max_displacement_m': .1, 'net_heading_rad': .1, 'max_heading_rad': .15}
        failures = [key for key, limit in limits.items() if any(v > limit for v in metrics[key])]
        if any_terminated.any() or any_timeout.any():
            failures.append('fall_or_timeout')
        if len(raw['age_s']) != steps:
            failures.append('incomplete_duration')
        stacked = {k: torch.tensor(v) if k == 'age_s' else torch.stack(v) for k, v in raw.items()}
        stacked.update(checkpoint_sha256=record['checkpoint_sha256'], seed=a.seed, dt=env.step_dt,
                       description='state, obs, feedback and action are aligned before each physical step; termination is after it')
        raw_path = a.out / 'raw.pt'
        torch.save(stacked, raw_path)
        record.update(status='passed' if not failures else 'failed', metrics=metrics, thresholds=limits,
                      failures=failures, actual_steps=len(raw['age_s']), raw_sha256=sha256(raw_path))
        if not failures:
            chosen = torch.arange(0, steps, 10)
            output = {'obs': stacked['observation'][chosen].reshape(-1, stacked['observation'].shape[-1]),
                      'feedback': stacked['feedback'][chosen].reshape(-1, 6),
                      'source_action': stacked['action'][chosen].reshape(-1, stacked['action'].shape[-1]),
                      'age': stacked['age_s'][chosen, None].expand(-1, a.num_envs).reshape(-1),
                      'source_sha256': record['checkpoint_sha256'], 'seed': a.seed,
                      'reference_sha256': record['reference_sha256'], 'standing_pose_sha256': record['standing_pose_sha256'],
                      'scope': record['scope']}
            target = a.out / 'retention.pt'
            torch.save(output, target)
            record.update(retention_path=str(target.resolve()), retention_sha256=sha256(target), retention_samples=len(output['age']))
        record['finished_epoch'] = time.time()
        report()
    except BaseException as error:
        (a.out / 'retention.pt').unlink(missing_ok=True)
        if raw['age_s']:
            import torch
            partial = {k: torch.tensor(v) if k == 'age_s' else torch.stack(v) for k, v in raw.items()}
            partial.update(checkpoint_sha256=record['checkpoint_sha256'], incomplete=True)
            torch.save(partial, a.out / 'raw.pt')
            record['raw_sha256'] = sha256(a.out / 'raw.pt')
        record.update(status='failed', error=repr(error), finished_epoch=time.time())
        report()
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
