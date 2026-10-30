"""Collect a fixed real-physics movement and transition retention set; not an acceptance test."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint', 'dataset', 'reference', 'standing-pose', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--num-envs', type=int, default=8)
    p.add_argument('--preserve-moving-feedback', action='store_true', help='Collect the complete BPTT movement-feedback policy')
    a = p.parse_args()
    python = Path('/home/2025-Long.xh/HHD/venvs/supermimic/bin/python')
    if Path(sys.prefix).resolve() != python.parent.parent.resolve() or os.path.realpath(sys.executable) != os.path.realpath(python):
        p.error(f'Use the dedicated server interpreter: {python}')
    if a.gpu < 0 or a.num_envs <= 0 or a.out.exists():
        p.error('Specify a valid GPU and environment count and a new output directory')
    for path in (a.checkpoint, a.dataset / 'walk.npz', a.reference, a.standing_pose):
        if not path.is_file():
            p.error(f'Missing required input: {path}')
    a.out.mkdir(parents=True, exist_ok=False)
    record = {'status': 'initializing', 'argv': sys.argv, 'started_epoch': time.time(), 'gpu': a.gpu,
              'source_sha256': sha(a.checkpoint), 'reference_sha256': sha(a.reference),
              'standing_pose_sha256': sha(a.standing_pose), 'walk_sha256': sha(a.dataset / 'walk.npz'),
              'entry_sha256': sha(Path(__file__)), 'seed': a.seed, 'num_envs': a.num_envs,
              'preserve_moving_feedback': a.preserve_moving_feedback,
              'scope': ('fixed complete-policy control retention; movement tracking and full control are not accepted'
                        if a.preserve_moving_feedback else 'fixed sampled original-policy retention states; movement tracking and full control are not accepted')}
    def save_report():
        tmp = a.out / 'report.json.tmp'
        tmp.write_text(json.dumps(record, indent=2) + '\n')
        tmp.replace(a.out / 'report.json')
    save_report()
    env = None
    raw = {k: [] for k in ('state_before', 'action', 'executed_action', 'command', 'feedback',
                           'phase_start_s', 'time_s', 'terminated', 'timeout')}
    xi_initial = None
    final_state = None

    def save_raw():
        if xi_initial is None or final_state is None:
            return
        import torch
        arrays = {key: torch.stack(values) if values else None for key, values in raw.items()}
        arrays.update(final_state=final_state, xi=xi_initial, env_id=torch.arange(a.num_envs),
            source_sha256=record['source_sha256'], reference_sha256=record['reference_sha256'],
            standing_pose_sha256=record['standing_pose_sha256'], walk_sha256=record['walk_sha256'],
            preserve_moving_feedback=a.preserve_moving_feedback, moving_projection_sha256=record.get('moving_projection_sha256'),
            dt=record.get('dt'), alignment='row t: before state, policy and executed action, command and feedback; next state is row t+1 or final_state',
            xi_scope='one value per environment; checked unchanged before and after each completed step')
        target = a.out / 'raw.pt'
        temporary = a.out / 'raw.pt.tmp'
        torch.save(arrays, temporary)
        temporary.replace(target)
        record['raw_path'] = str(target.resolve())
        record['raw_sha256'] = sha(target)
        record['raw_transitions'] = len(raw['state_before'])

    try:
        active = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        record['gpu_processes'] = active
        record['gpu_info'] = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-gpu=index,name,uuid,driver_version,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
        if active:
            raise RuntimeError('Selected GPU has active compute processes')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
        os.environ['SM_ROBOT'] = 'x2_ultra'
        os.environ['MUJOCO_GL'] = 'egl'
        for key in ('PYTHONHOME', 'SM_ROBOT_SPEC', 'SM_SOLE_BOX_CONFIG', 'SM_GPU_PINNED'):
            os.environ.pop(key, None)
        import torch
        from supermimic.envs import SuperMimicEnvCfgArgs, make_env
        from supermimic.envs.sim_context import SimContext
        from supermimic.envs.tracking_env_cfg import FOOT_GEOM_REGEX, ANCHOR_BODY
        from supermimic.eval.benchmark import load_policy_from_checkpoint
        from supermimic.locomotion.config import VelocityCfg
        from supermimic.locomotion.keyboard import CommandMailbox
        from supermimic.locomotion.task import measured_velocity
        from supermimic.policy.standing import StandingContext

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Expected exactly one visible CUDA device')
        source = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
        if source.get('standing_reference', {}).get('sha256') != record['reference_sha256']:
            raise ValueError('Standing reference does not match source checkpoint')
        policy, saved = load_policy_from_checkpoint(a.checkpoint, 'cuda:0')
        if not hasattr(policy, 'standing_skill') or policy.standing_skill.feedback_dim != 6:
            raise ValueError('A six-input standing expert is required')
        projection_sha = None
        if a.preserve_moving_feedback:
            if not hasattr(policy, 'velocity_feedback_weight') or source.get('standing_skill', {}).get('moving_feedback') is not True:
                raise ValueError('Preserving movement requires an enabled velocity feedback projection')
            records = [(key, source.get(key)) for key in ('supermimic_velocity_feedback', 'supermimic_composed_feedback')
                       if isinstance(source.get(key), dict)]
            if len(records) != 1:
                raise ValueError('Expected exactly one movement or composed BPTT record')
            kind, meta = records[0]
            if (meta.get('algorithm') != 'supermimic_bptt' or not isinstance(meta.get('actor_updates'), int)
                    or meta['actor_updates'] <= 0 or meta.get('reference_sha256') != record['reference_sha256']
                    or meta.get('standing_pose_sha256') != record['standing_pose_sha256']):
                raise ValueError('Invalid movement BPTT provenance, reference or pose')
            if kind == 'supermimic_composed_feedback':
                components = source.get('composed_training')
                if not isinstance(components, dict) or set(components) != {'stop', 'moving'}:
                    raise ValueError('Composed source lacks component training records')
                common = meta.get('source_common_sha')
                stop, moving = components['stop'], components['moving']
                evidence = meta.get('stop_pose_evidence', {})
                info = meta.get('components', {})
                if (not common or any(not isinstance(v, dict) or v.get('algorithm') != 'supermimic_bptt'
                        or v.get('source_sha256') != common or v.get('reference_sha256') != record['reference_sha256']
                        or not isinstance(v.get('actor_updates'), int) or v['actor_updates'] <= 0 for v in (stop, moving))
                        or meta['actor_updates'] != stop['actor_updates'] + moving['actor_updates']
                        or moving.get('standing_pose_sha256') != record['standing_pose_sha256']
                        or (stop.get('standing_pose_sha256') and stop['standing_pose_sha256'] != record['standing_pose_sha256'])
                        or not isinstance(evidence, dict) or evidence.get('standing_pose_sha256') != record['standing_pose_sha256']
                        or not isinstance(info, dict) or not isinstance(info.get('base'), dict)
                        or info['base'].get('sha256') != common):
                    raise ValueError('Invalid composed component provenance')
            elif not meta.get('source_sha256'):
                raise ValueError('Movement BPTT source hash missing')
            projection_sha = hashlib.sha256(policy.velocity_feedback_weight.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        record['moving_projection_sha256'] = projection_sha
        cfg = SuperMimicEnvCfgArgs(npz_files=[str(a.dataset / 'walk.npz')], num_envs=a.num_envs,
                                   seed=a.seed, future_frames=saved['env']['future_frames'], episode_length_s=71.,
                                   action_noise_std=0., obs_noise=False, reset_noise=False, domain_randomization=False,
                                   record_transitions=False, anchor_obs=saved['env']['anchor_obs'],
                                   velocity=VelocityCfg(**saved['velocity']))
        env = make_env(cfg, device='cuda:0')
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
        torch.backends.cudnn.fp32_precision = 'ieee'
        env.cfg.auto_reset = False
        cmd = env.command_manager.get_term('motion')
        cmd.is_velocity[:] = True
        cmd.force_standing_reset = True
        cmd.set_manual_command((0., 0., 0.))
        observations, _ = env.reset()
        context, mailbox = StandingContext(cmd, env), CommandMailbox()
        sim_context = SimContext(env, FOOT_GEOM_REGEX, ANCHOR_BODY)
        xi_initial = sim_context.read().detach().cpu().clone()
        final_state = cmd.state_io.read().detach().cpu().clone()
        action_term = env.action_manager.get_term('joint_pos')
        schedule = [(0., (0.,0.,0.)), (3.,(.3,0.,0.)), (10.,(0.,0.,0.)),
                    (15.,(-.2,0.,0.)), (22.,(0.,0.,0.)), (27.,(0.,0.,.4)),
                    (34.,(0.,0.,0.)), (39.,(0.,0.,-.4)), (46.,(0.,0.,0.)),
                    (51.,(.6,0.,0.)), (58.,(0.,0.,0.))]
        steps = round(66. / env.step_dt)
        if abs(steps * env.step_dt - 66.) > 1e-6:
            raise ValueError('The evaluator duration must align to the physical time step')
        samples = {k: [] for k in ('obs', 'command', 'feedback', 'source_action', 'age', 'env_id')}
        total = torch.zeros(a.num_envs, 3, device='cuda')
        phase_count = 0
        phase_sums = []
        previous_phase = None
        record.update(status='collecting', dt=env.step_dt, expected_steps=steps,
                      original_moving_feedback_disabled=not a.preserve_moving_feedback, control_matmul_precision='ieee',
                      noise=False, auto_reset=False, schedule=schedule,
                      sampling='every ten steps plus every near-zero ramp step',
                      torch=torch.__version__, cuda_runtime=torch.version.cuda,
                      cuda_device=torch.cuda.get_device_name(0), interpreter=sys.executable)
        save_report()
        terminated_any = torch.zeros(a.num_envs, dtype=torch.bool, device='cuda')
        timeout_any = torch.zeros_like(terminated_any)
        with torch.no_grad():
            for step in range(steps):
                age = step * env.step_dt
                phase, target = next((start, values) for start, values in reversed(schedule) if age >= start)
                if previous_phase != phase:
                    if previous_phase is not None:
                        phase_sums.append({'start': previous_phase, 'samples': phase_count,
                                           'mean_velocity': (total / max(phase_count, 1)).cpu().tolist()})
                    total.zero_(); phase_count = 0; previous_phase = phase
                mailbox.put(target, now=age)
                cmd.set_manual_command(mailbox.read(env.step_dt, now=age))
                actual = cmd.velocity_command.clone()
                obs = observations['policy'].clone()
                feedback = context.features(actual).clone()
                action = policy.velocity_action(obs, actual, feedback, disable_moving_feedback=not a.preserve_moving_feedback)
                physical_before = cmd.state_io.read().detach().cpu().clone()
                if not torch.equal(sim_context.read().detach().cpu(), xi_initial):
                    raise RuntimeError('Simulation context changed before step; constant xi trace rejected')
                if not all(torch.isfinite(t).all() for t in (obs, actual, feedback, action)):
                    raise FloatingPointError('Non-finite rollout observation, command, feedback or action')
                norms = actual.square().sum(-1).sqrt()
                retain = (step % 10 == 0) or bool(((norms > 0) & (norms < .1)).any())
                if retain:
                    samples['obs'].append(obs.cpu())
                    samples['command'].append(actual.cpu())
                    samples['feedback'].append(feedback.cpu())
                    samples['source_action'].append(action.cpu())
                    samples['age'].append(torch.full((a.num_envs,), age))
                    samples['env_id'].append(torch.arange(a.num_envs))
                observations, _, term, timeout, _ = env.step(action)
                term, timeout = term.clone(), timeout.clone()
                terminated_any |= term; timeout_any |= timeout
                state = cmd.state_io.read()
                executed = action_term.executed_raw.detach().cpu().clone()
                final_state = state.detach().cpu().clone()
                raw['state_before'].append(physical_before)
                raw['action'].append(action.detach().cpu().clone())
                raw['executed_action'].append(executed)
                raw['command'].append(actual.detach().cpu().clone())
                raw['feedback'].append(feedback.detach().cpu().clone())
                raw['phase_start_s'].append(torch.tensor(phase, dtype=torch.float64))
                raw['time_s'].append(torch.tensor(age, dtype=torch.float64))
                raw['terminated'].append(term.detach().cpu().clone())
                raw['timeout'].append(timeout.detach().cpu().clone())
                if not torch.equal(sim_context.read().detach().cpu(), xi_initial):
                    raise RuntimeError('Simulation context changed after step; constant xi trace rejected')
                if not torch.isfinite(state).all():
                    raise FloatingPointError('Non-finite physical state')
                if age - phase >= 1.:
                    total += measured_velocity(state); phase_count += 1
                if (term | timeout).any():
                    break
        if previous_phase is not None:
            phase_sums.append({'start': previous_phase, 'samples': phase_count,
                               'mean_velocity': (total / max(phase_count, 1)).cpu().tolist()})
        save_raw()
        result = {k: torch.cat(v, 0) for k, v in samples.items()}
        norms = result['command'].square().sum(-1).sqrt()
        expected = result['command'].new_tensor([[.3,0,0],[-.2,0,0],[0,0,.4],[0,0,-.4],[.6,0,0]])
        names = ('forward', 'backward', 'left', 'right', 'jog')
        coverage = {'zero': int((norms == 0).sum()), 'near_zero': int(((norms > 0) & (norms < .1)).sum())}
        coverage.update({name: int(torch.isclose(result['command'], c, atol=1e-6, rtol=0).all(-1).sum()) for name, c in zip(names, expected)})
        failed = bool(terminated_any.any() or timeout_any.any() or step + 1 != steps or not all(coverage.values()))
        if sha(a.checkpoint) != record['source_sha256'] or sha(a.reference) != record['reference_sha256'] or sha(a.standing_pose) != record['standing_pose_sha256']:
            raise RuntimeError('An input changed during collection')
        record.update(status='failed' if failed else 'collected', actual_steps=step+1, coverage=coverage,
                      terminated=terminated_any.cpu().tolist(), timeout=timeout_any.cpu().tolist(),
                      phase_velocity=phase_sums, finished_epoch=time.time())
        if not failed:
            result.update(source_sha256=record['source_sha256'], reference_sha256=record['reference_sha256'],
                          standing_pose_sha256=record['standing_pose_sha256'], seed=a.seed,
                          scope=record['scope'], sampling=record['sampling'],
                          preserve_moving_feedback=a.preserve_moving_feedback, moving_projection_sha256=projection_sha)
            path = a.out / 'retention.pt'
            torch.save(result, path)
            record.update(retention_path=str(path.resolve()), retention_sha256=sha(path), retention_samples=len(result['age']))
        save_report()
    except BaseException as error:
        save_raw()
        (a.out / 'retention.pt').unlink(missing_ok=True)
        record.update(status='failed', error=repr(error), finished_epoch=time.time())
        save_report()
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
