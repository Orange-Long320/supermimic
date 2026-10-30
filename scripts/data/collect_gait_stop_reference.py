"""Collect one simulated tracked gait followed by the checkpoint's standing policy.

Only a physically validated, continuous simulator trajectory is exported as a
candidate reference. This script does not train or alter robot state between phases.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'npz', 'standing-reference', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--stop-seconds', type=float, default=6.)
    p.add_argument('--num-envs', type=int, default=2)
    a = p.parse_args()
    expected = Path('/home/2025-Long.xh/HHD/venvs/supermimic').resolve()
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if Path(sys.prefix).resolve() != expected or not gpu or len(gpu.split(',')) != 1:
        p.error('Use the dedicated server environment and select one explicit GPU')
    if a.stop_seconds < 4 or a.num_envs < 1 or a.out.exists():
        p.error('Require at least four seconds, a positive environment count and a new output directory')
    for path in (a.checkpoint, a.npz, a.standing_reference):
        if not path.is_file():
            p.error(f'Missing input: {path}')
    import numpy as np
    import mujoco
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.kinematics.rotations import heading_yaw
    from supermimic.locomotion.config import VelocityCfg
    from supermimic.locomotion.task import measured_velocity, reference_commands
    from supermimic.policy.standing import StandingContext, standing_action
    from supermimic.robots.dimensions import NQ, PROFILE

    torch.manual_seed(a.seed)
    torch.set_num_threads(2)
    clip = np.load(a.npz, allow_pickle=False)
    ref = np.load(a.standing_reference, allow_pickle=False)
    names = [str(n) for n in clip['joint_names']]
    bodies = [str(n) for n in clip['body_names']]
    if names != list(PROFILE['joint_names']):
        raise ValueError('Clip joint order differs from robot profile')
    frame_count = len(clip['joint_pos'])
    fps = float(np.asarray(clip['fps']).reshape(-1)[0])
    if frame_count < 2 or fps <= 0:
        raise ValueError('Invalid clip length or frame rate')
    policy, saved = load_policy_from_checkpoint(a.checkpoint, 'cuda:0')
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if not saved.get('velocity', {}).get('enabled') or not getattr(policy, 'standing_skill', None):
        raise ValueError('Checkpoint needs both motion and standing experts')
    if checkpoint.get('standing_reference', {}).get('sha256') != digest(a.standing_reference):
        raise ValueError('Standing reference does not match checkpoint')
    cfg_args = SuperMimicEnvCfgArgs(npz_files=[str(a.npz)], num_envs=a.num_envs, seed=a.seed,
        future_frames=saved['env']['future_frames'], episode_length_s=frame_count / fps + a.stop_seconds + 3,
        action_noise_std=0., obs_noise=False, reset_noise=False, domain_randomization=False,
        record_transitions=False, sampling_mode='start', anchor_obs=saved['env']['anchor_obs'],
        velocity=VelocityCfg(**saved['velocity']))
    cfg = make_env_cfg(cfg_args)
    if 'clip_end' not in cfg.terminations or not cfg.terminations['clip_end'].time_out:
        raise ValueError('Expected clip_end timeout in native configuration')
    del cfg.terminations['clip_end']
    cfg.auto_reset = False
    configure_torch_backends()
    env = ManagerBasedRlEnv(cfg, device='cuda:0')
    torch.backends.cuda.matmul.fp32_precision = 'ieee'
    torch.backends.cudnn.fp32_precision = 'ieee'
    a.out.mkdir(parents=True, exist_ok=False)
    report = dict(status='collecting', source_clip=str(a.npz.resolve()), source_clip_sha256=digest(a.npz),
        checkpoint=str(a.checkpoint.resolve()), checkpoint_sha256=digest(a.checkpoint),
        standing_reference_sha256=digest(a.standing_reference), entry_sha256=digest(__file__),
        source_commit=os.environ.get('SM_SOURCE_COMMIT', 'unspecified'), argv=sys.argv,
        interpreter=sys.executable, torch_version=torch.__version__, cuda_runtime=torch.version.cuda,
        gpu=gpu, seed=a.seed, num_envs=a.num_envs, native_clip_end_timeout_removed=True,
        scope='Simulated policy-generated reference, not a human original or an accepted training input',
        reset_count=0, fps=fps, frame_count=frame_count, stop_seconds=a.stop_seconds,
        command_labels='tracking: reference velocity at the current clip frame; stop: zero',
        contact_scope='CPU geometric contact audit, not a measurement of contact force or stability')
    states, actions, phases, commands = [], [], [], []
    failures = []
    cmd = env.command_manager.get_term('motion')
    try:
        if abs(env.step_dt - 1 / fps) > 1e-6:
            raise ValueError('Clip FPS and simulation control frequency differ')
        cmd.is_velocity[:] = False
        cmd.force_standing_reset = False
        cmd.forced_frame[:] = 0
        observations, _ = env.reset()
        robot = env.scene['robot']
        limits = robot.data.soft_joint_pos_limits.clone()
        initial = cmd.state_io.read().clone()
        states.append(initial.cpu())
        report['first_goal_frame'] = cmd.frame.cpu().tolist()
        report['last_goal_frame'] = cmd.library.last_frame_of(cmd.frame).cpu().tolist()
        # The pinned env.reset writes physical frame 0, then computes the next
        # target at dt=0. First action therefore targets frame 1, not frame 0.
        if not (cmd.frame == 1).all() or not (cmd.library.last_frame_of(cmd.frame) == frame_count - 1).all():
            raise ValueError('Pinned next-frame goal indexing differs from expected reset flow')
        initial_ref = cmd.library.ref(torch.zeros_like(cmd.frame))
        expected_joint = initial_ref.joint_pos.clamp(limits[..., 0], limits[..., 1])
        if not torch.allclose(initial[:, 7:NQ], expected_joint, atol=1e-6, rtol=0):
            raise ValueError('Physical initialization does not match reference frame 0')
        tracking_steps = frame_count - 1
        report['tracking_steps'] = tracking_steps
        report['initial_physical_reference_frame'] = 0
        if limits.shape[1] != len(names):
            raise ValueError('Joint limit shape differs from source joint order')
        context = StandingContext(cmd, env)
        zero = torch.zeros(a.num_envs, 3, device='cuda')
        stop_steps = round(a.stop_seconds / env.step_dt)
        if abs(stop_steps * env.step_dt - a.stop_seconds) > 1e-6:
            raise ValueError('Stop duration must be an integer number of control steps')
        with torch.no_grad():
            for step in range(tracking_steps + stop_steps):
                tracking = step < tracking_steps
                if tracking:
                    if not (cmd.frame == step + 1).all():
                        raise RuntimeError('Tracking frame advanced unexpectedly')
                    action = policy(observations['policy'], observations['goal'])
                    label = reference_commands(cmd.library, cmd.frame).clone()
                else:
                    if step == tracking_steps:
                        cmd.is_velocity[:] = True
                        cmd.set_manual_command((0., 0., 0.))
                    action = standing_action(policy, observations['policy'], zero, context)
                    label = zero
                actions.append(action.detach().cpu())
                phases.append(0 if tracking else 1)
                commands.append(label.cpu().clone())
                observations, _, term, timeout, _ = env.step(action)
                states.append(cmd.state_io.read().clone().cpu())
                if (term | timeout).any():
                    failures.append(dict(step=step, terminated=term.cpu().tolist(), timeout=timeout.cpu().tolist()))
                    break
        s = torch.stack(states)
        act = torch.stack(actions)
        torch.save(dict(states=s, actions=act, phase=torch.tensor(phases), commands=torch.stack(commands),
                        stop_start_index=tracking_steps, failures=failures), a.out / 'raw.pt')
        report['raw_sha256'] = digest(a.out / 'raw.pt')
        report['steps_completed'] = len(actions)
        report['failures'] = failures
        if failures or len(actions) != tracking_steps + stop_steps:
            report['status'] = 'rejected_failure'
        else:
            x = s.to('cuda')
            # Compare the real tracking prefix against the original frames without replacing it.
            prefix_delta = x[:frame_count, :, 7:NQ] - torch.as_tensor(clip['joint_pos'], device='cuda')[:, None, :]
            report['tracking_joint_difference'] = dict(rmse=prefix_delta.square().mean((0, 2)).sqrt().tolist(),
                max_abs=prefix_delta.abs().amax((0, 2)).tolist())
            if [str(n) for n in ref['joint_names']] != names:
                raise ValueError('Standing reference joint order differs from clip')
            ref_arm = torch.as_tensor(ref['joint_pos'][0], device='cuda')[
                [i for i, n in enumerate(names) if any(k in n for k in ('shoulder', 'elbow', 'wrist'))]]
            start = tracking_steps
            one = start + round(1 / env.step_dt)
            late = x[-(round(3 / env.step_dt) + 1):]
            speed = measured_velocity(late)
            dq = late[..., NQ + 6:]
            arms = [i for i, n in enumerate(names) if any(k in n for k in ('shoulder', 'elbow', 'wrist'))]
            yaw = heading_yaw(x[..., 3:7])
            def angle_delta(z, base):
                return torch.atan2(torch.sin(z - base), torch.cos(z - base))
            disp = (late[-1, :, :2] - late[0, :, :2]).norm(dim=-1)
            turn = angle_delta(yaw[-1], yaw[-len(late)]).abs()
            transition_pos = (x[one:, :, :2] - x[one:one+1, :, :2]).norm(dim=-1)
            transition_yaw = angle_delta(yaw[one:], yaw[one:one+1]).abs()
            q = x[..., 7:NQ]
            low, high = limits[..., 0], limits[..., 1]
            violation = ((q < low[None]) | (q > high[None])).any(dim=(0, 2))
            # Offline forward kinematics/contact only: no writes to the live simulator.
            model = env.sim.mj_model
            if model.nq != NQ or model.nv != x.shape[-1] - NQ:
                raise ValueError('CPU contact model does not match collected robot state')
            if not torch.isfinite(x).all():
                raise ValueError('Collected physical state contains non-finite values')
            data = mujoco.MjData(model)
            pattern = re.compile(PROFILE['foot_geom_regex'])
            foot = {'left': set(), 'right': set()}
            for g in range(model.ngeom):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
                if not pattern.fullmatch(name.rsplit('/', 1)[-1]):
                    continue
                body = int(model.geom_bodyid[g]); labels = [name]
                while body:
                    labels.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or '')
                    body = int(model.body_parentid[body])
                sides = {side for side in foot if any(re.search(r'(^|[^a-z])' + side + r'([^a-z]|$)', v.lower()) for v in labels)}
                if len(sides) != 1:
                    raise ValueError(f'Unknown or ambiguous foot geometry mapping: {name}')
                foot[sides.pop()].add(g)
            ground = {g for g in range(model.ngeom) if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE)
                      and (model.geom_contype[g] or model.geom_conaffinity[g])}
            if not ground or any(not values for values in foot.values()):
                raise ValueError('Cannot identify both feet and ground geometries')
            counts = np.zeros((len(late), a.num_envs, 2), dtype=bool)
            cpu_late = late.cpu().numpy()
            for ti, frame in enumerate(cpu_late):
                for ei, state in enumerate(frame):
                    data.qpos[:] = state[:NQ]; data.qvel[:] = state[NQ:]
                    mujoco.mj_forward(model, data)
                    for contact in data.contact[:data.ncon]:
                        if contact.dist > 0:
                            continue
                        pair = {int(contact.geom1), int(contact.geom2)}
                        if pair & ground:
                            for si, side in enumerate(('left', 'right')):
                                counts[ti, ei, si] |= bool(pair & foot[side])
            contact_fraction = counts.mean(axis=0)
            report['contact_geometry'] = dict(foot_geom_ids={k: sorted(v) for k, v in foot.items()},
                ground_geom_ids=sorted(ground), per_env_fraction_left_right=contact_fraction.tolist(),
                final_contact_left_right=counts[-1].tolist(), scope='geometric contact only; no contact forces')
            final_speed = measured_velocity(x[-1])
            final_dq = x[-1, :, NQ+6:]
            metrics = dict(planar_mean=speed[..., :2].norm(dim=-1).mean(0).tolist(),
                yaw_abs_mean=speed[..., 2].abs().mean(0).tolist(),
                arm_velocity_rms=dq[..., arms].square().mean((0, 2)).sqrt().tolist(),
                arm_pose_rmse=(late[..., 7:NQ][..., arms] - ref_arm).square().mean((0, 2)).sqrt().tolist(),
                joint_velocity_rms=dq.square().mean((0, 2)).sqrt().tolist(),
                late_net_displacement=disp.tolist(), late_heading_change=turn.tolist(),
                transition_max_displacement=transition_pos.max(0).values.tolist(),
                transition_max_heading_change=transition_yaw.max(0).values.tolist(),
                soft_limit_violation=violation.tolist(), final_planar_speed=final_speed[..., :2].norm(dim=-1).tolist(),
                final_abs_yaw=final_speed[..., 2].abs().tolist(),
                final_arm_velocity_rms=final_dq[..., arms].square().mean(-1).sqrt().tolist(),
                final_joint_velocity_rms=final_dq.square().mean(-1).sqrt().tolist())
            report['metrics'] = metrics
            checks = [(metrics['planar_mean'], .03), (metrics['yaw_abs_mean'], .05),
                (metrics['arm_velocity_rms'], .12), (metrics['joint_velocity_rms'], .3), (metrics['arm_pose_rmse'], .2),
                (metrics['final_planar_speed'], .03), (metrics['final_abs_yaw'], .05),
                (metrics['final_arm_velocity_rms'], .12), (metrics['final_joint_velocity_rms'], .3),
                (metrics['late_net_displacement'], .03), (metrics['late_heading_change'], .08),
                (metrics['transition_max_displacement'], .05), (metrics['transition_max_heading_change'], .1)]
            passed = all(all(v <= limit for v in values) for values, limit in checks) and not any(metrics['soft_limit_violation']) and bool((contact_fraction >= .95).all()) and bool(counts[-1].all())
            report['status'] = 'validated_candidate' if passed else 'rejected_metrics'
            if passed:
                fk = G1Kinematics(load_g1_model(), device='cuda')
                indices = fk.body_indices(bodies)
                state = x[:, 0]
                out = fk.forward(state[:, :NQ], state[:, NQ:])
                target = a.out / 'candidate.npz'
                np.savez_compressed(target, fps=np.array(fps), joint_names=np.asarray(names), body_names=np.asarray(bodies),
                    joint_pos=state[:, 7:NQ].cpu().numpy(), joint_vel=state[:, NQ+6:].cpu().numpy(),
                    body_pos_w=out['pos'][:, indices].cpu().numpy(), body_quat_w=out['quat'][:, indices].cpu().numpy(),
                    body_lin_vel_w=out['lin_vel'][:, indices].cpu().numpy(), body_ang_vel_w=out['ang_vel'][:, indices].cpu().numpy())
                report['candidate_sha256'] = digest(target)
                (a.out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    except BaseException as error:
        report.update(status='error', error=repr(error))
        if states:
            torch.save(dict(states=torch.stack(states), actions=torch.stack(actions) if actions else None,
                            phase=phases, commands=torch.stack(commands) if commands else None,
                            failures=failures), a.out / 'raw.pt')
            report['raw_sha256'] = digest(a.out / 'raw.pt')
        raise
    finally:
        (a.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        env.close()


if __name__ == '__main__':
    main()
