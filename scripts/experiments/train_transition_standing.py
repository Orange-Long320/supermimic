"""Train a standing expert on continuous command transitions; physical acceptance is separate."""

from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'dataset', 'reference', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--standing-pose', type=Path)
    p.add_argument('--num-envs', type=int, default=512)
    p.add_argument('--iterations', type=int, default=120)
    p.add_argument('--hours', type=float, default=.2)
    p.add_argument('--lr', type=float, default=3e-6)
    p.add_argument('--sigma', type=float, default=.04)
    p.add_argument('--probe-only', action='store_true', help='Record first-rollout likelihood consistency without optimizing')
    a = p.parse_args()
    expected = '/home/2025-Long.xh/HHD/venvs/supermimic/bin/python'
    if Path(sys.prefix).resolve() != Path('/home/2025-Long.xh/HHD/venvs/supermimic').resolve() or os.path.realpath(sys.executable) != os.path.realpath(expected):
        p.error(f'Use the dedicated server interpreter: {expected}')
    if not 80 <= a.num_envs <= 1024 or not 1 <= a.iterations <= 1200 or not 0 < a.hours <= 1 or min(a.lr, a.sigma) <= 0:
        p.error('Use 80–1024 environments, 1–1200 iterations, at most one hour, and positive training parameters')
    if a.out.exists():
        p.error('Output directory already exists')

    import numpy as np
    import torch
    from torch import nn
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.locomotion.config import VelocityCfg
    from supermimic.locomotion.task import measured_velocity
    from supermimic.kinematics.rotations import heading_yaw, projected_gravity
    from supermimic.policy.standing import StandingContext
    from supermimic.robots.dimensions import PROFILE, NQ

    torch.set_num_threads(2)
    torch.manual_seed(29)
    policy, saved = load_policy_from_checkpoint(a.checkpoint, 'cuda:0')
    if not hasattr(policy, 'standing_skill') or policy.standing_skill.mode != 'expert' or policy.standing_skill.feedback_dim != 6:
        raise ValueError('Checkpoint must contain the six-input standing expert')
    source_checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    source_hash = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    if source_checkpoint.get('standing_reference', {}).get('sha256') != hashlib.sha256(a.reference.read_bytes()).hexdigest():
        raise ValueError('Standing reference does not match checkpoint')
    if hasattr(policy, 'velocity_feedback_weight'):
        delattr(policy, 'velocity_feedback_weight')
    policy.requires_grad_(False)
    actor = policy.standing_skill
    actor.requires_grad_(True)
    prior_actor = copy.deepcopy(actor).requires_grad_(False)
    params = list(actor.parameters())
    frozen = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items() if not k.startswith('standing_skill.')}
    cfg = SuperMimicEnvCfgArgs(
        npz_files=[str(a.dataset / 'walk.npz')], num_envs=a.num_envs, seed=29,
        future_frames=saved['env']['future_frames'], episode_length_s=66.,
        action_noise_std=0., obs_noise=False, reset_noise=False,
        domain_randomization=False, record_transitions=False,
        anchor_obs=saved['env']['anchor_obs'], velocity=VelocityCfg(**saved['velocity']))
    env = make_env(cfg, device='cuda:0')
    env.cfg.auto_reset = False
    cmd = env.command_manager.get_term('motion')
    cmd.is_velocity[:] = True
    cmd.force_standing_reset = True
    cmd.set_manual_command((0., 0., 0.))
    observations, _ = env.reset()
    x = observations['policy']
    goal = cmd.velocity_command.clone()
    context = StandingContext(cmd, env)
    reference = np.load(a.reference)
    if len(reference['joint_pos']) < 500 or np.max(np.abs(reference['joint_vel'])) >= 1e-6:
        raise ValueError('Standing reference must be static and at least 500 frames')
    qref = torch.as_tensor(reference['joint_pos'][0], device='cuda', dtype=x.dtype)
    height = float(reference['body_pos_w'][0, 0, 2])
    arm_ids = [i for i, name in enumerate(PROFILE['joint_names']) if any(s in name for s in ('shoulder', 'elbow', 'wrist'))]
    arm_target = qref[arm_ids]
    if a.standing_pose:
        pose = json.loads(a.standing_pose.read_text())['joint_positions_rad']
        arm_target = torch.tensor([pose[PROFILE['joint_names'][i]] for i in arm_ids], device='cuda', dtype=x.dtype)
    value = nn.Sequential(nn.Linear(x.shape[-1] + 12, 256), nn.Tanh(), nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1)).cuda()
    opt = torch.optim.Adam(params, lr=a.lr)
    vopt = torch.optim.Adam(value.parameters(), lr=3e-4)
    ids = torch.arange(a.num_envs, device='cuda')
    # Four interleaved groups: two continuous, one direct-switch, one standing.
    continuous = ids % 4 < 2
    direct = ids % 4 == 2
    hold = ids % 4 == 3
    choices = torch.tensor([[.3, 0., 0.], [-.2, 0., 0.], [0., 0., .4], [0., 0., -.4], [.6, 0., 0.]], device='cuda')
    # Both transition groups cover every first command, then cycle through all five.
    offset = (ids // 4) % 5
    rates = torch.tensor([1., 1., 1.5], device='cuda') * env.step_dt
    stop_age = torch.zeros(a.num_envs, device='cuda')
    ever_moved = torch.zeros(a.num_envs, device='cuda', dtype=torch.bool)
    gamma, lam = .995, .97

    def value_features(obs):
        state = cmd.state_io.read()
        yaw = heading_yaw(state[:, 3:7])
        dh = torch.atan2(torch.sin(yaw - context.anchor_yaw), torch.cos(yaw - context.anchor_yaw))
        return torch.cat((obs * actor.obs_scale, 10 * measured_velocity(state), state[:, 2:3],
                          (env.episode_length_buf * env.step_dt / 66)[:, None],
                          state[:, :2] - context.anchor_xy, dh[:, None], goal, (stop_age / 66)[:, None]), -1)

    def command_for_age(age):
        desired = torch.zeros_like(goal)
        for j, start in enumerate((3., 15., 27., 39., 51.)):
            active = continuous & (age >= start) & (age < start + 7.)
            desired[active] = choices[(offset[active] + j) % 5]
        for j, start in enumerate((3., 10., 17., 24., 31.)):
            active = direct & (age >= start) & (age < start + 7.)
            desired[active] = choices[(offset[active] + j) % 5]
        return desired

    def verify_core(obs):
        current = policy.state_dict()
        for key, tensor in frozen.items():
            assert torch.equal(current[key].detach().cpu(), tensor), key
        with torch.no_grad():
            for command in choices:
                c = command.expand(len(obs), -1)
                original = policy.decoder(policy.encoders['velocity'](obs, c), obs)
                actual = policy.velocity_action(obs, c, torch.zeros(len(obs), 6, device=obs.device))
                assert torch.equal(original, actual), 'moving expert changed'

    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'launch.json').write_text(json.dumps({
        'argv': sys.argv, 'interpreter': sys.executable, 'checkpoint_sha256': hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        'reference_sha256': hashlib.sha256(a.reference.read_bytes()).hexdigest(),
        'standing_pose_sha256': hashlib.sha256(a.standing_pose.read_bytes()).hexdigest() if a.standing_pose else None,
        'walk_sha256': hashlib.sha256((a.dataset / 'walk.npz').read_bytes()).hexdigest(),
        'scope': 'unvalidated transition training; physical evaluation required',
        'num_envs': a.num_envs, 'episode_seconds': 66, 'sigma': a.sigma, 'gamma': gamma, 'lambda': lam,
        'schedule': '50% continuous five commands, 25% direct switches, 25% standing',
    }, indent=2) + '\n')
    start_time = time.monotonic()
    total_falls = total_timeouts = 0
    last_saved = 0
    trace = []

    def save(iteration):
        nonlocal last_saved
        verify_core(x)
        ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
        ck['policy'] = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
        ck['standing_skill'] = dict(ck['standing_skill'], moving_feedback=False)
        ck.pop('standing_ppo_state', None)
        ck.pop('velocity_training', None)
        ck.pop('physical_stop', None)
        ck.pop('control_feedback', None)
        ck.pop('standing_feedback', None)
        ck['transition_standing'] = {
            'iterations': iteration, 'source_sha256': hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
            'reference_sha256': hashlib.sha256(a.reference.read_bytes()).hexdigest(),
            'standing_pose_sha256': hashlib.sha256(a.standing_pose.read_bytes()).hexdigest() if a.standing_pose else None,
            'movement_feedback': False, 'trainable': 'standing expert encoder, decoder, feedback',
            'schedule': '50% continuous five commands, 25% direct switches, 25% standing',
            'episode_seconds': 66, 'sigma': a.sigma, 'gamma': gamma, 'gae_lambda': lam,
            'position_cost': 200., 'heading_cost': 40., 'cold_action_prior_weight': .1,
            'validation': 'pending physical acceptance',
            'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
            'source_commit': os.environ.get('SM_SOURCE_COMMIT', 'unspecified'),
            'entry_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'trace': trace,
        }
        ck['transition_ppo_state'] = dict(actor_optimizer=opt.state_dict(), value=value.state_dict(),
                                         value_optimizer=vopt.state_dict(), sigma=a.sigma)
        path = a.out / f'candidate_{iteration}.pt'
        torch.save(ck, path)
        path.with_suffix('.ready').write_text(str(iteration))
        last_saved = iteration

    try:
        for iteration in range(1, a.iterations + 1):
            batch = []
            with torch.no_grad():
                for _ in range(32):
                    age = env.episode_length_buf * env.step_dt
                    desired = command_for_age(age)
                    goal = goal + (desired - goal).clamp(-rates, rates)
                    cmd.manual_command = goal
                    cmd._refresh_velocity()
                    zero = goal.square().sum(-1) < 1e-6
                    stop_age = torch.where(zero, stop_age + env.step_dt, torch.zeros_like(stop_age))
                    feedback = context.features(goal)
                    anchor_xy, anchor_yaw = context.anchor_xy.clone(), context.anchor_yaw.clone()
                    gate = actor.gate(goal)
                    eligible = gate > 0
                    ever_moved |= goal.square().sum(-1) >= actor.radius ** 2
                    group = torch.where(~zero, 2, torch.where(ever_moved, 1, 0))
                    stand_mean = actor(x, feedback)
                    base = policy.decoder(policy.encoders['velocity'](x, goal), x)
                    sampled = stand_mean.clone()
                    sampled[eligible] = torch.distributions.Normal(stand_mean[eligible], a.sigma).sample()
                    logp = torch.zeros(a.num_envs, device='cuda')
                    logp[eligible] = torch.distributions.Normal(stand_mean[eligible], a.sigma).log_prob(sampled[eligible]).sum(-1)
                    action = (1 - gate[:, None]) * base + gate[:, None] * sampled
                    vf = value_features(x)
                    v = value(vf).squeeze(-1)
                    old_x, old_goal = x.clone(), goal.clone()
                    nxt, _, term, tout, _ = env.step(action)
                    term, tout = term.clone(), tout.clone()
                    done = term | tout
                    state = cmd.state_io.read()
                    speed, q, dq = measured_velocity(state), state[:, 7:NQ], state[:, NQ+6:]
                    yaw = heading_yaw(state[:, 3:7])
                    yaw_error = torch.atan2(torch.sin(yaw - anchor_yaw), torch.cos(yaw - anchor_yaw))
                    reward = 2 - 30 * (speed[:, :2] - old_goal[:, :2]).square().sum(-1) - 40 * (speed[:, 2] - old_goal[:, 2]).square()
                    reward -= (200 * (state[:, :2] - anchor_xy).square().sum(-1) + 40 * yaw_error.square()
                               + 2 * projected_gravity(state[:, 3:7])[:, :2].square().sum(-1) + 10 * (state[:, 2] - height).square()) * zero
                    entry = stop_age < .8
                    reward -= (torch.where(entry, .1, 10.) * dq[:, arm_ids].square().mean(-1)
                               + torch.where(entry, 10., .5) * (q[:, arm_ids] - arm_target).square().mean(-1)
                               + 2 * (q - qref).square().mean(-1) + torch.where(entry, .01, .1) * dq.square().mean(-1)) * zero
                    reward -= 20 * term.float()
                    timeout_only = tout & ~term
                    if timeout_only.any():
                        reward += gamma * value(value_features(nxt['policy'])).squeeze(-1) * timeout_only
                    batch.append((old_x, feedback.clone(), sampled, logp, v, reward, done, vf, eligible, group.clone(), hold.clone(), stand_mean.clone(), term, tout))
                    total_falls += int(term.sum())
                    total_timeouts += int(tout.sum())
                    if done.any():
                        reset_ids = done.nonzero().squeeze(-1)
                        goal[reset_ids] = 0
                        stop_age[reset_ids] = 0
                        ever_moved[reset_ids] = False
                        nxt, _ = env.reset(env_ids=reset_ids)
                        context.reset(reset_ids)
                    x = nxt['policy']
                tail = value(value_features(x)).squeeze(-1)
                gae = torch.zeros_like(tail)
                advantages = []
                for b in reversed(batch):
                    delta = b[5] + gamma * tail * (~b[6]) - b[4]
                    gae = delta + gamma * lam * (~b[6]) * gae
                    advantages.append(gae)
                    tail = b[4]
                advantages = torch.stack(advantages[::-1])
                returns = (advantages + torch.stack([b[4] for b in batch])).flatten()
                advantages = advantages.flatten()
                advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
                fields = [torch.cat([b[i] for b in batch]) for i in (0, 1, 2, 3, 7, 8, 9, 10, 11)]
                xx, ff, uu, old_lp, vv, eligible, group, cold_hold, old_mean = fields
                eligible_ids = eligible.nonzero().squeeze(-1)
                holdout = []
                for g in range(3):
                    candidates = (eligible & (group == g)).nonzero().squeeze(-1)
                    if len(candidates):
                        holdout.append((g, candidates[torch.randperm(len(candidates), device='cuda')[:128]]))

            def grouped_kl():
                with torch.no_grad():
                    return {g: float(((actor(xx[ix], ff[ix]) - old_mean[ix]).square().sum(-1) / (2 * a.sigma ** 2)).mean()) for g, ix in holdout}

            if iteration == 1:
                with torch.no_grad():
                    check_ids = eligible_ids[:128]
                    small_mean = actor(xx[check_ids], ff[check_ids])
                    logp = torch.distributions.Normal(small_mean, a.sigma).log_prob(uu[check_ids]).sum(-1)
                    saved_logp = torch.distributions.Normal(old_mean[check_ids], a.sigma).log_prob(uu[check_ids]).sum(-1)
                    same_mean = actor(xx[:a.num_envs], ff[:a.num_envs])
                    same_logp = torch.distributions.Normal(same_mean, a.sigma).log_prob(uu[:a.num_envs]).sum(-1)
                    torch.save(dict(obs=xx[:a.num_envs],feedback=ff[:a.num_envs],sampled=uu[:a.num_envs],
                                    old_logp=old_lp[:a.num_envs],old_mean=old_mean[:a.num_envs]), a.out/'likelihood_probe.pt')
                    probe = dict(saved_mean_logp_max_error=float((saved_logp-old_lp[check_ids]).abs().max()),
                        same_batch_mean_max_error=float((same_mean-old_mean[:a.num_envs]).abs().max()),
                        same_batch_logp_max_error=float((same_logp-old_lp[:a.num_envs]).abs().max()),
                        subset_mean_max_error=float((small_mean-old_mean[check_ids]).abs().max()),
                        subset_logp_max_error=float((logp-old_lp[check_ids]).abs().max()),
                        initial_group_kl=grouped_kl(),
                        cuda_matmul_fp32_precision=torch.backends.cuda.matmul.fp32_precision)
                    (a.out / 'likelihood_probe.json').write_text(json.dumps(probe, indent=2) + '\n')
                    print(json.dumps(dict(likelihood_probe=probe)), flush=True)
                    if a.probe_only:
                        return
                    assert torch.allclose(logp, old_lp[check_ids], atol=1e-4, rtol=1e-5), 'PPO branch labels changed before update'
                    assert max(grouped_kl().values(), default=0.) < 1e-8
                    verify_core(xx[:128])
                print(json.dumps(dict(branch_likelihood_check='passed', original_moving_action_check='passed',
                                      source_sha256=source_hash)), flush=True)

            accepted = 0
            for _ in range(4):
                for ix in eligible_ids[torch.randperm(len(eligible_ids), device='cuda')].split(2048):
                    if not len(ix):
                        continue
                    mean = actor(xx[ix], ff[ix])
                    lp = torch.distributions.Normal(mean, a.sigma).log_prob(uu[ix]).sum(-1)
                    ratio = (lp - old_lp[ix]).exp()
                    loss = -torch.minimum(ratio * advantages[ix], ratio.clamp(.8, 1.2) * advantages[ix]).mean()
                    protected = cold_hold[ix]
                    if protected.any():
                        with torch.no_grad():
                            prior = prior_actor(xx[ix][protected], ff[ix][protected])
                        loss = loss + .1 * (mean[protected] - prior).square().mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, .5)
                    backup = [v.detach().clone() for v in params]
                    optimizer_backup = copy.deepcopy(opt.state_dict())
                    opt.step()
                    kls = grouped_kl()
                    if any(v > .03 for v in kls.values()):
                        with torch.no_grad():
                            for param, old_param in zip(params, backup):
                                param.copy_(old_param)
                        opt.load_state_dict(optimizer_backup)
                    else:
                        accepted += 1
                for ix in torch.randperm(len(vv), device='cuda').split(2048):
                    vl = (value(vv[ix]).squeeze(-1) - returns[ix]).square().mean()
                    vopt.zero_grad(set_to_none=True)
                    vl.backward()
                    torch.nn.utils.clip_grad_norm_(value.parameters(), 1.)
                    vopt.step()
            if iteration % 5 == 0 or iteration == 1:
                row = {'iteration': iteration, 'seconds': time.monotonic() - start_time,
                                  'eligible': int(eligible.sum()), 'group_counts': {str(g): int((eligible & (group == g)).sum()) for g in range(3)},
                                  'group_kl': grouped_kl(), 'accepted_actor_steps': accepted,
                                  'falls': total_falls, 'timeouts': total_timeouts,
                                  'reward': float(torch.stack([b[5] for b in batch]).mean()),
                                  'cuda_peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2}
                trace.append(row)
                print(json.dumps(row), flush=True)
            if iteration % 20 == 0:
                save(iteration)
            if time.monotonic() - start_time >= a.hours * 3600:
                break
        if iteration != last_saved:
            save(iteration)
        (a.out / 'report.json').write_text(json.dumps(dict(status='completed_candidate', iterations=iteration,
            seconds=time.monotonic()-start_time, trace=trace), indent=2) + '\n')
    finally:
        env.close()


if __name__ == '__main__':
    main()
