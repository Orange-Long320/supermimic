"""Fit a real-transition world model and train only the standing expert in imagination.

The model gate is diagnostic and never constitutes physical policy acceptance.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sampleable_start_mask(replay, history):
    """Use the same ring-history protection for readiness and actual sampling."""
    import torch
    safe = replay.valid & ~replay.terminal
    tgrid = torch.arange(replay.T, device=replay.device)[None, :]
    return safe & ((replay.ptr[:, None]-tgrid-1) % replay.T < replay.T-history-1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint', 'dataset', 'reference', 'standing-pose', 'standing-retention', 'control-retention', 'control-dynamics', 'control-report', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=256)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--wm-batch', type=int, default=128)
    p.add_argument('--wm-batches', type=int, default=20)
    p.add_argument('--iterations', type=int, default=120)
    p.add_argument('--hours', type=float, default=.25)
    p.add_argument('--lr', type=float, default=None, help='Override the native checkpoint policy learning rate')
    p.add_argument('--train-base', action='store_true', help='Explicitly train the full standing expert as a control')
    p.add_argument('--preserve-moving-feedback', action='store_true', help='Preserve a validated-provenance BPTT movement projection')
    p.add_argument('--ema-decay', type=float, default=.99)
    a = p.parse_args()
    expected = Path('/home/2025-Long.xh/HHD/venvs/supermimic').resolve()
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if Path(sys.prefix).resolve() != expected or not gpu or len(gpu.split(',')) != 1:
        p.error('Use the dedicated server environment and explicitly select one GPU')
    if not 8 <= a.num_envs <= 1024 or min(a.batch, a.wm_batch, a.wm_batches, a.iterations) < 1 or a.iterations > 1200 or not 0 < a.hours <= 1 or (a.lr is not None and a.lr <= 0):
        p.error('Invalid training bounds')
    if a.out.exists():
        p.error('Output directory already exists')
    if a.batch < 8 or a.wm_batches < 2:
        p.error('Use at least eight actor samples and two world-model batches for the strata')
    if not 0 <= a.ema_decay < 1:
        p.error('EMA decay must be in [0, 1)')

    import numpy as np
    import torch
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    from supermimic.kinematics.rotations import heading_yaw
    from supermimic.locomotion.config import VelocityCfg
    from supermimic.locomotion.task import measured_velocity
    from supermimic.policy.standing import StandingContext
    from supermimic.robots.dimensions import PROFILE
    from supermimic.world_model.state_spec import StateSpecCfg
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg
    from supermimic_stop_imagination import StopImaginationTrainer
    from supermimic_stop_ema import StandingEMA
    from fixed_control_dynamics import FixedControlDynamics

    torch.set_num_threads(2)
    torch.manual_seed(29)
    policy, saved = load_policy_from_checkpoint(a.checkpoint, 'cuda:0')
    source = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if a.lr is None:
        a.lr = float(saved['policy_trainer']['lr'])
        lr_source = 'checkpoint.config.policy_trainer.lr'
    else:
        lr_source = 'explicit_override'
    if not np.isfinite(a.lr) or a.lr <= 0:
        raise ValueError('Invalid policy learning rate')
    if not hasattr(policy, 'standing_skill') or policy.standing_skill.mode != 'expert' or policy.standing_skill.feedback_dim != 6:
        raise ValueError('Six-input standing expert required')
    if source.get('standing_reference', {}).get('sha256') != sha(a.reference):
        raise ValueError('Reference mismatch')
    retention = torch.load(a.standing_retention, map_location='cpu', weights_only=False)
    for key, expected_hash in (('source_sha256', sha(a.checkpoint)), ('reference_sha256', sha(a.reference)),
                               ('standing_pose_sha256', sha(a.standing_pose))):
        if retention.get(key) != expected_hash:
            raise ValueError(f'Standing retention {key} does not match this initialization')
    if any(key not in retention for key in ('obs', 'feedback', 'source_action', 'age')):
        raise ValueError('Standing retention is missing observations, feedback, source actions or ages')
    control_retention = torch.load(a.control_retention, map_location='cpu', weights_only=False)
    for key, expected_hash in (('source_sha256', sha(a.checkpoint)), ('reference_sha256', sha(a.reference)),
                               ('standing_pose_sha256', sha(a.standing_pose))):
        if control_retention.get(key) != expected_hash:
            raise ValueError(f'Control retention {key} does not match this initialization')
    if any(key not in control_retention for key in ('obs', 'command', 'feedback', 'source_action')):
        raise ValueError('Control retention is missing aligned policy inputs and source actions')
    moving_projection_sha = None
    if a.preserve_moving_feedback:
        if not hasattr(policy, 'velocity_feedback_weight') or source.get('standing_skill', {}).get('moving_feedback') is not True:
            raise ValueError('Preserving movement requires a checkpoint with enabled velocity feedback')
        records = [(key, source.get(key)) for key in ('supermimic_velocity_feedback', 'supermimic_composed_feedback')
                   if isinstance(source.get(key), dict)]
        if len(records) != 1:
            raise ValueError('Expected exactly one BPTT movement or composed record')
        kind, meta = records[0]
        if (meta.get('algorithm') != 'supermimic_bptt' or not isinstance(meta.get('actor_updates'), int)
                or meta['actor_updates'] <= 0 or meta.get('reference_sha256') != sha(a.reference)):
            raise ValueError('Invalid movement BPTT provenance or reference')
        if meta.get('standing_pose_sha256') != sha(a.standing_pose):
            raise ValueError('Movement BPTT standing-pose provenance mismatch')
        if kind == 'supermimic_composed_feedback':
            components = source.get('composed_training')
            if not isinstance(components, dict) or set(components) != {'stop', 'moving'}:
                raise ValueError('Composed source lacks component training records')
            common = meta.get('source_common_sha')
            stop, moving = components['stop'], components['moving']
            evidence = meta.get('stop_pose_evidence', {})
            components_info = meta.get('components', {})
            if (not common or any(not isinstance(v, dict) or v.get('algorithm') != 'supermimic_bptt'
                    or v.get('source_sha256') != common or v.get('reference_sha256') != sha(a.reference)
                    or not isinstance(v.get('actor_updates'), int) or v['actor_updates'] <= 0 for v in (stop, moving))
                    or meta['actor_updates'] != stop['actor_updates'] + moving['actor_updates']
                    or moving.get('standing_pose_sha256') != sha(a.standing_pose)
                    or (stop.get('standing_pose_sha256') and stop['standing_pose_sha256'] != sha(a.standing_pose))
                    or not isinstance(evidence, dict) or evidence.get('standing_pose_sha256') != sha(a.standing_pose)
                    or not isinstance(components_info, dict) or not isinstance(components_info.get('base'), dict)
                    or components_info['base'].get('sha256') != common):
                raise ValueError('Invalid composed component provenance')
        elif not meta.get('source_sha256'):
            raise ValueError('Movement BPTT source hash missing')
        moving_projection_sha = hashlib.sha256(policy.velocity_feedback_weight.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    elif hasattr(policy, 'velocity_feedback_weight'):
        delattr(policy, 'velocity_feedback_weight')
    policy.requires_grad_(False)
    initial_policy = copy.deepcopy(policy).requires_grad_(False)

    wc = copy.deepcopy(saved['world_model'])
    wc['state_spec'] = StateSpecCfg(**wc['state_spec'])
    wm = WorldModel(WorldModelCfg(**wc), None, 'cuda:0')
    missing, extra = wm.load_state_dict(source['wm'], strict=False)
    if missing or any(not k.startswith('heads.') for k in extra):
        raise ValueError(f'Unexpected world model keys: {missing}, {extra}')
    wm_opt = torch.optim.Adam(wm.dynamics.parameters(), lr=saved['wm_trainer']['lr'])
    cfg = SuperMimicEnvCfgArgs(npz_files=[str(a.dataset / 'walk.npz')], num_envs=a.num_envs,
        seed=29, future_frames=saved['env']['future_frames'], episode_length_s=66.,
        action_noise_std=0., obs_noise=False, reset_noise=False, domain_randomization=False,
        record_transitions=True, replay_capacity=256, anchor_obs=saved['env']['anchor_obs'],
        velocity=VelocityCfg(**saved['velocity']))
    env = make_env(cfg, device='cuda:0')
    # make_env configures the backend; force IEEE precision after that configuration.
    torch.backends.cuda.matmul.fp32_precision = 'ieee'
    torch.backends.cudnn.fp32_precision = 'ieee'
    env.cfg.auto_reset = False
    cmd = env.command_manager.get_term('motion')
    cmd.is_velocity[:] = True
    cmd.force_standing_reset = True
    cmd.set_manual_command((0., 0., 0.))
    obs, _ = env.reset()
    context = StandingContext(cmd, env)
    replay = env.recorder_manager.get_term('replay').buffer
    ref = np.load(a.reference)
    if len(ref['joint_pos']) < 500 or np.max(np.abs(ref['joint_vel'])) >= 1e-6:
        raise ValueError('Reference must be static')
    qref = torch.as_tensor(ref['joint_pos'][0], device='cuda', dtype=obs['policy'].dtype)
    arm_ids = [i for i, n in enumerate(PROFILE['joint_names']) if any(k in n for k in ('shoulder', 'elbow', 'wrist'))]
    pose = json.loads(a.standing_pose.read_text())['joint_positions_rad']
    arm_target = torch.tensor([pose[PROFILE['joint_names'][i]] for i in arm_ids], device='cuda', dtype=qref.dtype)
    trainer = StopImaginationTrainer(policy, wm, cmd.library.default_joint_pos, qref, arm_ids, arm_target,
        float(ref['body_pos_w'][0, 0, 2]), VelocityCfg(**saved['velocity']), saved['policy']['history'], env.step_dt,
        lr=a.lr, horizon=32, train_base=a.train_base)
    trainer.set_cold_retention(retention['obs'].to('cuda:0'), retention['feedback'].to('cuda:0'),
                               retention['source_action'].to('cuda:0'))
    trainer.set_control_retention(*(control_retention[k].to('cuda:0')
                                   for k in ('obs', 'command', 'feedback', 'source_action')))
    fixed_dynamics = FixedControlDynamics(a.control_dynamics, a.control_retention, a.control_report,
        source_sha256=sha(a.checkpoint), reference_sha256=sha(a.reference), standing_pose_sha256=sha(a.standing_pose),
        report_sha256=sha(a.control_report), dt=env.step_dt, device='cuda:0')
    if fixed_dynamics.provenance['preserve_moving_feedback'] != a.preserve_moving_feedback:
        raise ValueError('Fixed dynamics collection used a different movement-feedback mode')
    trainable_keys = {'standing_skill.' + name for name in trainer.actor_parameter_names}
    actual_trainable = {name for name, param in policy.named_parameters() if param.requires_grad}
    if trainable_keys != actual_trainable:
        raise AssertionError('Unexpected trainable policy parameters')
    frozen = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items() if k not in trainable_keys}
    ema = StandingEMA(policy.standing_skill, a.ema_decay)
    E, T = replay.E, replay.T
    meta_xy = torch.zeros(E, T, 2, device='cuda')
    meta_yaw = torch.zeros(E, T, device='cuda')
    meta_age = torch.zeros(E, T, device='cuda')
    meta_cold = torch.zeros(E, T, device='cuda', dtype=torch.bool)
    meta_stop = torch.zeros(E, T, device='cuda', dtype=torch.bool)
    ids = torch.arange(E, device='cuda')
    train_env = ids % 4 != 3
    # Both disjoint train and held-out environments include permanent standing.
    cold = (ids // 4) % 4 == 0
    choices = torch.tensor([[.3, 0., 0.], [-.2, 0., 0.], [0., 0., .4], [0., 0., -.4], [.6, 0., 0.]], device='cuda')
    offset = (ids // 2) % 5
    rates = torch.tensor([1., 1., 1.5], device='cuda') * env.step_dt
    goal = cmd.velocity_command.clone()
    stop_age = torch.zeros(E, device='cuda')
    ever_moved = torch.zeros(E, device='cuda', dtype=torch.bool)
    history = saved['policy']['history']
    trace = []
    total_falls = total_timeouts = actor_updates = 0
    started = time.monotonic()
    runtime_config = dict(algorithm='supermimic_bptt', imagination_horizon=32, wm_window=32,
        wm_batches_per_update=a.wm_batches, wm_batch=a.wm_batch, actor_batch=a.batch,
        history=history, dt=env.step_dt, gamma=trainer.cfg.gamma, gae_lambda=trainer.cfg.gae_lambda,
        collection_steps_per_update=32, episode_seconds=66., num_envs=E, replay_capacity=T,
        obs_noise=False, reset_noise=False, domain_randomization=False, stop_collection_noise=.03,
        movement_collection_noise=0., actual_stop_and_cold_stratified=True,
        moving_core_frozen=True, moving_feedback=a.preserve_moving_feedback,
        preserve_moving_feedback=a.preserve_moving_feedback, moving_projection_sha256=moving_projection_sha,
        moving_source_sha256=sha(a.checkpoint), optimizer_initialization='fresh',
        control_matmul_precision='ieee', actor_update='deterministic BPTT through frozen world model',
        model_gate_velocity_8=[.10,.10,.15], model_gate_velocity_32=[.20,.20,.30],
        model_gate_position_heading_32=[.03,.08], actor_lr=a.lr, actor_lr_source=lr_source,
        train_base=a.train_base, trainable_policy_parameters=sorted(trainable_keys),
        standing_retention_sha256=sha(a.standing_retention), standing_retention_samples=len(retention['obs']),
        control_retention_sha256=sha(a.control_retention), control_retention_samples=len(control_retention['obs']),
        control_dynamics=fixed_dynamics.provenance, wm_fixed_batches_per_update=(a.wm_batches+1)//2,
        wm_recent_batches_per_update=a.wm_batches//2, actor_fixed_source_fraction=.5,
        standing_retention_source_action_max_abs_diff=trainer.cold_retention['source_action_max_abs_diff'])
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'launch.json').write_text(json.dumps(dict(argv=sys.argv, gpu=gpu, checkpoint_sha256=sha(a.checkpoint),
        reference_sha256=sha(a.reference), pose_sha256=sha(a.standing_pose), walk_sha256=sha(a.dataset / 'walk.npz'),
        entry_sha256=sha(__file__), standing_retention_sha256=sha(a.standing_retention),
        control_retention_sha256=sha(a.control_retention),
        control_dynamics=fixed_dynamics.provenance,
        fixed_dynamics_helper_sha256=sha(Path(__file__).with_name('fixed_control_dynamics.py')),
        actor_lr=a.lr, actor_lr_source=lr_source, train_base=a.train_base,
        preserve_moving_feedback=a.preserve_moving_feedback, moving_projection_sha256=moving_projection_sha,
        trainable_policy_parameters=sorted(trainable_keys), ema=ema.metadata(),
        ema_helper_sha256=sha(Path(__file__).with_name('supermimic_stop_ema.py')),
        model_gate='diagnostic only; physical policy acceptance required'), indent=2) + '\n')
    (a.out/'runtime_config.json').write_text(json.dumps(runtime_config, indent=2)+'\n')

    def verify_core(x):
        for key, old in frozen.items():
            if not torch.equal(policy.state_dict()[key].detach().cpu(), old):
                raise AssertionError(f'Frozen policy tensor changed: {key}')
        with torch.no_grad():
            feedback = torch.zeros(len(x), 6, device=x.device, dtype=x.dtype)
            for c in choices:
                c = c.expand(len(x), -1)
                original = initial_policy.velocity_action(x, c, feedback)
                actual = policy.velocity_action(x, c, feedback)
                if not torch.equal(original, actual):
                    raise AssertionError('Moving action changed')

    def sample_windows(mask, n):
        e, t = replay._sample_from_mask(replay.window_mask(32) & mask[:, None], n)
        tt = (t[None] + torch.arange(33, device='cuda')[:, None]) % T
        ee = e[None].expand(33, -1)
        return replay.s[ee, tt], replay.a_exec[ee[:-1], tt[:-1]], replay.xi[e, t]

    def starts(mask, n):
        e, t = replay._sample_from_mask(sampleable_start_mask(replay, history) & mask, n)
        tt = (t[None] + torch.arange(history, device='cuda')[:, None] - (history - 1)) % T
        ee = e[None].expand(history, -1)
        same = (replay.ep_id[ee, tt] == replay.ep_id[e, t][None]) & replay.valid[ee, tt]
        first = same.float().argmax(0)
        tt = torch.where(same, tt, tt[first, torch.arange(n, device='cuda')][None].expand_as(tt))
        prev = (tt - 1) % T
        prev_ok = (replay.ep_id[ee, prev] == replay.ep_id[e, t][None]) & replay.valid[ee, prev] & ~replay.ep_start[ee, tt]
        actions = torch.where(prev_ok[..., None], replay.a[ee, prev], torch.zeros_like(replay.a[ee, prev]))
        return e, t, replay.s[ee, tt], actions

    def actor_batch():
        eligible = train_env[:, None] & sampleable_start_mask(replay, history)
        after_mask = eligible & meta_stop & ~meta_cold
        cold_mask = eligible & meta_stop & meta_cold
        move_mask = eligible & ~meta_stop & ~meta_cold
        nsource = a.batch // 2
        nrecent = a.batch - nsource
        nmove = nrecent // 2 if move_mask.any() else 0
        nafter = (nrecent-nmove)//2 if after_mask.any() else 0
        ncold = nrecent-nmove-nafter
        parts = []
        if nafter:
            parts.append(starts(after_mask, nafter))
        parts.append(starts(cold_mask, ncold))
        if nmove:
            parts.append(starts(move_mask, nmove))
        nstop = nafter+ncold
        e = torch.cat([v[0] for v in parts]); t = torch.cat([v[1] for v in parts])
        hs = torch.cat([v[2] for v in parts], dim=1); ha = torch.cat([v[3] for v in parts], dim=1)
        xy = meta_xy[e, t].clone(); yaw = meta_yaw[e, t].clone(); age = meta_age[e, t].clone()
        xy[nstop:] = hs[-1, nstop:, :2]; yaw[nstop:] = heading_yaw(hs[-1, nstop:, 3:7]); age[nstop:] = 0
        recent = dict(hist_states=hs, hist_actions=ha, xi=replay.xi[e, t], anchor_xy=xy,
                      anchor_yaw=yaw, stop_age=age, is_cold=meta_cold[e, t])
        fixed = fixed_dynamics.sample_stop_starts(nsource, history)
        return {key: torch.cat((recent[key], fixed[key]), dim=1 if key in ('hist_states', 'hist_actions') else 0)
                for key in recent}

    @torch.no_grad()
    def heldout():
        states, actions, xi = sample_windows(~train_env, a.wm_batch)
        pred = states[0]; out = {}; pose_error = {}
        for j in range(32):
            pred = wm.step(pred, actions[j], xi)
            if j + 1 in (1, 8, 32):
                out[str(j + 1)] = (measured_velocity(pred) - measured_velocity(states[j + 1])).square().mean(0).sqrt().tolist()
                delta_yaw = heading_yaw(pred[:, 3:7])-heading_yaw(states[j+1, :, 3:7])
                delta_yaw = torch.atan2(torch.sin(delta_yaw), torch.cos(delta_yaw))
                pose_error[str(j+1)] = [float((pred[:, :2]-states[j+1, :, :2]).square().sum(-1).mean().sqrt()),
                                        float(delta_yaw.square().mean().sqrt())]
        return out, pose_error

    def save(iteration):
        verify_core(obs['policy'][:128])
        ck = copy.deepcopy(source)
        ck['policy'] = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
        ck['standing_skill'] = dict(ck['standing_skill'], moving_feedback=a.preserve_moving_feedback)
        ck['wm'].update({k: v.detach().cpu() for k, v in wm.state_dict().items()})
        for key in ('standing_ppo_state', 'transition_ppo_state', 'physical_stop', 'control_feedback', 'standing_feedback', 'transition_standing',
                    'wm_opt', 'policy_opt', 'critic_opt', 'velocity_training',
                    'supermimic_velocity_feedback', 'supermimic_composed_feedback', 'composed_training'):
            ck.pop(key, None)
        ck['supermimic_stop'] = dict(algorithm='supermimic_bptt', iteration=iteration, source_sha256=sha(a.checkpoint), reference_sha256=sha(a.reference),
            standing_pose_sha256=sha(a.standing_pose), standing_retention_sha256=sha(a.standing_retention),
            control_retention_sha256=sha(a.control_retention),
            entry_sha256=sha(__file__), config={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
            trace=trace, wm_optimizer=wm_opt.state_dict(), actor_state=trainer.save_state(),
            actor_updates=actor_updates, control_matmul_precision='ieee', source_commit=os.environ.get('SM_SOURCE_COMMIT', 'unspecified'),
            runtime_config=runtime_config,
            validation='model gate only; policy physical acceptance pending')
        ck['policy_ema'] = dict(ema.metadata(), variant='online',
                                shadow_state={k: v.detach().cpu().clone() for k, v in ema.model.state_dict().items()})
        path = a.out / f'candidate_{iteration}.pt'
        temporary = path.with_suffix('.pt.tmp')
        torch.save(ck, temporary)
        temporary.replace(path)
        ema_ck = dict(ck)
        ema_ck['policy'] = ema.policy_state(ck['policy'])
        for key, old in frozen.items():
            if not torch.equal(ema_ck['policy'][key], old):
                raise AssertionError(f'EMA changed frozen policy tensor: {key}')
        ema_ck['policy_ema'] = dict(ema.metadata(), variant='ema', paired_online_sha256=sha(path))
        # An EMA export is an evaluation artifact, not an optimizer continuation.
        ema_ck['supermimic_stop'] = {k: v for k, v in ck['supermimic_stop'].items()
                                    if k not in ('wm_optimizer', 'actor_state')}
        ema_dir = a.out / 'ema'
        ema_dir.mkdir(exist_ok=True)
        ema_path = ema_dir / path.name
        temporary = ema_path.with_suffix('.pt.tmp')
        torch.save(ema_ck, temporary)
        temporary.replace(ema_path)
        (ema_dir / path.with_suffix('.ready').name).write_text(str(iteration))
        # Publish the online marker only after the complete pair is available.
        path.with_suffix('.ready').write_text(str(iteration))

    try:
        last_saved = 0
        for iteration in range(1, a.iterations + 1):
            with torch.no_grad():
                for _ in range(32):
                    age = env.episode_length_buf * env.step_dt
                    desired = torch.zeros_like(goal)
                    for j, start in enumerate((3., 15., 27., 39., 51.)):
                        active = ~cold & (age >= start) & (age < start + 7.)
                        desired[active] = choices[(offset[active] + j) % 5]
                    goal = goal + (desired - goal).clamp(-rates, rates)
                    cmd.manual_command = goal
                    cmd._refresh_velocity()
                    zero = goal.square().sum(-1) < 1e-6
                    stop_age = torch.where(zero, stop_age + env.step_dt, torch.zeros_like(stop_age))
                    feedback = context.features(goal)
                    ever_moved |= goal.square().sum(-1) >= policy.standing_skill.radius ** 2
                    gate = policy.standing_skill.gate(goal)
                    if a.preserve_moving_feedback:
                        mean = policy.velocity_action(obs['policy'], goal, feedback)
                        action = mean + gate[:, None] * .03 * torch.randn_like(mean)
                    else:
                        base = policy.decoder(policy.encoders['velocity'](obs['policy'], goal), obs['policy'])
                        stand = policy.standing_skill(obs['policy'], feedback)
                        action = (1 - gate[:, None]) * base + gate[:, None] * (stand + .03 * torch.randn_like(stand))
                    ptr = replay.ptr.clone()
                    before_state = cmd.state_io.read().clone() if iteration == 1 else None
                    meta_xy[ids, ptr] = context.anchor_xy
                    meta_yaw[ids, ptr] = context.anchor_yaw
                    meta_age[ids, ptr] = stop_age
                    meta_cold[ids, ptr] = ~ever_moved
                    meta_stop[ids, ptr] = zero
                    obs, _, term, tout, _ = env.step(action)
                    done = term.clone() | tout.clone()
                    total_falls += int(term.sum()); total_timeouts += int(tout.sum())
                    if before_state is not None and (~done).any():
                        check_ids = (~done).nonzero().squeeze(-1)
                        assert torch.equal(replay.s[check_ids, ptr[check_ids]], before_state[check_ids]), 'Replay metadata points to a different physical state'
                    if done.any():
                        reset_ids = done.nonzero().squeeze(-1)
                        goal[reset_ids] = 0; stop_age[reset_ids] = 0; ever_moved[reset_ids] = False
                        obs, _ = env.reset(env_ids=reset_ids)
                        context.reset(reset_ids)
            row = dict(iteration=iteration, seconds=time.monotonic()-started, replay_rows=replay.size,
                       actor_updated=False, model_gate=False, total_falls=total_falls, total_timeouts=total_timeouts)
            mask = replay.window_mask(32)
            if iteration >= 2 and (mask & train_env[:, None]).any() and (mask & ~train_env[:, None]).any():
                if iteration == 2:
                    row['heldout_before_first_fit'] = heldout()
                wm.requires_grad_(True)
                for fit_index in range(a.wm_batches):
                    states, actions, xi = (fixed_dynamics.sample_train(a.wm_batch) if fit_index % 2 == 0
                                           else sample_windows(train_env, a.wm_batch))
                    pred = states[0]; loss = 0.
                    for j in range(32):
                        pred = wm.step(pred, actions[j], xi)
                        loss = loss + wm.spec.error_vector(pred, states[j+1]).square().sum(-1).mean() / 32
                    wm_opt.zero_grad(set_to_none=True); loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(wm.dynamics.parameters(), saved['wm_trainer']['grad_clip'])
                    if not torch.isfinite(loss) or not torch.isfinite(norm):
                        raise FloatingPointError('Nonfinite world model loss or gradient')
                    wm_opt.step()
                row['wm_loss'] = float(loss.detach())
                metrics, pose_metrics = heldout(); row['heldout_velocity_rmse'] = metrics; row['heldout_pose_rmse'] = pose_metrics
                row['recent_model_gate'] = (all(x < y for x, y in zip(metrics['8'], (.10, .10, .15)))
                    and all(x < y for x, y in zip(metrics['32'], (.20, .20, .30)))
                    and all(x < y for x, y in zip(pose_metrics['32'], (.03, .08))))
                row['fixed_phase_model_gate'] = fixed_dynamics.heldout_gate(wm)
                row['model_gate'] = row['recent_model_gate'] and row['fixed_phase_model_gate']['passed']
                actor_eligible = sampleable_start_mask(replay, history) & train_env[:, None]
                cold_starts = int((actor_eligible & meta_stop & meta_cold).sum())
                after_starts = int((actor_eligible & meta_stop & ~meta_cold).sum())
                row['sampleable_cold_starts'] = cold_starts
                row['sampleable_post_motion_stop_starts'] = after_starts
                actor_ready = cold_starts > 0
                row['actor_samples_ready'] = actor_ready
                row['fixed_source_stop_batch'] = a.batch // 2
                if row['model_gate'] and actor_ready:
                    wm.requires_grad_(False)
                    row['actor_result'] = trainer.update(actor_batch())
                    row['actor_updated'] = (row['actor_result']['actor_step_accepted']
                                            and row['actor_result']['actor_parameter_change_l2'] > 0)
                    ema.update(policy.standing_skill, changed=row['actor_updated'])
                    actor_updates += int(row['actor_updated'])
            row['actor_updates'] = actor_updates
            row['ema_updates'] = ema.updates
            row['cuda_peak_allocated_mib'] = torch.cuda.max_memory_allocated()/2**20
            trace.append(row)
            print(json.dumps(row), flush=True)
            if iteration % 20 == 0:
                save(iteration); last_saved = iteration
            if time.monotonic() - started >= a.hours * 3600:
                break
        if iteration != last_saved:
            save(iteration)
        (a.out/'report.json').write_text(json.dumps(dict(status='completed_candidate', algorithm='supermimic_bptt',
            iterations=iteration, actor_updates=actor_updates, ema=ema.metadata(), trace=trace), indent=2)+'\n')
    finally:
        env.close()


if __name__ == '__main__':
    main()
