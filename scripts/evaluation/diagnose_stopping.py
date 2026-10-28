"""Paired stopping counterfactuals and native checks of model reward gradients."""

from pathlib import Path


def main(checkpoint: Path, standing_checkpoint: Path, out: Path, device: str = "cuda:0"):
    import json
    from types import SimpleNamespace

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from torch.func import functional_call

    from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg
    from supermimic.envs.mdp.motion_library import discover_npz
    from supermimic.envs.sim_context import SimContext
    from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, FOOT_GEOM_REGEX
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    from supermimic.locomotion.config import VelocityCfg
    from supermimic.locomotion.task import fall_signals, measured_velocity, velocity_reward
    from supermimic.robots.paths import lafan_npz_dir
    from supermimic.world_model.heads import analytic_observation, flatten_history_term_major
    from supermimic.world_model.state_spec import StateSpecCfg
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(1)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    policy, cfg = load_policy_from_checkpoint(checkpoint, device)
    old, _ = load_policy_from_checkpoint(standing_checkpoint, device)
    vc = VelocityCfg(**cfg["velocity"])
    origin_steps = (400, 500, 600)  # switch to zero at 8 s; also inspect 2/4 s after
    steps = (0., -0.001, 0.001, -0.003, 0.003)
    per_origin = len(steps) + 2
    n = len(origin_steps) * per_origin
    files = discover_npz(lafan_npz_dir(), names=["walk1_subject1"])
    env_cfg = make_env_cfg(SuperMimicEnvCfgArgs(
        npz_files=[str(p) for p in files], num_envs=n, seed=0, episode_length_s=30.,
        action_noise_std=0., obs_noise=False, reset_noise=False, domain_randomization=False,
        record_transitions=False, anchor_obs=cfg["env"]["anchor_obs"], velocity=vc,
    ))
    env_cfg.auto_reset = False
    env = ManagerBasedRlEnv(env_cfg, device=device)
    cmd = env.command_manager.get_term("motion")
    cmd.is_velocity[:] = True
    cmd.force_standing_reset = True
    cmd.forced_frame[:] = cmd.library.clip_start[0]
    cmd.set_manual_command((0., 0., 0.))
    robot = env.scene["robot"]
    context = SimContext(env, FOOT_GEOM_REGEX, ANCHOR_BODY)
    limits = robot.data.soft_joint_pos_limits[0].clone()
    q0 = cmd.library.default_joint_pos
    settings = dict(cfg["world_model"])
    settings["state_spec"] = StateSpecCfg(**settings["state_spec"])
    wm = WorldModel(WorldModelCfg(**settings), SimpleNamespace(soft_limits=limits), device)
    wm.load_state_dict(saved["wm"])
    wm.requires_grad_(False)
    policy.decoder.requires_grad_(False)
    encoder = policy.encoders["velocity"]
    parameters = dict(encoder.named_parameters())
    param_norm = torch.sqrt(sum(p.detach().square().sum() for p in parameters.values()))
    gamma, dt = cfg["policy_trainer"]["gamma"], env.step_dt
    captures, trace, states, histories, previous_actions, contexts = {}, [], [], [], [], []
    variants, records = [], []

    def observations(state, action, xi):
        return analytic_observation(state, action, xi, q0)

    def action_for(kind, params, obs, target):
        if kind == "old_encoder_zero":
            latent = old.encoders["velocity"](obs, target)
        else:
            latent = encoder(obs, target) if params is None else functional_call(encoder, params, (obs, target))
        return policy.decoder(latent, obs)

    def model_rollout(capture, kind="current_zero", params=None, horizon=32, retain=False):
        state, history, previous, xi = capture
        target = torch.zeros(1, 3, device=device)
        if kind == "current_forward":
            target[:, 0] = 0.3
        survival = torch.ones(1, device=device)
        objective = torch.zeros((), device=device)
        velocities = []
        rates = []
        for step in range(horizon):
            obs = flatten_history_term_major(history)
            action = action_for(kind, params, obs, target)
            state = wm.step(state, action, xi)
            reward = velocity_reward(state, target, action, previous, limits, cmd.cfg.nominal_height, q0)
            objective += gamma**step * (survival * reward).mean() * dt / 32
            survival = survival * (1 - fall_signals(state, cmd.cfg.nominal_height, vc)[1])
            history = torch.cat((history[1:], observations(state, action, xi)[None]), 0)
            previous = action
            if retain:
                velocities.append(measured_velocity(state).detach())
                rates.append(reward.detach())
        return objective, velocities, rates

    try:
        obs, _ = env.reset()
        history = observations(cmd.state_io.read(), env.action_manager.action, context.read())[None].repeat(10, 1, 1)
        parity = 0.
        with torch.no_grad():
            for step in range(800):
                target = (0., 0., 0.) if step < 100 or step >= 400 else (0.3, 0., 0.)
                cmd.set_manual_command(target)
                if step in origin_steps:
                    captures[step] = (cmd.state_io.read()[:1].clone(), history[:, :1].clone(),
                                      env.action_manager.action[:1].clone(), context.read()[:1].clone())
                    parity = max(parity, float((flatten_history_term_major(history) - obs["policy"]).abs().max()))
                action = policy.velocity_action(obs["policy"], cmd.velocity_command)
                obs, _, terminated, timeout, _ = env.step(action)
                state, xi = cmd.state_io.read(), context.read()
                history = torch.cat((history[1:], observations(state, action, xi)[None]), 0)
                if step % 10 == 0:
                    trace.append({"time": (step + 1) * dt, "command": list(target),
                                  "actual_command": cmd.velocity_command[0].tolist(),
                                  "velocity": measured_velocity(state)[0].tolist()})
                if (terminated | timeout).any():
                    raise RuntimeError("Collection trajectory failed; do not silently reset")
        for origin in origin_steps:
            capture = captures[origin]
            value, _, _ = model_rollout(capture)
            gradient = torch.autograd.grad(value, tuple(parameters.values()))
            norm = torch.sqrt(sum(g.square().sum() for g in gradient))
            directions = [g.detach() * param_norm / norm.clamp_min(1e-12) for g in gradient]
            for epsilon in steps:
                params = {name: p.detach() + epsilon * g for (name, p), g in zip(parameters.items(), directions)}
                variants.append(("current_zero", params))
                records.append({"origin_step": origin, "kind": "current_zero", "epsilon": epsilon,
                                "reward_gradient_norm": float(norm)})
            for kind in ("current_forward", "old_encoder_zero"):
                variants.append((kind, None))
                records.append({"origin_step": origin, "kind": kind, "epsilon": 0.})
            for _ in range(per_origin):
                s, h, a, x = capture
                states.append(s[0]); histories.append(h[:, 0]); previous_actions.append(a[0]); contexts.append(x[0])
        with torch.no_grad():
            for (kind, params), row in zip(variants, records):
                value, velocity, rates = model_rollout(captures[row["origin_step"]], kind, params, horizon=64, retain=True)
                v, r = torch.cat(velocity), torch.cat(rates)
                # Recompute the comparison objective at the actual actor horizon.
                objective, _, _ = model_rollout(captures[row["origin_step"]], kind, params)
                row.update(model_objective_32=float(objective), model_velocity_1_8_32_64=v[[0, 7, 31, 63]].tolist(),
                           model_mean_reward_rate_32=float(r[:32].mean()))
        env.reset()
        ids = torch.arange(n, device=device)
        cmd.state_io.write(ids, torch.stack(states))
        previous = torch.stack(previous_actions)
        env.action_manager.action[:] = previous
        history, xi = torch.stack(histories, 1), torch.stack(contexts)
        targets = torch.zeros(n, 3, device=device)
        for index, (kind, _) in enumerate(variants):
            if kind == "current_forward":
                targets[index, 0] = 0.3
        # Per-environment manual commands keep the physical reward target fixed.
        cmd.manual_command = targets
        cmd._refresh_velocity()
        values, rates, alive = torch.zeros(n, device=device), [], torch.ones(n, dtype=torch.bool, device=device)
        survival = torch.ones(n, device=device)
        velocities, native_states, native_actions = [], [cmd.state_io.read().clone()], []
        reward_parity = 0.
        with torch.no_grad():
            for step in range(200):
                obs = flatten_history_term_major(history)
                actions = [action_for(kind, params, obs[i:i + 1], targets[i:i + 1])
                           for i, (kind, params) in enumerate(variants)]
                action = torch.cat(actions)
                _, native_reward, terminated, timeout, _ = env.step(action)
                state = cmd.state_io.read()
                reward = velocity_reward(state, targets, action, previous, limits, cmd.cfg.nominal_height, q0)
                reward_parity = max(reward_parity, float((native_reward / dt - reward).abs().max()))
                if step < 32:
                    values += gamma**step * survival * reward * dt / 32
                survival *= 1 - fall_signals(state, cmd.cfg.nominal_height, vc)[1]
                alive &= ~(terminated | timeout)
                velocities.append(measured_velocity(state).clone())
                rates.append(reward.clone())
                native_states.append(state.clone()); native_actions.append(action.clone())
                history = torch.cat((history[1:], observations(state, action, xi)[None]), 0)
                previous = action
                if (terminated | timeout).any():
                    break
        v, r = torch.stack(velocities), torch.stack(rates)
        for i, row in enumerate(records):
            row.update(native_objective_32=float(values[i]), failed=not bool(alive[i]),
                       native_velocity_1_8_32_64=v[[0, 7, 31, min(63, len(v)-1)], i].tolist(),
                       native_mean_velocity_last_second=v[-50:, i].mean(0).tolist(),
                       native_mean_reward_rate_32=float(r[:32, i].mean()),
                       native_mean_reward_rate_last_second=float(r[-50:, i].mean()))
        result = {"checkpoint": str(checkpoint), "standing_checkpoint": str(standing_checkpoint),
                  "horizon": 32, "native_steps": len(v), "observation_parity_max": parity,
                  "native_reward_parity_max": reward_parity, "no_optimizer_updates": True,
                  "collection_trace": trace, "records": records}
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"captures": captures, "native_states": torch.stack(native_states),
                    "native_actions": torch.stack(native_actions), "xi": xi, "targets": targets}, out.with_suffix(".pt"))
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
