"""Read-only paired component swaps and dynamics checks for saved velocity policies.

Run on the project server through scripts/robot.py. No optimizer steps, checkpoint
writes, changes to the training environment or model-parameter repairs are made.
"""

from pathlib import Path


def main(early: Path, late: Path, out: Path, seconds: float = 12., device: str = "cuda:0"):
    import hashlib
    import json
    from types import SimpleNamespace

    import torch
    from mjlab.envs import ManagerBasedRlEnv

    from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg
    from supermimic.envs.mdp.motion_library import discover_npz
    from supermimic.envs.sim_context import SimContext
    from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, FOOT_GEOM_REGEX
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    from supermimic.kinematics.rotations import projected_gravity
    from supermimic.locomotion.config import VelocityCfg
    from supermimic.locomotion.policy import VelocityPolicy
    from supermimic.locomotion.task import bounded_reference, measured_velocity, velocity_reward
    from supermimic.policy.nets import PolicyCfg
    from supermimic.robots.dimensions import NQ
    from supermimic.robots.paths import lafan_npz_dir
    from supermimic.world_model.heads import analytic_observation, flatten_history_term_major
    from supermimic.world_model.state_spec import StateSpecCfg
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(1)
    source = [torch.load(path, map_location=device, weights_only=False) for path in (early, late)]
    policies = [load_policy_from_checkpoint(path, device)[0] for path in (early, late)]
    config = source[1]["config"]
    vc = VelocityCfg(**config["velocity"])
    policy_cfg = PolicyCfg(**config["policy"])
    combinations = [(0, 0), (1, 0), (0, 1), (1, 1)]  # velocity encoder, shared decoder
    names = ["early_encoder_early_decoder", "late_encoder_early_decoder", "early_encoder_late_decoder", "late_encoder_late_decoder"]
    variants = []
    weight_changes = {}
    for prefix in ("encoders.velocity.", "encoders.motion.", "decoder."):
        keys = [k for k, v in source[0]["policy"].items() if k.startswith(prefix) and v.is_floating_point()]
        numerator = sum((source[1]["policy"][k] - source[0]["policy"][k]).square().sum() for k in keys)
        denominator = sum(source[0]["policy"][k].square().sum() for k in keys)
        weight_changes[prefix] = float((numerator / denominator.clamp_min(1e-12)).sqrt())
    for encoder, decoder in combinations:
        policy = VelocityPolicy(policy_cfg, vc).to(device).eval()
        policy.load_state_dict(source[encoder]["policy"])
        policy.decoder.load_state_dict(policies[decoder].decoder.state_dict())
        variants.append(policy)
    commands = torch.tensor([[0., 0., 0.], [0.3, 0., 0.], [0.5, 0., 0.], [0.3, 0., 0.4], [0.3, 0., -0.4]], device=device)
    per_variant = len(commands)
    n = len(variants) * per_variant
    files = discover_npz(lafan_npz_dir(), manifest=config["data"]["manifest"])
    env_cfg = make_env_cfg(SuperMimicEnvCfgArgs(
        npz_files=[str(p) for p in files], num_envs=n, seed=0, episode_length_s=seconds + 5,
        action_noise_std=0., obs_noise=False, reset_noise=False, domain_randomization=False,
        record_transitions=False, anchor_obs=config["env"]["anchor_obs"], velocity=vc,
    ))
    env_cfg.auto_reset = False
    env = ManagerBasedRlEnv(env_cfg, device=device)
    cmd = env.command_manager.get_term("motion")
    cmd.is_velocity[:] = True
    cmd.force_standing_reset = True
    cmd.command_period = 10**9
    cmd.command_from_reference[:] = False
    cmd.velocity_command[:] = 0
    # All environments start at the same standing pose/reference heading.
    cmd.forced_frame[:] = cmd.library.clip_start[0]
    robot = env.scene["robot"]
    context = SimContext(env, FOOT_GEOM_REGEX, ANCHOR_BODY)
    heads = SimpleNamespace(soft_limits=robot.data.soft_joint_pos_limits[0].clone())
    models = []
    for checkpoint in source:
        settings = dict(checkpoint["config"]["world_model"])
        settings["state_spec"] = StateSpecCfg(**settings["state_spec"])
        wm = WorldModel(WorldModelCfg(**settings), heads, device).eval()
        wm.load_state_dict(checkpoint["wm"])
        models.append(wm)
    states, actions, contexts, desired, rewards, decomposition, observations_log, active_log = [], [], [], [], [], [], [], []
    failures = torch.full((n,), -1., device=device)

    def reward_parts(state, target, action, previous):
        velocity = measured_velocity(state)
        gravity = projected_gravity(state[:, 3:7])
        limits = robot.data.soft_joint_pos_limits
        joints = state[:, 7:NQ]
        return torch.stack([
            2 * torch.exp(-((velocity[:, :2] - target[:, :2]) / 0.5).square().sum(-1)),
            2 * torch.exp(-(velocity[:, 2] - target[:, 2]).square() / 0.5),
            -2 * gravity[:, :2].square().sum(-1),
            -0.5 * state[:, NQ + 2].square(),
            -0.05 * state[:, NQ + 3:NQ + 5].square().sum(-1),
            -2 * ((state[:, 2] - cmd.cfg.nominal_height) / cmd.cfg.nominal_height).square(),
            -0.1 * (action - previous).square().sum(-1),
            -2 * (torch.relu(limits[:, :, 0] - joints) + torch.relu(joints - limits[:, :, 1])).sum(-1),
            -0.5 * (joints - robot.data.default_joint_pos).square().mean(-1) * (target.square().sum(-1) < 1e-6),
        ], -1)

    try:
        obs, _ = env.reset()
        initial_context = context.read()
        with torch.no_grad():
            for step in range(round(seconds / env.step_dt)):
                target = commands.repeat(len(variants), 1) if step * env.step_dt >= 2 else torch.zeros(n, 3, device=device)
                cmd.command_from_reference[:] = False
                cmd.velocity_command[:] = target
                before = cmd.state_io.read()
                states.append(before.clone())
                contexts.append(context.read().clone())
                previous = env.action_manager.action.clone()
                action = torch.zeros_like(previous)
                for i, policy in enumerate(variants):
                    ids = slice(i * per_variant, (i + 1) * per_variant)
                    action[ids] = policy.velocity_action(obs["policy"][ids], target[ids])
                observations_log.append(obs["policy"].clone())
                obs, reward, terminated, timeout, _ = env.step(action)
                after = cmd.state_io.read()
                parts = reward_parts(after, target, action, previous)
                just_failed = (terminated | timeout) & (failures < 0)
                failures[just_failed] = (step + 1) * env.step_dt
                actions.append(action.clone())
                desired.append(target.clone())
                rewards.append(reward.clone() / env.step_dt)
                decomposition.append(parts)
                active_log.append((failures < 0).clone())
                done = (terminated | timeout).nonzero().squeeze(-1)
                if done.numel():
                    obs, _ = env.reset(env_ids=done)
            states.append(cmd.state_io.read().clone())
        state = torch.stack(states)
        action = torch.stack(actions)
        xi = torch.stack(contexts)
        target = torch.stack(desired)
        real_reward = torch.stack(rewards)
        parts = torch.stack(decomposition)
        active = torch.stack(active_log)
        velocities = measured_velocity(state[1:])
        results = []
        part_names = ["linear", "yaw", "upright", "vertical", "roll_pitch", "height", "action_rate", "joint_limit", "standing_posture"]
        start_index = round(4 / env.step_dt)
        for i, name in enumerate(names):
            for j in range(per_variant):
                index = i * per_variant + j
                use = active[start_index:, index]
                result = {"variant": name, "command": commands[j].tolist(), "first_failure_s": float(failures[index])}
                if use.any():
                    result.update(
                        measured_velocity_mean=velocities[start_index:, index][use].mean(0).tolist(),
                        reward_rate=float(real_reward[start_index:, index][use].mean()),
                        reward_parts={key: float(parts[start_index:, index, k][use].mean()) for k, key in enumerate(part_names)},
                        decomposition_parity_max=float((parts[start_index:, index][use].sum(-1) - real_reward[start_index:, index][use]).abs().max()),
                    )
                results.append(result)
        wm_checks = []
        with torch.no_grad():
            for index, wm in enumerate(models):
                predicted = wm.step(state[:-1].reshape(-1, state.shape[-1]), action.reshape(-1, action.shape[-1]), xi.reshape(-1, xi.shape[-1])).reshape_as(state[1:])
                one_step_error = measured_velocity(predicted) - velocities
                for variant, name in enumerate(names):
                    ids = slice(variant * per_variant, (variant + 1) * per_variant)
                    wm_checks.append({"model": "early" if index == 0 else "late", "policy_variant": name,
                        "one_step_velocity_bias": one_step_error[start_index:, ids].mean((0, 1)).tolist(),
                        "one_step_velocity_rmse": one_step_error[start_index:, ids].square().mean((0, 1)).sqrt().tolist()})
                for horizon in (8, 32, 64):
                    origins = [round(t / env.step_dt) for t in (4, 6, 8) if round(t / env.step_dt) + horizon < len(action)]
                    if not origins:
                        continue
                    predicted_state = state[origins].reshape(-1, state.shape[-1])
                    predicted_return = torch.zeros(len(origins) * n, device=device)
                    for offset in range(horizon):
                        step_action = action[[origin + offset for origin in origins]].reshape(-1, action.shape[-1])
                        step_context = xi[[origin + offset for origin in origins]].reshape(-1, xi.shape[-1])
                        predicted_state = wm.step(predicted_state, step_action, step_context)
                        prev = action[[origin + offset - 1 for origin in origins]].reshape_as(step_action)
                        goal = target[[origin + offset for origin in origins]].reshape(-1, 3)
                        predicted_return += velocity_reward(predicted_state, goal, step_action, prev, heads.soft_limits, cmd.cfg.nominal_height, robot.data.default_joint_pos[0])
                    estimate = measured_velocity(predicted_state).reshape(len(origins), n, 3)
                    truth = measured_velocity(state[[origin + horizon for origin in origins]])
                    native_return = torch.stack([real_reward[origin:origin + horizon].sum(0) for origin in origins])
                    for variant, name in enumerate(names):
                        ids = slice(variant * per_variant, (variant + 1) * per_variant)
                        error = estimate[:, ids] - truth[:, ids]
                        wm_checks.append({"model": "early" if index == 0 else "late", "policy_variant": name, "horizon": horizon,
                            "velocity_bias": error.mean((0, 1)).tolist(), "velocity_rmse": error.square().mean((0, 1)).sqrt().tolist(),
                            "predicted_reward_rate": float(predicted_return.reshape(len(origins), n)[:, ids].mean() / horizon),
                            "native_reward_rate": float(native_return[:, ids].mean() / horizon)})
        table = cmd.reference_velocity
        _, reference_valid = bounded_reference(table, torch.arange(len(table), device=device), vc)
        coverage = []
        for clip in cmd.library.clips:
            indices = slice(clip.start, clip.end)
            coverage.append({"clip": clip.name, "reference_valid_fraction": float(reference_valid[indices].float().mean()),
                             "raw_command_mean": table[indices].mean(0).tolist()})
        history_index = round(4 / env.step_dt)
        history = torch.stack([analytic_observation(state[t], action[t - 1], xi[t], cmd.library.default_joint_pos) for t in range(history_index - 9, history_index + 1)])
        reconstructed = flatten_history_term_major(history)
        sensitivity = []
        with torch.no_grad():
            fixed_obs = observations_log[history_index][0:1].repeat(per_variant, 1)
            for name, policy in zip(names, variants):
                latent = policy.encoders["velocity"](fixed_obs, commands)
                output = policy.velocity_action(fixed_obs, commands)
                sensitivity.append({"variant": name, "action_rms_from_zero_command": (output - output[:1]).square().mean(-1).sqrt().tolist(),
                                    "latent_rms_from_zero_command": (latent - latent[:1]).square().mean(-1).sqrt().tolist(),
                                    "latent_saturated_fraction": float((latent.abs() > 0.95).float().mean())})
        hashes = {}
        for path in (early, late):
            with path.open("rb") as stream:
                hashes[str(path.resolve())] = hashlib.file_digest(stream, "sha256").hexdigest()
        result = {"checkpoints": hashes, "duration_s": seconds, "initialization": "identical default standing and forced first reference frame", "weight_relative_l2_changes": weight_changes,
                  "component_swap_rollouts": results, "world_model_checks": wm_checks, "reference_coverage": coverage,
                  "observation_history_reconstruction_max_abs_error": float((reconstructed - observations_log[history_index]).abs().max()),
                  "command_sensitivity": sensitivity, "context_initial": initial_context[0].tolist()}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k not in ("world_model_checks", "reference_coverage", "context_initial")}, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
