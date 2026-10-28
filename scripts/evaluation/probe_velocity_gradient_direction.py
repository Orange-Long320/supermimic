"""Compare learned-model gradient directions with paired native physics returns.

Only functional encoder parameter variants are evaluated. Checkpoints and live
training sources remain unchanged; there are no optimizer steps.
"""

from pathlib import Path


def main(checkpoint: Path, trace: Path, out: Path, device: str = "cuda:0"):
    import json
    from types import SimpleNamespace

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from torch.func import functional_call

    from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg
    from supermimic.envs.mdp.motion_library import discover_npz
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    from supermimic.locomotion.config import VelocityCfg
    from supermimic.locomotion.task import fall_signals, velocity_reward
    from supermimic.robots.paths import lafan_npz_dir
    from supermimic.world_model.heads import analytic_observation, flatten_history_term_major
    from supermimic.world_model.state_spec import StateSpecCfg
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(1)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    data = torch.load(trace, map_location=device, weights_only=False)
    policy, cfg = load_policy_from_checkpoint(checkpoint, device)
    vc = VelocityCfg(**cfg["velocity"])
    files = discover_npz(lafan_npz_dir(), names=["walk1_subject1"])
    origins, column, horizon = [200, 300, 400], 16, 32
    relative_steps = [-0.001, -0.0003, 0., 0.0003, 0.001]
    env_cfg = make_env_cfg(SuperMimicEnvCfgArgs(
        npz_files=[str(p) for p in files], num_envs=len(origins) * len(relative_steps), seed=0,
        episode_length_s=10., action_noise_std=0., obs_noise=False, reset_noise=False,
        domain_randomization=False, record_transitions=False, anchor_obs=True, velocity=vc,
    ))
    env_cfg.auto_reset = False
    env = ManagerBasedRlEnv(env_cfg, device=device)
    cmd = env.command_manager.get_term("motion")
    cmd.is_velocity[:] = True
    cmd.force_standing_reset = True
    cmd.command_period = 10**9
    cmd.command_from_reference[:] = False
    cmd.velocity_command[:] = 0
    robot = env.scene["robot"]
    limits = robot.data.soft_joint_pos_limits[0].clone()
    q0 = cmd.library.default_joint_pos
    settings = dict(cfg["world_model"])
    settings["state_spec"] = StateSpecCfg(**settings["state_spec"])
    wm = WorldModel(WorldModelCfg(**settings), SimpleNamespace(soft_limits=limits), device)
    wm.load_state_dict(saved["wm"])
    wm.requires_grad_(False)
    policy.decoder.requires_grad_(False)
    encoder = policy.encoders["velocity"]
    names = [name for name, _ in encoder.named_parameters()]
    parameters = list(encoder.parameters())
    norm = torch.sqrt(sum(p.detach().square().sum() for p in parameters))
    command = torch.tensor([[0.3, 0., 0.]], device=device)
    gamma = cfg["policy_trainer"]["gamma"]
    dt = env.step_dt
    variants, initial_states, initial_histories, initial_previous, records = [], [], [], [], []

    def initial(origin):
        history = [analytic_observation(data["states"][t, column:column + 1], data["actions"][t - 1, column:column + 1], data["context"][t, column:column + 1], q0) for t in range(origin - 9, origin + 1)]
        return data["states"][origin, column:column + 1], history, data["actions"][origin - 1, column:column + 1], data["context"][origin, column:column + 1]

    def imagined(params, origin):
        state, history, previous, xi = initial(origin)
        value = torch.zeros((), device=device)
        survival = torch.ones(1, device=device)
        for step in range(horizon):
            obs = flatten_history_term_major(torch.stack(history))
            latent = encoder(obs, command) if params is None else functional_call(encoder, params, (obs, command))
            action = policy.decoder(latent, obs)
            state = wm.step(state, action, xi)
            reward = velocity_reward(state, command, action, previous, limits, cmd.cfg.nominal_height, q0)
            value = value + gamma**step * (survival * reward).mean() * dt / horizon
            survival = survival * (1 - fall_signals(state, cmd.cfg.nominal_height, vc)[1])
            history = history[1:] + [analytic_observation(state, action, xi, q0)]
            previous = action
        return value

    try:
        for origin in origins:
            value = imagined(None, origin)
            gradient = torch.autograd.grad(value, parameters)
            gradient_norm = torch.sqrt(sum(g.square().sum() for g in gradient))
            direction = [g.detach() / gradient_norm.clamp_min(1e-12) * norm for g in gradient]
            for relative in relative_steps:
                params = {name: p.detach() + relative * direction_value for name, p, direction_value in zip(names, parameters, direction)}
                with torch.no_grad():
                    predicted = float(imagined(params, origin))
                    state, history, previous, _ = initial(origin)
                variants.append(params)
                initial_states.append(state[0].clone())
                initial_histories.append(torch.cat(history, 0))
                initial_previous.append(previous[0].clone())
                records.append({"trace_step": origin, "relative_parameter_step": relative, "model_objective": predicted,
                                "model_gradient_norm": float(gradient_norm), "native_objective": 0., "failed": False})
        env.reset()
        ids = torch.arange(env.num_envs, device=device)
        cmd.state_io.write(ids, torch.stack(initial_states))
        env.action_manager.action[:] = torch.stack(initial_previous)
        previous = torch.stack(initial_previous)
        histories = torch.stack(initial_histories, 1)
        xi = data["context"][origins, column].repeat_interleave(len(relative_steps), 0)
        target = command.expand(env.num_envs, -1)
        alive = torch.ones(env.num_envs, dtype=torch.bool, device=device)
        survival = torch.ones(env.num_envs, device=device)
        total = torch.zeros(env.num_envs, device=device)
        parity = torch.zeros((), device=device)
        with torch.no_grad():
            for step in range(horizon):
                obs = flatten_history_term_major(histories)
                actions = []
                for index, params in enumerate(variants):
                    latent = functional_call(encoder, params, (obs[index:index + 1], command))
                    actions.append(policy.decoder(latent, obs[index:index + 1]))
                action = torch.cat(actions)
                cmd.command_from_reference[:] = False
                cmd.velocity_command[:] = target
                _, native, terminated, timeout, _ = env.step(action)
                state = cmd.state_io.read()
                reward = velocity_reward(state, target, action, previous, limits, cmd.cfg.nominal_height, q0)
                parity = torch.maximum(parity, (reward - native / dt).abs().max())
                total += gamma**step * survival * reward * dt / horizon
                survival *= 1 - fall_signals(state, cmd.cfg.nominal_height, vc)[1]
                alive &= ~(terminated | timeout)
                histories = torch.cat((histories[1:], analytic_observation(state, action, xi, q0)[None]), 0)
                previous = action
                done = (terminated | timeout).nonzero().squeeze(-1)
                if done.numel():
                    env.reset(env_ids=done)
        for index, row in enumerate(records):
            row["native_objective"] = float(total[index])
            row["failed"] = not bool(alive[index])
        derivatives = []
        for origin in origins:
            rows = {r["relative_parameter_step"]: r for r in records if r["trace_step"] == origin}
            for epsilon in (0.0003, 0.001):
                positive, negative = rows[epsilon], rows[-epsilon]
                derivatives.append({"trace_step": origin, "epsilon": epsilon,
                    "model_directional_derivative": (positive["model_objective"] - negative["model_objective"]) / (2 * epsilon),
                    "native_directional_derivative": (positive["native_objective"] - negative["native_objective"]) / (2 * epsilon)})
        result = {"checkpoint": str(checkpoint.resolve()), "trace": str(trace.resolve()), "horizon": horizon,
                  "method": "Symmetric functional encoder perturbations along learned-model reward ascent; shared native initial states and histories; no optimizer updates.",
                  "native_reward_parity_max": float(parity), "records": records, "directional_derivatives": derivatives}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
