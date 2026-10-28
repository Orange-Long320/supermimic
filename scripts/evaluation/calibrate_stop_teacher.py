"""Native counterfactual calibration of an old stop skill, for training only."""

from pathlib import Path


def main(checkpoint: Path, stop_teacher: Path, out: Path, device: str = "cuda:0"):
    import json

    import torch
    from mjlab.envs import ManagerBasedRlEnv

    from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg
    from supermimic.envs.mdp.motion_library import discover_npz
    from supermimic.envs.sim_context import SimContext
    from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, FOOT_GEOM_REGEX
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    from supermimic.locomotion.config import VelocityCfg
    from supermimic.locomotion.task import measured_velocity
    from supermimic.robots.paths import lafan_npz_dir
    from supermimic.world_model.heads import analytic_observation, flatten_history_term_major

    if out.exists():
        raise FileExistsError(out)
    policy, cfg = load_policy_from_checkpoint(checkpoint, device)
    stop, _ = load_policy_from_checkpoint(stop_teacher, device)
    queries = torch.tensor([[v, 0., w] for v in (-.1, -.05, 0.) for w in (0., .1, .2, .3, .4)], device=device)
    goals = torch.tensor([[.3, 0., 0.], [.5, 0., 0.], [.3, 0., .4], [.3, 0., -.4], [0., 0., 0.]], device=device)
    n, k = len(queries) * len(goals), len(queries)
    files = discover_npz(lafan_npz_dir(), names=["walk1_subject1"])
    env_cfg = make_env_cfg(SuperMimicEnvCfgArgs(
        npz_files=[str(p) for p in files], num_envs=n, seed=0, episode_length_s=20,
        action_noise_std=0., obs_noise=False, reset_noise=False, domain_randomization=False,
        record_transitions=False, anchor_obs=True, velocity=VelocityCfg(**cfg["velocity"]),
    ))
    env_cfg.auto_reset = False
    env = ManagerBasedRlEnv(env_cfg, device=device)
    cmd = env.command_manager.get_term("motion")
    cmd.is_velocity[:] = True
    cmd.force_standing_reset = True
    cmd.forced_frame[:] = cmd.library.clip_start[0]
    cmd.set_manual_command((0., 0., 0.))
    context = SimContext(env, FOOT_GEOM_REGEX, ANCHOR_BODY)
    q0 = cmd.library.default_joint_pos
    try:
        obs, _ = env.reset()
        initial = cmd.state_io.read().clone()
        xi = context.read()
        initial_obs = analytic_observation(initial, env.action_manager.action, xi, q0)
        history = initial_obs[None].repeat(10, 1, 1)
        target = goals.repeat_interleave(k, 0)
        with torch.no_grad():
            for step in range(350):
                cmd.manual_command = target if step >= 50 else torch.zeros_like(target)
                cmd._refresh_velocity()
                action = policy.velocity_action(obs["policy"], cmd.velocity_command)
                obs, _, terminated, timeout, _ = env.step(action)
                if (terminated | timeout).any():
                    raise RuntimeError("Pre-calibration walking trajectory failed")
                state = cmd.state_io.read()
                history = torch.cat((history[1:], analytic_observation(state, action, xi, q0)[None]), 0)
            ids = torch.arange(4 * k, 5 * k, device=device)
            cmd.state_io.write(ids, initial[ids])
            env.action_manager.action[ids] = 0
            history[:, ids] = initial_obs[ids][None]
            cmd.manual_command = torch.zeros_like(target)
            cmd._refresh_velocity()
            query = queries.repeat(len(goals), 1)
            velocities, failed = [], torch.zeros(n, dtype=torch.bool, device=device)
            for step in range(200):
                obs = flatten_history_term_major(history)
                action = policy.decoder(stop.encoders["velocity"](obs, query), obs)
                _, _, terminated, timeout, _ = env.step(action)
                failed |= terminated | timeout
                state = cmd.state_io.read()
                velocities.append(measured_velocity(state).clone())
                history = torch.cat((history[1:], analytic_observation(state, action, xi, q0)[None]), 0)
            velocities = torch.stack(velocities)
        results = []
        for j, query in enumerate(queries):
            rows = []
            for i, goal in enumerate(goals):
                index = i * k + j
                end = velocities[-50:, index]
                rows.append({"prior_goal": goal.tolist(), "failed": bool(failed[index]),
                             "velocity_last_second": end.mean(0).tolist(),
                             "speed_last_second": float(end[:, :2].norm(dim=-1).mean()),
                             "speed_second_2": float(velocities[50:100, index, :2].norm(dim=-1).mean())})
            results.append({"query": query.tolist(), "origins": rows,
                            "mean_speed": sum(r["speed_last_second"] for r in rows) / len(rows),
                            "mean_abs_yaw": sum(abs(r["velocity_last_second"][2]) for r in rows) / len(rows),
                            "any_failed": any(r["failed"] for r in rows)})
        result = {"checkpoint": str(checkpoint), "teacher": str(stop_teacher),
                  "physical_command": [0., 0., 0.], "results": results}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(sorted(results, key=lambda r: r["mean_speed"] + r["mean_abs_yaw"])[:5], indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
