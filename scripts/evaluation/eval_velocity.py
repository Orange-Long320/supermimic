"""Continuous start/stop/turn evaluation with every fall retained as a failed trial."""

from pathlib import Path


def main(ckpt: Path, out: Path, seconds: float = 60., num_envs: int = 8, device: str = "cuda:0"):
    import json
    from dataclasses import asdict

    import torch
    from mjlab.envs import ManagerBasedRlEnv

    from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg
    from supermimic.envs.mdp.motion_library import discover_npz
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    from supermimic.locomotion.config import VelocityCfg
    from supermimic.locomotion.keyboard import CommandMailbox
    from supermimic.locomotion.task import measured_velocity
    from supermimic.robots.paths import lafan_npz_dir

    if out.exists():
        raise FileExistsError(out)
    if seconds <= 0 or num_envs < 2:
        raise ValueError("Use a positive duration and at least two environments")
    policy, saved = load_policy_from_checkpoint(ckpt, device)
    if not saved.get("velocity", {}).get("enabled", False):
        raise ValueError("Checkpoint has no velocity interface")
    vc = VelocityCfg(**saved["velocity"])
    files = discover_npz(lafan_npz_dir(), names=["walk1_subject1"])
    args = SuperMimicEnvCfgArgs(
        npz_files=[str(p) for p in files], num_envs=num_envs, future_frames=saved["env"]["future_frames"],
        episode_length_s=seconds + 5, action_noise_std=0., obs_noise=False, reset_noise=False,
        domain_randomization=False, record_transitions=False, anchor_obs=saved["env"]["anchor_obs"], velocity=vc,
    )
    config = make_env_cfg(args)
    config.auto_reset = False
    env = ManagerBasedRlEnv(config, device=device)
    cmd = env.command_manager.get_term("motion")
    cmd.is_velocity[:] = True
    cmd.force_standing_reset = True
    cmd.set_manual_command((0., 0., 0.))
    schedule = [
        (0., (0., 0., 0.)), (3., (0.3, 0., 0.)), (13., (0., 0., 0.)),
        (18., (0.3, 0., 0.4)), (28., (0., 0., 0.)), (33., (0.3, 0., -0.4)),
        (43., (0., 0., 0.)), (48., (0.5, 0., 0.)), (58., (0., 0., 0.)),
    ]
    mailbox = CommandMailbox()
    records = []
    failed_at = torch.full((num_envs,), -1., device=device)
    resets = torch.zeros(num_envs, dtype=torch.long, device=device)
    try:
        observations, _ = env.reset()
        with torch.no_grad():
            for step in range(round(seconds / env.step_dt)):
                elapsed = step * env.step_dt
                target = next(values for start, values in reversed(schedule) if elapsed >= start)
                mailbox.put(target, now=elapsed)
                cmd.set_manual_command(mailbox.read(env.step_dt, now=elapsed))
                used_command = cmd.velocity_command.clone()
                action = policy.velocity_action(observations["policy"], used_command)
                observations, _, terminated, timeout, _ = env.step(action)
                measured = measured_velocity(cmd.state_io.read())
                newly_failed = (terminated | timeout) & (failed_at < 0)
                failed_at[newly_failed] = elapsed + env.step_dt
                if step % 10 == 0:
                    records.append({"time_s": elapsed + env.step_dt, "command": used_command.cpu().tolist(), "velocity": measured.cpu().tolist(), "active": (failed_at < 0).cpu().tolist()})
                done = (terminated | timeout).nonzero().squeeze(-1)
                if done.numel():
                    # Failed trials remain failed. Reset only to let the vector
                    # simulator continue collecting the other, surviving trials.
                    resets[done] += 1
                    observations, _ = env.reset(env_ids=done)
        result = {
            "checkpoint": str(ckpt.resolve()), "duration_s": seconds, "seed": args.seed,
            "num_envs": num_envs, "survived_without_reset": int((failed_at < 0).sum()),
            "first_failure_s": failed_at.cpu().tolist(), "resets": resets.cpu().tolist(),
            "schedule": schedule, "velocity_config": asdict(vc), "samples": records,
            "scope": "Deterministic simulation from default standing; survival alone does not prove command tracking.",
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
