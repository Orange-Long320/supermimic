"""Continuous start/stop/turn evaluation with every fall retained as a failed trial."""

from pathlib import Path


def main(ckpt: Path, out: Path, seconds: float = 60., num_envs: int = 8, device: str = "cuda:0", seed: int = 0, npz_dir: Path | None = None, teacher_still_arms: bool = False, extended_stop: bool = False):
    import json
    from dataclasses import asdict

    import torch
    from mjlab.envs import ManagerBasedRlEnv

    from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg
    from supermimic.envs.mdp.motion_library import discover_npz
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    from supermimic.robots.dimensions import PROFILE, NQ
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
    files = discover_npz(npz_dir or lafan_npz_dir(), names=["walk"] if npz_dir else ["walk1_subject1"])
    args = SuperMimicEnvCfgArgs(
        npz_files=[str(p) for p in files], num_envs=num_envs, seed=seed, future_frames=saved["env"]["future_frames"],
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
        (0., (0.,0.,0.)), (3.,(.3,0.,0.)), (10.,(0.,0.,0.)),
        (15.,(-.2,0.,0.)), (22.,(0.,0.,0.)), (27.,(0.,0.,.4)),
        (34.,(0.,0.,0.)), (39.,(0.,0.,-.4)), (46.,(0.,0.,0.)),
        (51.,(.6,0.,0.)), (58.,(0.,0.,0.)),
    ]
    arm_names = [n for n in PROFILE['joint_names'] if any(k in n for k in ('shoulder', 'elbow', 'wrist'))]
    arm_ids = [PROFILE['joint_names'].index(n) for n in arm_names]
    mj_model=env.sim.mj_model
    action_joint_names=[mj_model.joint(int(mj_model.actuator_trnid[i,0])).name.split('/')[-1] for i in range(mj_model.nu)]
    if action_joint_names != PROFILE['joint_names']:
        raise ValueError('Actuator order differs from the profile; resolve mapping before diagnostics')
    arm_stats = {}
    if extended_stop:
        schedule = schedule[:5]
    mailbox = CommandMailbox()
    records = []
    failed_at = torch.full((num_envs,), -1., device=device)
    resets = torch.zeros(num_envs, dtype=torch.long, device=device)
    error_sum = torch.zeros(num_envs, 3, device=device)
    error_count = torch.zeros(num_envs, device=device)
    stop_speed_sum = torch.zeros(num_envs, device=device)
    stop_count = torch.zeros(num_envs, device=device)
    try:
        observations, _ = env.reset()
        with torch.no_grad():
            for step in range(round(seconds / env.step_dt)):
                elapsed = step * env.step_dt
                phase_start, target = next((start, values) for start, values in reversed(schedule) if elapsed >= start)
                mailbox.put(target, now=elapsed)
                cmd.set_manual_command(mailbox.read(env.step_dt, now=elapsed))
                used_command = cmd.velocity_command.clone()
                action = policy.velocity_action(observations["policy"], used_command)
                if teacher_still_arms and target == (0., 0., 0.):
                    action[:, arm_ids] = 0.
                observations, _, terminated, timeout, _ = env.step(action)
                physical_state = cmd.state_io.read()
                measured = measured_velocity(physical_state)
                newly_failed = (terminated | timeout) & (failed_at < 0)
                failed_at[newly_failed] = elapsed + env.step_dt
                valid = (failed_at < 0) & (elapsed - phase_start >= 1.)
                if target == (0., 0., 0.) and valid.any():
                    q = physical_state[:, 7:NQ][:, arm_ids]
                    dq = physical_state[:, NQ + 6:][:, arm_ids]
                    if phase_start not in arm_stats:
                        arm_stats[phase_start] = dict(square=torch.zeros(num_envs,device=device),count=torch.zeros(num_envs,device=device),minimum=torch.full_like(q,float('inf')),maximum=torch.full_like(q,-float('inf')))
                    st = arm_stats[phase_start]
                    st['square'] += dq.square().mean(-1) * valid
                    st['count'] += valid
                    st['minimum'] = torch.where(valid[:,None], torch.minimum(st['minimum'],q),st['minimum'])
                    st['maximum'] = torch.where(valid[:,None], torch.maximum(st['maximum'],q),st['maximum'])
                error_sum += (measured - used_command).square() * valid[:, None]
                error_count += valid
                if target == (0., 0., 0.):
                    stop_speed_sum += measured[:, :2].norm(dim=-1) * valid
                    stop_count += valid
                if step % 10 == 0:
                    records.append({"time_s": elapsed + env.step_dt, "command": used_command.cpu().tolist(), "velocity": measured.cpu().tolist(), "active": (failed_at < 0).cpu().tolist()})
                done = (terminated | timeout).nonzero().squeeze(-1)
                if done.numel():
                    # Failed trials remain failed. Reset only to let the vector
                    # simulator continue collecting the other, surviving trials.
                    resets[done] += 1
                    observations, _ = env.reset(env_ids=done)
        result = {
            "teacher_action_override": teacher_still_arms,
            "arm_joint_names": arm_names,
            "action_joint_names": action_joint_names,
            "stop_arm_metrics": {str(k):dict(joint_velocity_rms_rad_s=torch.sqrt(v["square"]/v["count"].clamp_min(1)).cpu().tolist(),mean_joint_range_rad=(v["maximum"]-v["minimum"]).mean(-1).cpu().tolist(),samples=v["count"].cpu().tolist()) for k,v in arm_stats.items()},
            "checkpoint": str(ckpt.resolve()), "duration_s": seconds, "seed": args.seed,
            "num_envs": num_envs, "survived_without_reset": int((failed_at < 0).sum()),
            "first_failure_s": failed_at.cpu().tolist(), "resets": resets.cpu().tolist(),
            "steady_rmse_vx_vy_wz": torch.sqrt(error_sum / error_count.clamp_min(1)[:, None]).cpu().tolist(),
            "steady_samples": error_count.cpu().tolist(),
            "mean_stop_planar_speed_mps": (stop_speed_sum / stop_count.clamp_min(1)).cpu().tolist(),
            "stop_samples": stop_count.cpu().tolist(),
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
