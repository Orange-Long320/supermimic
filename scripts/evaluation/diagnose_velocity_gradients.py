"""Collect fresh diagnostic replay and inspect objective gradients without any updates."""

from pathlib import Path


def main(checkpoint: Path, manifest: Path, out: Path, collection_steps: int = 300):
    import json

    import torch
    import tyro

    from supermimic.config import TrainConfig
    from supermimic.imagination.objective import policy_objective
    from supermimic.locomotion.task import measured_velocity
    from supermimic.training.loop import Trainer

    if out.exists():
        raise FileExistsError(out)
    cfg = tyro.cli(TrainConfig, args=[
        "--robot", "x2_ultra", "--velocity.enabled", "--env.anchor-obs",
        "--run-name", out.parent.name + "_collection", "--resume", str(checkpoint),
        "--data.manifest", str(manifest), "--env.num-envs", "128", "--env.replay-capacity", "384",
        "--wm-trainer.batch-size", "128", "--wm-trainer.batches-per-update", "2", "--wm-trainer.compile", "None",
        "--policy-trainer.batch-size", "128", "--policy-trainer.horizon", "32", "--policy-trainer.compile", "None",
    ])
    trainer = Trainer(cfg)
    actor = trainer.policy_trainer
    buffer = trainer.buffer
    optimizer_before = {key: value.clone() for key, value in trainer.policy.state_dict().items()}
    fractions = []
    try:
        obs, _ = trainer.env.reset()
        with torch.no_grad():
            for step in range(collection_steps):
                obs, _, _, _, _ = trainer.env.step(trainer.act(obs))
                if step % 10 == 0:
                    mask = trainer.command.is_velocity
                    fractions.append({"step": step + 1, "reference_fraction": float(trainer.command.command_from_reference[mask].float().mean()),
                                      "mean_command": trainer.command.velocity_command[mask].mean(0).tolist()})
        valid_velocity = buffer.valid & ~buffer.terminal & buffer.velocity_env_mask[:, None]
        reference = buffer.command_reference & valid_velocity
        replay_stats = {
            "velocity_rows": int(valid_velocity.sum()), "reference_rows": int(reference.sum()),
            "reference_fraction": float(reference.sum() / valid_velocity.sum()),
            "mean_command": buffer.command[valid_velocity].mean(0).tolist(),
            "zero_command_fraction": float((buffer.command[valid_velocity].norm(dim=-1) < 1e-5).float().mean()),
        }
        parameters = list(trainer.policy.encoders["velocity"].parameters())
        for module in (trainer.wm, trainer.policy.decoder, actor.velocity_critic):
            module.requires_grad_(False)
        gradients = []
        for kind, mask in (("normal_velocity_replay", buffer.velocity_env_mask), ("reference_only_replay", None)):
            if kind == "normal_velocity_replay":
                start = buffer.sample_starts(256, 10, env_mask=mask)
            else:
                # Preserve the standard history sampler, then retain only rows
                # where a real recorded reference command was active.
                large = buffer.sample_starts(4096, 10, env_mask=buffer.velocity_env_mask)
                keep = large.command_reference.nonzero().squeeze(-1)[:256]
                if keep.numel() == 0:
                    continue
                from supermimic.data.replay_buffer import StartBatch

                start = StartBatch(
                    large.state[keep], large.xi[keep], large.frame[keep], large.prev_action[keep],
                    large.hist_states[:, keep], large.hist_actions[:, keep], large.command[keep], large.command_reference[keep],
                )
            rollout, align = actor._imagine_velocity(start)
            xi = trainer.wm.spec.normalize_xi(start.xi)
            final_feature = trainer.wm.spec.canonical(rollout.states[-1])
            tail = actor.velocity_critic(final_feature, rollout.goals[-1], xi)
            immediate = -policy_objective(rollout, None, cfg.policy_trainer.gamma) / cfg.policy_trainer.horizon
            bootstrap = -(cfg.policy_trainer.gamma**cfg.policy_trainer.horizon) * (rollout.survival[-1] * tail).mean() / cfg.policy_trainer.horizon
            alignment = cfg.velocity.alignment_weight * align
            vectors = {}
            for name, loss in (("reward", immediate), ("critic_bootstrap", bootstrap), ("alignment", alignment)):
                values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
                vectors[name] = torch.cat([torch.zeros_like(p).reshape(-1) if value is None else value.reshape(-1) for p, value in zip(parameters, values)])
            total = sum(vectors.values())
            def cosine(a, b):
                return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-12))
            gradients.append({
                "sample": kind, "count": len(start.state), "reference_count": int(start.command_reference.sum()),
                "loss": {"reward": float(immediate.detach()), "critic_bootstrap": float(bootstrap.detach()), "alignment": float(alignment.detach())},
                "gradient_norm": {key: float(value.norm()) for key, value in vectors.items()},
                "cosine_with_total": {key: cosine(value, total) for key, value in vectors.items()},
                "reward_vs_bootstrap_cosine": cosine(vectors["reward"], vectors["critic_bootstrap"]),
                "reward_vs_alignment_cosine": cosine(vectors["reward"], vectors["alignment"]),
                "predicted_velocity_mean": measured_velocity(rollout.states[-1]).mean(0).detach().tolist(),
                "initial_velocity_mean": measured_velocity(start.state).mean(0).tolist(),
                "command_mean": rollout.goals[0].mean(0).tolist(),
            })
            del rollout, total, vectors
        world_model_checks = []
        with torch.no_grad():
            for horizon in (1, 8, 32):
                mask = buffer.window_mask(horizon) & buffer.velocity_env_mask[:, None]
                e, t = buffer._sample_from_mask(mask, 256)
                state = buffer.s[e, t]
                original = state.clone()
                model_reward, native_reward = torch.zeros(256, device=trainer.device), torch.zeros(256, device=trainer.device)
                from supermimic.locomotion.task import velocity_reward

                for offset in range(horizon):
                    time = (t + offset) % buffer.T
                    action = buffer.a_exec[e, time]
                    state = trainer.wm.step(state, action, buffer.xi[e, time])
                    previous = buffer.a[e, (time - 1) % buffer.T]
                    previous = torch.where(buffer.ep_start[e, time, None], 0., previous)
                    model_reward += velocity_reward(state, buffer.command[e, time], buffer.a[e, time], previous,
                                                    trainer.wm.heads.soft_limits, trainer.command.cfg.nominal_height, trainer.lib.default_joint_pos)
                    native_reward += buffer.r[e, time] / trainer.env.step_dt
                truth = buffer.s[e, (t + horizon) % buffer.T]
                prediction, actual = measured_velocity(state), measured_velocity(truth)
                error = prediction - actual
                world_model_checks.append({"horizon": horizon, "velocity_bias": error.mean(0).tolist(),
                    "velocity_rmse": error.square().mean(0).sqrt().tolist(), "predicted_velocity": prediction.mean(0).tolist(),
                    "actual_velocity": actual.mean(0).tolist(), "initial_velocity": measured_velocity(original).mean(0).tolist(),
                    "mean_command": buffer.command[e, t].mean(0).tolist(), "model_reward_rate": float(model_reward.mean() / horizon),
                    "native_reward_rate": float(native_reward.mean() / horizon)})
        unchanged = all(torch.equal(value, trainer.policy.state_dict()[key]) for key, value in optimizer_before.items())
        result = {"checkpoint": str(checkpoint.resolve()), "collection_steps": collection_steps, "no_optimizer_updates": True,
                  "policy_weights_unchanged": unchanged, "replay_stats": replay_stats, "reference_fraction_over_time": fractions,
                  "objective_gradients": gradients, "world_model_on_fresh_replay": world_model_checks}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        trainer.env.close()
        if trainer.writer is not None:
            trainer.writer.close()


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
