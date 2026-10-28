"""Read-only decomposition of the current actor update and stop-task interference."""

from pathlib import Path


def main(checkpoint: Path, manifest: Path, out: Path):
    import json

    import torch
    import tyro

    from supermimic.config import TrainConfig
    from supermimic.locomotion.task import measured_velocity
    from supermimic.training.loop import Trainer

    if out.exists():
        raise FileExistsError(out)
    cfg = tyro.cli(TrainConfig, args=[
        "--robot", "x2_ultra", "--velocity.enabled", "--env.anchor-obs",
        "--run-name", out.parent.name + "_collection", "--resume", str(checkpoint),
        "--data.manifest", str(manifest), "--env.num-envs", "128", "--env.replay-capacity", "384",
        "--wm-trainer.compile", "None", "--policy-trainer.compile", "None", "--policy-trainer.horizon", "32",
    ])
    trainer = Trainer(cfg)
    actor, buffer = trainer.policy_trainer, trainer.buffer
    params = list(trainer.policy.encoders["velocity"].parameters())
    before = [p.detach().clone() for p in params]

    def flat(values):
        return torch.cat([(torch.zeros_like(p) if v is None else v).detach().flatten() for p, v in zip(params, values)])

    def cosine(a, b):
        return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-12))

    reports = []
    try:
        obs, _ = trainer.env.reset()
        with torch.no_grad():
            for _ in range(300):
                obs, _, _, _, _ = trainer.env.step(trainer.act(obs))
        valid = buffer.valid & ~buffer.terminal & buffer.velocity_env_mask[:, None]
        zero = valid & (buffer.command.square().sum(-1) < 1e-6)
        speeds = measured_velocity(buffer.s[zero])
        stats = {"zero_rows": int(zero.sum()), "velocity_rows": int(valid.sum()),
                 "zero_mean_velocity": speeds.mean(0).tolist(),
                 "zero_fraction_moving_over_0_2": float((speeds[:, :2].norm(dim=-1) > .2).float().mean())}
        for module in (trainer.wm, trainer.policy.decoder, actor.velocity_critic):
            module.requires_grad_(False)
        for seed in (101, 102, 103):
            torch.manual_seed(seed)
            start = buffer.sample_starts(256, 10, env_mask=buffer.velocity_env_mask)
            ro, align = actor._imagine_velocity(start)
            H = cfg.policy_trainer.horizon
            discount = cfg.policy_trainer.gamma ** torch.arange(H, device=trainer.device)
            reward = -(ro.rewards * ro.survival[:-1] * discount[:, None]).sum(0) / H
            tail = actor.velocity_critic(trainer.wm.spec.canonical(ro.states[-1]), ro.goals[-1], trainer.wm.spec.normalize_xi(start.xi))
            bootstrap = -(cfg.policy_trainer.gamma ** H) * ro.survival[-1] * tail / H
            zero = start.command.square().sum(-1) < 1e-6
            losses = {"stop_reward": reward[zero].mean(), "move_reward": reward[~zero].mean(),
                      "all_reward": reward.mean(), "bootstrap": bootstrap.mean(),
                      "alignment": cfg.velocity.alignment_weight * align}
            gradients = {name: flat(torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True))
                         for name, loss in losses.items()}
            full = gradients["all_reward"] + gradients["bootstrap"] + gradients["alignment"]
            clipped = full * min(1., cfg.policy_trainer.grad_clip_norm / float(full.norm().clamp_min(1e-12)))
            clipped = clipped.clamp(-cfg.policy_trainer.grad_clip_value, cfg.policy_trainer.grad_clip_value)
            updates, offset = [], 0
            for p in params:
                group = actor.velocity_opt.param_groups[0]
                state = actor.velocity_opt.state[p]
                g = clipped[offset:offset + p.numel()].reshape_as(p)
                offset += p.numel()
                b1, b2 = group["betas"]
                step = int(state["step"]) + 1
                m = b1 * state["exp_avg"] + (1 - b1) * g
                v = b2 * state["exp_avg_sq"] + (1 - b2) * g.square()
                updates.append(-group["lr"] * (m / (1 - b1**step)) / ((v / (1 - b2**step)).sqrt() + group["eps"]))
            adam = flat(updates)
            reports.append({
                "seed": seed, "stop_count": int(zero.sum()), "reference_count": int(start.command_reference.sum()),
                "losses": {key: float(value.detach()) for key, value in losses.items()},
                "gradient_norms": {key: float(value.norm()) for key, value in gradients.items()},
                "cosines_with_stop_reward_gradient": {key: cosine(value, gradients["stop_reward"]) for key, value in gradients.items()},
                "total_gradient_cosine_with_stop": cosine(full, gradients["stop_reward"]),
                "predicted_adam_update_cosine_with_stop_descent": cosine(adam, -gradients["stop_reward"]),
                "stop_model_velocity_32": measured_velocity(ro.states[-1])[zero].mean(0).detach().tolist(),
                "stop_initial_velocity": measured_velocity(start.state)[zero].mean(0).tolist(),
            })
            del ro, gradients, losses, full, reward, bootstrap, align, tail
        result = {"checkpoint": str(checkpoint), "no_optimizer_updates": True,
                  "encoder_unchanged": all(torch.equal(p, b) for p, b in zip(params, before)),
                  "fresh_replay": stats, "batches": reports}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        trainer.env.close()
        trainer.writer.close()


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
