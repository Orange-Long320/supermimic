"""Learn conditional stop/move latents from physically verified teacher rollouts.

The deployed policy remains one velocity encoder and the original decoder.
Teachers are used only during collection/distillation; no runtime action override.
"""

from pathlib import Path


def main(checkpoint: Path, stop_teacher: Path, manifest: Path, out: Path, updates: int = 2000,
         stop_forward_query: float = 0., stop_yaw_query: float = 0.):
    import copy
    import json
    import time
    from dataclasses import asdict

    import torch
    import tyro

    from supermimic.config import TrainConfig
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    from supermimic.locomotion.task import measured_velocity
    from supermimic.training.loop import Trainer

    if (out / "collection" / "config.json").exists():
        raise FileExistsError(out)
    cfg = tyro.cli(TrainConfig, args=[
        "--robot", "x2_ultra", "--velocity.enabled", "--env.anchor-obs", "--run-name", "collection",
        "--log-root", str(out), "--resume", str(checkpoint), "--data.manifest", str(manifest),
        "--env.num-envs", "32", "--env.replay-capacity", "1000", "--env.action-noise-std", "0",
        "--env.episode-length-s", "25",
        "--env.no-obs-noise", "--env.no-reset-noise", "--env.no-domain-randomization",
        "--wm-trainer.compile", "None", "--policy-trainer.compile", "None",
    ])
    trainer = Trainer(cfg)
    trainer.env.cfg.auto_reset = False
    original = torch.load(checkpoint, map_location=trainer.device, weights_only=False)
    stop, _ = load_policy_from_checkpoint(stop_teacher, trainer.device)
    stop = stop.encoders["velocity"].requires_grad_(False)
    move = copy.deepcopy(trainer.policy.encoders["velocity"]).requires_grad_(False)
    student = trainer.policy.encoders["velocity"]
    decoder = trainer.policy.decoder
    decoder.requires_grad_(False)
    cmd = trainer.command
    cmd.is_velocity[:] = True
    cmd.force_standing_reset = True
    cmd.set_manual_command((0., 0., 0.))
    records, commands, trace = [], [], []
    goals = torch.tensor([[.3, 0., 0.], [.5, 0., 0.], [.3, 0., .4], [.3, 0., -.4]], device=trainer.device).repeat(8, 1)
    applied = torch.zeros_like(goals)
    ramp = torch.tensor([1., 1., 1.5], device=trainer.device) * trainer.env.step_dt

    def stopping_target(observation, goal):
        query = torch.zeros_like(goal)
        query[:, 0], query[:, 2] = stop_forward_query, stop_yaw_query
        return stop(observation, query)

    try:
        obs, _ = trainer.env.reset()
        with torch.no_grad():
            for step in range(850):
                phase = 0 if step < 100 else 1 if step < 300 else 2 if step < 450 else 3 if step < 650 else 4
                target = goals if phase == 1 else goals.roll(1, 0) if phase == 3 else torch.zeros_like(goals)
                applied = applied + (target - applied).clamp(-ramp, ramp)
                applied[applied.abs() < 1e-7] = 0
                cmd.manual_command = applied
                cmd._refresh_velocity()
                x = obs["policy"]
                zero = applied.square().sum(-1) < 1e-6
                latent = torch.where(zero[:, None], stopping_target(x, applied), move(x, applied))
                action = decoder(latent, x)
                records.append(x.clone()); commands.append(applied.clone())
                obs, _, terminated, timeout, _ = trainer.env.step(action)
                if (terminated | timeout).any():
                    failure = {"step": step + 1, "phase": phase,
                               "terminated_ids": terminated.nonzero().flatten().tolist(),
                               "timeout_ids": timeout.nonzero().flatten().tolist(), "trace": trace}
                    (out / "teacher_failure.json").write_text(json.dumps(failure, indent=2) + "\n")
                    raise RuntimeError(f"Teacher collection failed at step {step + 1}; term={failure['terminated_ids']}, timeout={failure['timeout_ids']}")
                if step % 10 == 0:
                    trace.append({"time": (step + 1) * trainer.env.step_dt, "phase": phase,
                                  "command": applied.tolist(), "velocity": measured_velocity(cmd.state_io.read()).tolist()})
        x, c = torch.stack(records), torch.stack(commands)
        # Entire environments are held out, rather than adjacent samples from a gait.
        train_x, val_x = x[:, :24].reshape(-1, x.shape[-1]), x[:, 24:].reshape(-1, x.shape[-1])
        nonzero = c[c.square().sum(-1) >= 1e-6]
        torch.save({"observations": x, "commands": c, "trace": trace}, out / "teacher_rollouts.pt")
        opt = torch.optim.Adam(student.parameters(), lr=2e-4, betas=cfg.policy_trainer.betas)
        history, started = [], time.monotonic()
        best_score, best_state, best_optimizer, best_step = float("inf"), None, None, None
        torch.manual_seed(20261028)
        val_obs = val_x[torch.randint(len(val_x), (2048,), device=trainer.device)]
        val_cmd = nonzero[torch.randint(len(nonzero), (2048,), device=trainer.device)].clone()
        val_cmd[:1024] = 0
        with torch.no_grad():
            val_target = torch.where((val_cmd.square().sum(-1) < 1e-6)[:, None],
                                     stopping_target(val_obs, val_cmd), move(val_obs, val_cmd))
        for step in range(updates):
            sample = train_x[torch.randint(len(train_x), (1024,), device=trainer.device)]
            goal = nonzero[torch.randint(len(nonzero), (1024,), device=trainer.device)].clone()
            goal[:512] = 0
            with torch.no_grad():
                target = torch.cat((stopping_target(sample[:512], goal[:512]), move(sample[512:], goal[512:])))
            prediction = student(sample, goal)
            loss = (prediction - target).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
            opt.step()
            if (step + 1) % 200 == 0:
                with torch.no_grad():
                    pred = student(val_obs, val_cmd)
                    error = (pred - val_target).square().mean(-1).sqrt()
                    action_error = (decoder(pred, val_obs) - decoder(val_target, val_obs)).square().mean(-1).sqrt()
                metrics = {"step": step + 1, "loss": float(loss), "stop_latent_rmse": float(error[:1024].mean()),
                           "move_latent_rmse": float(error[1024:].mean()),
                           "stop_action_rmse": float(action_error[:1024].mean()),
                           "move_action_rmse": float(action_error[1024:].mean())}
                history.append(metrics)
                print(metrics, flush=True)
                score = float(action_error.square().mean())
                if score < best_score:
                    best_score, best_step = score, step + 1
                    best_state, best_optimizer = copy.deepcopy(student.state_dict()), copy.deepcopy(opt.state_dict())
        student.load_state_dict(best_state)
        opt.load_state_dict(best_optimizer)
        saved = dict(original)
        saved["policy"] = trainer.policy.state_dict()
        saved["velocity_training"] = dict(original["velocity_training"])
        saved["velocity_training"]["actor_optimizer"] = opt.state_dict()
        saved["config"] = asdict(cfg)
        saved["stop_distillation"] = {"teacher": str(stop_teacher), "moving_teacher": str(checkpoint),
                                      "updates": updates, "physical_collection_steps": 850,
                                      "stop_teacher_query": [stop_forward_query, 0., stop_yaw_query],
                                      "selected_step": best_step, "selection": "held-out action MSE",
                                      "full_joint_updates_unchanged": True, "teachers_required_at_inference": False}
        torch.save(saved, out / "ckpt_final.pt")
        summary = {"updates": updates, "elapsed_s": time.monotonic() - started, "teacher_failures": 0,
                   "train_environments": 24, "heldout_environments": 8, "teacher_trajectory_seconds": 17,
                   "inference": "Original velocity encoder and decoder only; no teacher or action switch.",
                   "history": history, "teacher_trace": trace}
        (out / "training.json").write_text(json.dumps(summary, indent=2) + "\n")
    finally:
        trainer.env.close()
        trainer.writer.close()


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
