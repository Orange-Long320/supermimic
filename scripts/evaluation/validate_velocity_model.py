"""Collect fresh replay and report reference coverage plus 1/8/32-step WM errors."""

from pathlib import Path


def main(checkpoint: Path, manifest: Path, out: Path, collection_steps: int = 300, seed: int = 0):
    import json

    import torch
    import tyro

    from supermimic.config import TrainConfig
    from supermimic.locomotion.validation import validate_velocity_model
    from supermimic.training.loop import Trainer

    if out.exists():
        raise FileExistsError(out)
    if collection_steps < 34:
        raise ValueError("Need at least 34 collection steps for a 32-step check")
    cfg = tyro.cli(TrainConfig, args=[
        "--robot", "x2_ultra", "--velocity.enabled", "--env.anchor-obs", "--seed", str(seed),
        "--run-name", out.parent.name + "_validation", "--resume", str(checkpoint),
        "--data.manifest", str(manifest), "--env.num-envs", "128",
        "--env.replay-capacity", str(collection_steps + 64),
        "--wm-trainer.batch-size", "128", "--wm-trainer.compile", "None",
        "--policy-trainer.batch-size", "128", "--policy-trainer.horizon", "32",
        "--policy-trainer.compile", "None",
    ])
    trainer = Trainer(cfg)
    buffer = trainer.buffer
    fractions = []
    try:
        obs, _ = trainer.env.reset()
        with torch.no_grad():
            for step in range(collection_steps):
                obs, _, _, _, _ = trainer.env.step(trainer.act(obs))
                if step % 10 == 0:
                    mask = trainer.command.is_velocity
                    fractions.append({"step": step + 1, "reference_fraction":
                                      float(trainer.command.command_from_reference[mask].float().mean())})
        valid = buffer.valid & ~buffer.terminal & buffer.velocity_env_mask[:, None]
        reference = valid & buffer.command_reference
        report = {
            "checkpoint": str(checkpoint.resolve()), "seed": seed,
            "collection_steps": collection_steps, "no_optimizer_updates": True,
            "replay_stats": {"velocity_rows": int(valid.sum()), "reference_rows": int(reference.sum()),
                             "reference_fraction": float(reference.sum() / valid.sum()),
                             "mean_command": buffer.command[valid].mean(0).tolist(),
                             "zero_command_fraction": float((buffer.command[valid].norm(dim=-1) < 1e-5).float().mean())},
            "reference_fraction_over_time": fractions,
            "world_model": validate_velocity_model(trainer.wm, buffer, trainer.command.cfg.nominal_height,
                                                    trainer.lib.default_joint_pos, trainer.env.step_dt),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        trainer.env.close()
        if trainer.writer is not None:
            trainer.writer.close()


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
