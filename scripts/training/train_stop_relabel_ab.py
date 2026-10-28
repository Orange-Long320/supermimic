"""Bounded actor-only A/B on one fresh replay, followed by native evaluation."""

from pathlib import Path


def main(checkpoint: Path, manifest: Path, out: Path, updates: int = 64):
    import copy
    import json
    import subprocess
    import sys
    import time
    from dataclasses import asdict, replace

    import torch
    import tyro

    from supermimic.config import TrainConfig
    from supermimic.locomotion.trainer import JointPolicyTrainer
    from supermimic.training.loop import Trainer

    if (out / "collection" / "config.json").exists() or (out / "training.json").exists():
        raise FileExistsError(out)
    cfg = tyro.cli(TrainConfig, args=[
        "--robot", "x2_ultra", "--velocity.enabled", "--env.anchor-obs",
        "--run-name", "collection", "--log-root", str(out), "--resume", str(checkpoint),
        "--data.manifest", str(manifest), "--env.num-envs", "128", "--env.replay-capacity", "384",
        "--wm-trainer.batch-size", "512", "--wm-trainer.compile", "None",
        "--policy-trainer.batch-size", "512", "--policy-trainer.compile", "None", "--policy-trainer.horizon", "32",
    ])
    trainer = Trainer(cfg)
    original = torch.load(checkpoint, map_location=trainer.device, weights_only=False)
    summary = {"input": str(checkpoint), "updates_per_arm": updates,
               "world_model_updates": 0, "decoder_updates": 0, "arms": {}}
    try:
        obs, _ = trainer.env.reset()
        with torch.no_grad():
            for _ in range(300):
                obs, _, _, _, _ = trainer.env.step(trainer.act(obs))
        buffer = trainer.buffer
        original_commands = buffer.command.clone()
        original_reference = buffer.command_reference.clone()
        for kind, fraction in (("original", 0.), ("stop_relabel", 0.5)):
            run = out / kind
            run.mkdir()
            vc = replace(cfg.velocity, stop_relabel_fraction=fraction)
            policy = copy.deepcopy(trainer.policy)
            critic = copy.deepcopy(trainer.critic)
            actor = JointPolicyTrainer(trainer.wm, policy, critic, trainer.lib, cfg.policy_trainer,
                                       vc, cfg.critic, trainer.command.cfg.nominal_height, trainer.command.reference_velocity)
            # Optimizer loading can retain same-device state tensor storage.
            # Give each arm its own moments instead of sharing the first arm's.
            actor.load_checkpoint_state(copy.deepcopy(original["velocity_training"]))
            actor.opt.load_state_dict(original["policy_opt"])
            actor.critic_opt.load_state_dict(original["critic_opt"])
            started = time.monotonic()
            history = []
            for step in range(updates):
                torch.manual_seed(7000 + step)
                metrics = actor._velocity_update(buffer)
                history.append({"step": step + 1, **metrics})
                if (step + 1) % 8 == 0:
                    print(kind, step + 1, metrics, flush=True)
            if not torch.equal(original_commands, buffer.command) or not torch.equal(original_reference, buffer.command_reference):
                raise RuntimeError("Counterfactual actor updates modified physical replay")
            if not all(torch.equal(v, original["wm"][k]) for k, v in trainer.wm.state_dict().items()):
                raise RuntimeError("World model changed in actor-only experiment")
            if not all(torch.equal(v, trainer.policy.decoder.state_dict()[k]) for k, v in policy.decoder.state_dict().items()):
                raise RuntimeError("Decoder changed in actor-only experiment")
            elapsed = time.monotonic() - started
            saved = dict(original)
            saved["policy"] = policy.state_dict()
            saved["velocity_training"] = actor.checkpoint_state()
            saved["config"] = asdict(replace(cfg, run_name=kind, velocity=vc))
            saved["stop_actor_experiment"] = {"actor_updates": updates, "wm_updates": 0,
                                              "new_collection_steps": 300, "full_joint_updates_unchanged": True}
            torch.save(saved, run / "ckpt_final.pt")
            (run / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")
            summary["arms"][kind] = {"fraction": fraction, "elapsed_s": elapsed,
                                     "checkpoint": str(run / "ckpt_final.pt"), "last_metrics": history[-1]}
            with (run / "eval_60s.log").open("w") as log:
                subprocess.run([sys.executable, "scripts/robot.py", "x2_ultra", "scripts/evaluation/eval_velocity.py",
                                "--ckpt", str(run / "ckpt_final.pt"), "--out", str(run / "eval_60s.json")],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            evaluation = json.loads((run / "eval_60s.json").read_text())
            summary["arms"][kind]["survived"] = evaluation["survived_without_reset"]
            summary["arms"][kind]["stop_speed"] = sum(evaluation["mean_stop_planar_speed_mps"]) / evaluation["num_envs"]
            print(kind, "EVALUATED", summary["arms"][kind], flush=True)
            del actor, policy, critic, saved
        (out / "training.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        trainer.env.close()
        trainer.writer.close()


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
