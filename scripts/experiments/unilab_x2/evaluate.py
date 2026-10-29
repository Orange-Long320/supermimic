"""Headless single-clip FlashSAC evaluation through official UniLab APIs.

One deterministic reset at clip start; autoreset is disabled. Stop on the first
failure or timeout, retaining the pre-reset terminal state. Execute on server.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--unilab-root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--backend", choices=["mujoco", "mjwarp"])
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    if not a.checkpoint.is_file():
        raise FileNotFoundError(a.checkpoint)
    root = a.unilab_root.resolve()
    sys.path.insert(0, str(root / "src"))
    package_dir = str(Path(__file__).resolve().parent)
    os.environ["PYTHONPATH"] = os.pathsep.join([package_dir, str(root / "src")])
    os.environ["UNILAB_EXTRA_REGISTRY_PACKAGES"] = "x2_unilab"
    import numpy as np
    import torch
    from omegaconf import OmegaConf, open_dict
    from unilab.base.config_adapter import EnvConfigAdapter, create_env
    from unilab.base.process_device import (
        apply_backend_env_device_override,
        apply_collector_tensor_device_override,
        configure_backend_process_device,
    )
    from unilab.training.common import ensure_registries
    from unilab.utils.sim2sim import resolve_sim2sim_config
    from unilab.visualization.interactive_playback import (
        build_play_actor, extract_play_obs, flashsac_play_actor_semantics,
        load_play_actor, resolve_play_obs_dims,
    )

    cfg = OmegaConf.load(a.config)
    if str(cfg.algo.algo) != "flashsac":
        raise ValueError("This evaluator supports the official FlashSAC checkpoint only")
    # Validate the supplied source config against the checkpoint's run before
    # deliberately changing the evaluation reset/noise protocol below.
    resolve_sim2sim_config(a.checkpoint.parent, cfg, algo_name="flashsac", strict=True)
    source_motion = cfg.env.commands.motion.params.motion_file
    if not isinstance(source_motion, str):
        raise ValueError("Evaluate one explicitly selected clip at a time")
    with np.load(source_motion, allow_pickle=False) as clip:
        clip_frames = len(clip["joint_pos"])
        fps = float(np.asarray(clip["fps"]).reshape(-1)[0])
    clip_duration = (clip_frames - 1) / fps
    with open_dict(cfg):
        if a.backend:
            cfg.training.sim_backend = a.backend
        cfg.env.seed = a.seed
        cfg.env.auto_reset = False
        cfg.env.max_episode_seconds = max(float(cfg.env.max_episode_seconds), clip_duration + 1.0)
        cfg.env.observations.actor.enable_corruption = False
        command = cfg.env.commands.motion.params
        # Tensor MotionCommand supports adaptive/mixed only. mixed + start=1
        # is its supported deterministic-start path; mode='start' would fail.
        command.sampling_mode = "mixed"
        command.sampling_start_ratio = 1.0
        command.pose_range = {}
        command.velocity_range = {}
        command.joint_position_range = [0.0, 0.0]
        command.joint_default_position_range = [0.0, 0.0]
    backend = str(cfg.training.sim_backend)
    torch.manual_seed(a.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    selected = configure_backend_process_device(backend, device)
    device = selected or device
    ensure_registries()
    overrides = EnvConfigAdapter(cfg, root_dir=root, algo_name="flashsac").build_task_env_cfg_override()
    overrides = apply_backend_env_device_override(overrides, backend, learner_device=device)
    overrides = apply_collector_tensor_device_override(
        overrides, backend, collector_tensor_device="auto", learner_device=device)
    a.out.mkdir(parents=True)
    OmegaConf.save(cfg, a.out / "evaluation-config.yaml", resolve=True)
    env = create_env(cfg, num_envs=1, env_cfg_override=overrides)
    records = []
    started = time.monotonic()
    try:
        env.init_state()
        obs, _ = env.reset(seed=a.seed)
        motion = env.command_manager.get_term("motion")
        if motion.motion.num_clips != 1:
            raise ValueError("Evaluation requires exactly one clip")
        initial_frame = int(motion.time_steps[0].item())
        if initial_frame != 0:
            raise ValueError(f"Clip-start reset failed: initial_frame={initial_frame}")
        obs_dim, critic_obs_dim = resolve_play_obs_dims(env.obs_groups_spec)
        action_dim = int(env.action_space.shape[0])
        actor, normalizer, _, _ = build_play_actor(
            "flashsac", cfg, obs_dim=obs_dim, critic_obs_dim=critic_obs_dim,
            action_dim=action_dim, device=device)
        checkpoint = torch.load(a.checkpoint, map_location=device, weights_only=True)
        load_play_actor("flashsac", actor, normalizer, checkpoint,
                        flashsac_play_actor_semantics(cfg, obs_dim=obs_dim,
                                                     action_dim=action_dim, actor=actor))
        dt = float(cfg.env.ctrl_dt)
        # Two extra control ticks allow the command's terminal boundary to
        # publish after reaching the last frame, without rolling past failure.
        budget = math.ceil(clip_duration / dt) + 2
        failure = None
        terminal_names = []
        last_frame = initial_frame
        with torch.inference_mode():
            for step in range(1, budget + 1):
                actor_obs = extract_play_obs(obs).to(device=device, dtype=torch.float32)
                if normalizer is not None:
                    actor_obs = normalizer(actor_obs, update=False)
                action = actor.explore(actor_obs, deterministic=True)
                if not bool(torch.isfinite(action).all()):
                    raise ValueError("Nonfinite actor output")
                state = env.step(action.to(device=env.device, dtype=torch.float32).contiguous())
                obs = state.obs
                last_frame = int(motion.time_steps[0].item())
                failures = []
                timeouts = []
                for name in env.termination_manager.active_terms:
                    if bool(env.termination_manager.get_term(name)[0].item()):
                        target = timeouts if env.termination_manager.get_term_cfg(name).time_out else failures
                        target.append(name)
                components = dict(env.reward_manager.get_active_iterable_terms(0))
                tracking_rate = sum(float(values[0]) for name, values in components.items()
                                    if name.startswith("motion_"))
                metrics = {name: float(value[0].item()) for name, value in motion.metrics.items()
                           if name.startswith("error_")}
                row = dict(step=step, elapsed_s=step * dt, frame=last_frame,
                           reward=float(state.reward[0].item()), tracking_reward_rate=tracking_rate,
                           errors=metrics, failure_terms=failures, timeout_terms=timeouts)
                if not all(math.isfinite(v) for v in [row["reward"], tracking_rate, *metrics.values()]):
                    raise ValueError("Nonfinite evaluation metric")
                records.append(row)
                if failures or bool(state.terminated[0].item()):
                    failure = dict(step=step, elapsed_s=step * dt, frame=last_frame, terms=failures)
                    terminal_names = failures + timeouts
                    break
                if timeouts or bool(state.truncated[0].item()):
                    terminal_names = timeouts
                    break
        end_frame = int(motion.current_clip_end_frames[0].item())
        completed = failure is None and last_frame >= end_frame
        error_names = records[0]["errors"] if records else {}
        result = dict(status="completed", backend=backend, seed=a.seed,
                      checkpoint=str(a.checkpoint.resolve()), checkpoint_sha256=sha256(a.checkpoint),
                      source_config=str(a.config.resolve()), source_config_sha256=sha256(a.config),
                      motion_file=source_motion, motion_sha256=sha256(Path(source_motion)),
                      clip_frames=clip_frames, fps=fps, clip_duration_s=clip_duration,
                      initial_frame=initial_frame, final_frame=last_frame, clip_end_frame=end_frame,
                      executed_steps=len(records), executed_s=len(records) * dt,
                      first_non_timeout_failure=failure, terminal_terms=terminal_names,
                      complete_clip_without_non_timeout_failure=completed,
                      mean_tracking_reward_rate=float(np.mean([r["tracking_reward_rate"] for r in records])) if records else None,
                      mean_total_reward_per_step=float(np.mean([r["reward"] for r in records])) if records else None,
                      mean_errors={name: float(np.mean([r["errors"][name] for r in records]))
                                   for name in error_names},
                      wall_seconds=time.monotonic() - started,
                      autoreset=False, resets_after_start=0, deterministic_actor=True,
                      source_training_contract_validated=True,
                      evaluation_protocol_overrides=[
                          "mixed sampler with start_ratio=1 (training sampling mode may differ)",
                          "observation corruption disabled", "reset perturbations disabled",
                          "autoreset disabled", "episode timeout extended if needed to cover full clip"],
                      caveat="Task termination/tracking assessment only; not the project's strict physical acceptance or a convergence claim")
        (a.out / "steps.json").write_text(json.dumps(records, indent=2) + "\n")
        (a.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
