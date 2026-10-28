"""Record one uninterrupted physical velocity-policy rollout on the server."""

from pathlib import Path


def main(ckpt: Path, out: Path, device: str = "cuda:0", seed: int = 0):
    import datetime
    import hashlib
    import json
    import shutil
    import subprocess

    import mujoco
    import numpy as np
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from PIL import Image, ImageDraw, ImageFont

    from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg
    from supermimic.envs.mdp.motion_library import discover_npz
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    from supermimic.locomotion.config import VelocityCfg
    from supermimic.locomotion.keyboard import CommandMailbox
    from supermimic.locomotion.task import measured_velocity
    from supermimic.robots.paths import lafan_npz_dir

    if out.exists() or out.with_suffix(".json").exists():
        raise FileExistsError(out)
    before = ckpt.stat()
    with ckpt.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    policy, saved = load_policy_from_checkpoint(ckpt, device)
    after = ckpt.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError("Checkpoint changed while loading; select an immutable checkpoint")
    if not saved.get("velocity", {}).get("enabled", False):
        raise ValueError("This checkpoint has no velocity interface")
    velocity_cfg = VelocityCfg(**saved["velocity"])
    files = discover_npz(lafan_npz_dir(), names=["walk1_subject1"])
    args = SuperMimicEnvCfgArgs(
        npz_files=[str(p) for p in files], num_envs=2, seed=seed,
        future_frames=saved["env"]["future_frames"], episode_length_s=30.,
        action_noise_std=0., obs_noise=False, reset_noise=False,
        domain_randomization=False, record_transitions=False,
        anchor_obs=saved["env"].get("anchor_obs", False), velocity=velocity_cfg,
    )
    config = make_env_cfg(args)
    config.auto_reset = False
    env = ManagerBasedRlEnv(config, device=device)
    command = env.command_manager.get_term("motion")
    command.is_velocity[:] = True
    command.force_standing_reset = True
    command.set_manual_command((0., 0., 0.))
    schedule = [
        (0., "STAND", (0., 0., 0.)),
        (2., "FORWARD", (0.3, 0., 0.)),
        (9., "TURN LEFT", (0.3, 0., 0.4)),
        (16., "STOP", (0., 0., 0.)),
    ]
    duration, width, height, fps = 24., 1280, 720, 25
    model = env.sim.mj_model
    model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(height, model.vis.global_.offheight)
    physical = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.azimuth, camera.elevation = 3.3, 135., -18.
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 25)
    small_font = ImageFont.truetype(font_path, 19)
    out.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    writer = subprocess.Popen([
        ffmpeg, "-n", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(out),
    ], stdin=subprocess.PIPE)
    mailbox = CommandMailbox()
    records, trail = [], []
    failed_at, frames = None, 0
    end_reason = "requested_duration"
    try:
        observations, _ = env.reset()
        with torch.no_grad():
            for step in range(round(duration / env.step_dt)):
                elapsed = step * env.step_dt
                _, phase, target = next(item for item in reversed(schedule) if elapsed >= item[0])
                mailbox.put(target, now=elapsed)
                command.set_manual_command(mailbox.read(env.step_dt, now=elapsed))
                used = command.velocity_command.clone()
                action = policy.velocity_action(observations["policy"], used)
                if not torch.isfinite(action).all():
                    end_reason = "nonfinite_action"
                    break
                observations, _, terminated, timeout, _ = env.step(action)
                if not torch.isfinite(env.sim.data.qpos[:]).all():
                    end_reason = "nonfinite_state"
                    break
                measured = measured_velocity(command.state_io.read())[0].cpu().numpy()
                if bool(terminated[0] | timeout[0]):
                    failed_at = (step + 1) * env.step_dt
                    end_reason = "physical_failure" if bool(terminated[0]) else "timeout"
                if step % 2 == 0:
                    physical.qpos[:] = env.sim.data.qpos[:].cpu().numpy()[0]
                    physical.qvel[:] = env.sim.data.qvel[:].cpu().numpy()[0]
                    mujoco.mj_forward(model, physical)
                    camera.lookat[:] = physical.qpos[:3] + np.array([0., 0., 0.15])
                    renderer.update_scene(physical, camera=camera)
                    pixels = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(pixels)
                    draw.rounded_rectangle((18, 16, 900, 173), radius=12, fill=(18, 26, 31))
                    draw.text((35, 27), f"X2 Ultra | checkpoint {int(ckpt.stem.rsplit('_', 1)[-1])} | {elapsed + env.step_dt:05.2f}s | {phase}", font=font, fill=(235, 241, 245))
                    sent = used[0].cpu().tolist()
                    draw.text((35, 69), f"COMMAND  vx {sent[0]:+.2f} m/s    yaw {sent[2]:+.2f} rad/s", font=font, fill=(113, 222, 177))
                    draw.text((35, 108), f"ACTUAL      vx {measured[0]:+.2f} m/s    yaw {measured[2]:+.2f} rad/s", font=font, fill=(255, 208, 132))
                    draw.text((35, 146), "Continuous physical rollout | no resets | 1x speed", font=small_font, fill=(193, 205, 215))
                    # A map with metre grid makes residual drift during STOP visible
                    # even though the 3D camera follows the physical robot.
                    trail.append(physical.qpos[:2].copy())
                    center = np.array(trail[-1])
                    draw.rectangle((1030, 18, 1260, 248), fill=(20, 29, 34), outline=(110, 135, 146))
                    draw.text((1042, 26), "PATH | 1 m grid", font=small_font, fill=(218, 228, 235))
                    scale, origin = 24., np.array([1145., 146.])
                    for offset in range(-4, 5):
                        draw.line((1040, 146 + offset * 24, 1250, 146 + offset * 24), fill=(40, 55, 63))
                        draw.line((1145 + offset * 24, 65, 1145 + offset * 24, 238), fill=(40, 55, 63))
                    mapped = [origin + np.array([p[0] - center[0], -(p[1] - center[1])]) * scale for p in trail]
                    visible = [tuple(p) for p in mapped if 1040 <= p[0] <= 1250 and 65 <= p[1] <= 238]
                    if len(visible) > 1:
                        draw.line(visible, fill=(113, 222, 177), width=3)
                    draw.ellipse((1140, 141, 1150, 151), fill=(255, 208, 132))
                    for start, label, _ in schedule:
                        x1 = int(20 + start / duration * 1240)
                        next_start = next((s[0] for s in schedule if s[0] > start), duration)
                        x2 = int(20 + next_start / duration * 1240)
                        draw.rectangle((x1, 660, x2 - 2, 707), fill=(44, 112, 82) if label == phase else (29, 42, 49))
                        draw.text((x1 + 8, 669), label, font=small_font, fill=(235, 241, 245))
                    if failed_at is not None:
                        draw.text((35, 205), f"{end_reason.upper()} at {failed_at:.2f}s - recording stopped", font=font, fill=(255, 90, 90))
                    writer.stdin.write(pixels.tobytes())
                    frames += 1
                    records.append({"time_s": (step + 1) * env.step_dt, "phase": phase, "command": sent, "velocity": measured.tolist(), "root_position": physical.qpos[:3].tolist()})
                if failed_at is not None:
                    break
    finally:
        writer.stdin.close()
        return_code = writer.wait()
        renderer.close()
        env.close()
    if return_code:
        raise RuntimeError(f"Video encoding failed: {return_code}")
    metadata = {
        "checkpoint": str(ckpt.resolve()), "checkpoint_sha256": digest,
        "rendered_utc": datetime.datetime.now(datetime.UTC).isoformat(), "seed": seed,
        "schedule": schedule, "requested_seconds": duration, "video_frames": frames, "fps": fps,
        "video_seconds": frames / fps, "end_reason": end_reason, "first_failure_s": failed_at,
        "resets": 0, "recorded_environment": 0, "num_simulated_envs": 2,
        "note": "One unselected deterministic trial; closed-loop policy physics, no reference pose input to velocity actor.",
        "samples": records,
    }
    out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({key: value for key, value in metadata.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
