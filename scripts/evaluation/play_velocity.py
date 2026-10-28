"""Keyboard locomotion in the server simulation. Select the robot via scripts/robot.py."""

from pathlib import Path


def main(ckpt: Path, npz_dir: Path | None = None, keyboard_port: int = 8081, device: str = "cuda:0"):
    import torch
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.viewer import ViserPlayViewer

    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.motion_library import discover_npz
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    from supermimic.locomotion.config import VelocityCfg
    from supermimic.locomotion.keyboard import CommandMailbox, start_keyboard_server
    from supermimic.robots.paths import lafan_npz_dir

    policy, saved = load_policy_from_checkpoint(ckpt, device)
    if not saved.get("velocity", {}).get("enabled", False):
        raise ValueError("This checkpoint has no trained velocity interface")
    velocity = VelocityCfg(**saved["velocity"])
    files = discover_npz(npz_dir or lafan_npz_dir(), names=["walk1_subject1"])
    args = SuperMimicEnvCfgArgs(
        npz_files=[str(p) for p in files], num_envs=2, future_frames=saved["env"]["future_frames"],
        episode_length_s=1e5, action_noise_std=0., obs_noise=False, reset_noise=False,
        domain_randomization=False, record_transitions=False, anchor_obs=saved["env"]["anchor_obs"], velocity=velocity,
    )
    env = make_env(args, device=device)
    command = env.command_manager.get_term("motion")
    command.is_velocity[:] = True
    command.force_standing_reset = True
    command.set_manual_command((0., 0., 0.))
    mailbox = CommandMailbox()
    server = start_keyboard_server(mailbox, keyboard_port, min(0.5, velocity.forward[1]), min(0.5, velocity.yaw[1]))

    class Controller:
        @torch.no_grad()
        def __call__(self, observations):
            command.set_manual_command(mailbox.read(env.step_dt))
            return policy.velocity_action(observations["policy"], command.velocity_command)

    print(f"Keyboard: http://127.0.0.1:{keyboard_port}; forward viewer and keyboard ports over SSH.")
    try:
        ViserPlayViewer(RslRlVecEnvWrapper(env), Controller()).run()
    finally:
        server.shutdown()
        server.server_close()
        env.close()


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
