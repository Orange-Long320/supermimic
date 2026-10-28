"""Joint motion/velocity environment built on the pinned tracking environment."""

from dataclasses import dataclass, field, fields, replace

import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from supermimic.envs.mdp.commands import MultiClipMotionCommand, MultiClipMotionCommandCfg
from supermimic.envs.state_io import StateIO
from supermimic.locomotion.config import VelocityCfg
from supermimic.locomotion.task import bounded_reference, fall_signals, measured_velocity, reference_command_table, velocity_reward


class VelocityMotionCommand(MultiClipMotionCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.state_io = StateIO(env)
        count = max(1, min(self.num_envs - 1, round(self.num_envs * cfg.velocity.fraction)))
        self.is_velocity = torch.arange(self.num_envs, device=self.device) >= self.num_envs - count
        self.velocity_command = torch.zeros(self.num_envs, 3, device=self.device)
        self.command_from_reference = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.command_age = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.command_period = max(1, round(cfg.velocity.resample_seconds / env.step_dt))
        self.manual_command = None
        self.force_standing_reset = False
        self.reference_velocity = reference_command_table(self.library)
        self.metrics["velocity_error"] = torch.zeros(self.num_envs, device=self.device)

    def _bounded_reference(self):
        return bounded_reference(self.reference_velocity, self.frame, self.cfg.velocity)

    def _sample_velocity(self, ids):
        if ids.numel() == 0:
            return
        cfg = self.cfg.velocity
        ranges = torch.tensor((cfg.forward, cfg.lateral, cfg.yaw), device=self.device)
        random = torch.rand(len(ids), 3, device=self.device)
        self.velocity_command[ids] = ranges[:, 0] + random * (ranges[:, 1] - ranges[:, 0])
        stand = torch.rand(len(ids), device=self.device) < cfg.standing_probability
        self.velocity_command[ids[stand]] = 0
        self.command_from_reference[ids] = (~stand) & (torch.rand(len(ids), device=self.device) < cfg.reference_probability)
        self.command_age[ids] = 0
        self._refresh_velocity()

    def _refresh_velocity(self):
        reference, valid = self._bounded_reference()
        # Only small command projections (within the explicit tolerance) may
        # supervise the latent; larger command/reference mismatches are rejected.
        self.command_from_reference &= valid & self.is_velocity
        self.velocity_command = torch.where(self.command_from_reference[:, None], reference, self.velocity_command)
        if self.manual_command is not None:
            self.velocity_command[self.is_velocity] = self.manual_command
            self.command_from_reference[self.is_velocity] = False

    def set_manual_command(self, command):
        value = torch.as_tensor(command, device=self.device, dtype=torch.float32)
        ranges = torch.tensor((self.cfg.velocity.forward, self.cfg.velocity.lateral, self.cfg.velocity.yaw), device=self.device)
        if value.shape != (3,) or not torch.isfinite(value).all():
            raise ValueError("Manual command needs three finite values")
        self.manual_command = value.clamp(ranges[:, 0], ranges[:, 1])
        self._refresh_velocity()

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        self._sample_velocity(env_ids[self.is_velocity[env_ids]])

    def _write_reference_state_to_sim(self, env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel):
        from supermimic.kinematics.rotations import heading_quat

        standing = self.is_velocity[env_ids] & ((torch.rand(len(env_ids), device=self.device) < self.cfg.velocity.standing_probability) | self.force_standing_reset)
        if standing.any():
            root_pos = root_pos.clone()
            root_ori = root_ori.clone()
            root_lin_vel = root_lin_vel.clone()
            root_ang_vel = root_ang_vel.clone()
            joint_pos = joint_pos.clone()
            joint_vel = joint_vel.clone()
            root_pos[standing, 2] = self.cfg.nominal_height + self._env.scene.env_origins[env_ids[standing], 2]
            root_ori[standing] = heading_quat(root_ori[standing])
            root_lin_vel[standing] = 0
            root_ang_vel[standing] = 0
            joint_pos[standing] = self.robot.data.default_joint_pos[env_ids[standing]]
            joint_vel[standing] = 0
        super()._write_reference_state_to_sim(env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel)

    def _finalize_episodes(self, env_ids):
        # Only tracking returns update the tracking curriculum.
        tracking = env_ids[~self.is_velocity[env_ids]]
        if tracking.numel():
            super()._finalize_episodes(tracking)
        self.ep_len[env_ids] = 0

    def _update_command(self, env_ids=None):
        super()._update_command(env_ids)
        ended = self.clip_end & self.is_velocity
        self.command_from_reference[ended] = False
        # A velocity episode continues physically after its reference clip ends.
        self.clip_end[self.is_velocity] = False
        if env_ids is None:
            reset = self._env.reset_buf
            self.command_age[self.is_velocity & ~reset] += 1
            ids = (self.is_velocity & (self.command_age >= self.command_period)).nonzero().squeeze(-1)
            self._sample_velocity(ids)
        self._refresh_velocity()

    def _update_metrics(self):
        super()._update_metrics()
        self.metrics["velocity_error"] = (measured_velocity(self.state_io.read()) - self.velocity_command).norm(dim=-1)


@dataclass(kw_only=True)
class VelocityMotionCommandCfg(MultiClipMotionCommandCfg):
    velocity: VelocityCfg = field(default_factory=VelocityCfg)
    nominal_height: float = 1.0

    def build(self, env):
        return VelocityMotionCommand(self, env)


def motion_reward(env, reward_func, reward_params):
    cmd = env.command_manager.get_term("motion")
    reward = reward_func(env, **reward_params)
    return torch.where(cmd.is_velocity, 0.0, reward)


def velocity_task_reward(env):
    cmd = env.command_manager.get_term("motion")
    robot = cmd.robot
    value = velocity_reward(
        cmd.state_io.read(), cmd.velocity_command, env.action_manager.action,
        env.action_manager.prev_action, robot.data.soft_joint_pos_limits,
        cmd.cfg.nominal_height, robot.data.default_joint_pos,
    )
    return torch.where(cmd.is_velocity, value, 0.0)


def interface_termination(env, original_func, original_params, fall=False):
    cmd = env.command_manager.get_term("motion")
    tracking = original_func(env, **original_params)
    velocity = fall_signals(cmd.state_io.read(), cmd.cfg.nominal_height, cmd.cfg.velocity)[0] if fall else torch.zeros_like(tracking)
    return torch.where(cmd.is_velocity, velocity, tracking)


def configure_velocity_env(env_cfg, velocity_cfg, nominal_height):
    """Customize a tracking configuration without modifying the baseline factory."""
    velocity_cfg.validate()
    if env_cfg.scene.num_envs < 2:
        raise ValueError("Joint velocity training requires at least two environments")
    original = env_cfg.commands["motion"]
    values = {f.name: getattr(original, f.name) for f in fields(original) if f.init}
    cmd_cfg = VelocityMotionCommandCfg(**values, velocity=velocity_cfg, nominal_height=nominal_height)
    env_cfg.commands["motion"] = cmd_cfg
    for name, term in list(env_cfg.rewards.items()):
        env_cfg.rewards[name] = replace(term, func=motion_reward, params={"reward_func": term.func, "reward_params": term.params})
    env_cfg.rewards["velocity_task"] = RewardTermCfg(func=velocity_task_reward, weight=1.0)
    for name in ("root_height", "root_tilt", "clip_end"):
        term = env_cfg.terminations[name]
        env_cfg.terminations[name] = TerminationTermCfg(
            func=interface_termination,
            params={"original_func": term.func, "original_params": term.params, "fall": name == "root_height"},
            time_out=term.time_out,
        )
    if "replay" in env_cfg.recorders:
        env_cfg.recorders["replay"].params["velocity"] = True
    return env_cfg
