"""Shared differentiable task functions for physical and imagined velocity rollouts.

State angular velocity is BODY-frame, whereas yaw commands use WORLD-frame z.
Linear commands use the robot's heading frame. No motion-reference input is used.
"""

import torch

from supermimic.kinematics.rotations import (
    heading_quat,
    projected_gravity,
    quat_apply,
    quat_apply_inverse,
)
from supermimic.robots.dimensions import NQ


def measured_velocity(state):
    q = state[..., 3:7]
    v = quat_apply_inverse(heading_quat(q), state[..., NQ:NQ + 3])
    w_world = quat_apply(q, state[..., NQ + 3:NQ + 6])
    return torch.cat((v[..., :2], w_world[..., 2:3]), dim=-1)


def velocity_reward(state, command, action, previous_action, joint_limits, nominal_height, default_joints):
    measured = measured_velocity(state)
    linear = 2.0 * torch.exp(-((measured[..., :2] - command[..., :2]) / 0.5).square().sum(-1))
    yaw = 2.0 * torch.exp(-(measured[..., 2] - command[..., 2]).square() / 0.5)
    gravity = projected_gravity(state[..., 3:7])
    upright = -2.0 * gravity[..., :2].square().sum(-1)
    vertical = -0.5 * state[..., NQ + 2].square()
    roll_pitch = -0.05 * state[..., NQ + 3:NQ + 5].square().sum(-1)
    height = -2.0 * ((state[..., 2] - nominal_height) / nominal_height).square()
    smooth = -0.1 * (action - previous_action).square().sum(-1)
    joints = state[..., 7:NQ]
    limits = -2.0 * (torch.relu(joint_limits[..., 0] - joints) + torch.relu(joints - joint_limits[..., 1])).sum(-1)
    standing = command.square().sum(-1) < 1e-6
    posture = -0.5 * (joints - default_joints).square().mean(-1) * standing
    return linear + yaw + upright + vertical + roll_pitch + height + smooth + limits + posture


def fall_signals(state, nominal_height, cfg):
    height = state[..., 2]
    tilt = 1.0 + projected_gravity(state[..., 3:7])[..., 2]
    minimum = nominal_height * cfg.min_height_fraction
    hard = (height < minimum) | (tilt > cfg.max_tilt) | ~torch.isfinite(state).all(-1)
    soft_height = torch.sigmoid((minimum - height) / 0.02)
    soft_tilt = torch.sigmoid((tilt - cfg.max_tilt) / 0.04)
    return hard, 1.0 - (1.0 - soft_height) * (1.0 - soft_tilt)


def reference_commands(library, frames):
    ref = library.ref(frames)
    v = quat_apply_inverse(heading_quat(ref.root_quat_w), ref.root_lin_vel_w)
    return torch.cat((v[..., :2], ref.root_ang_vel_w[..., 2:3]), dim=-1)


@torch.no_grad()
def reference_command_table(library, window=25):
    """Centered per-clip average, preventing sway from dominating speed labels."""
    frames = torch.arange(library.num_frames, device=library.device)
    command = reference_commands(library, frames)
    prefix = torch.cat((torch.zeros_like(command[:1]), command.cumsum(0)), 0)
    clip = library.frame_clip
    lo = torch.maximum(frames - window // 2, library.clip_start[clip])
    hi = torch.minimum(frames + window // 2 + 1, library.clip_last[clip] + 1)
    return (prefix[hi] - prefix[lo]) / (hi - lo)[:, None]


def bounded_reference(table, frames, cfg):
    raw = table[frames]
    ranges = torch.tensor((cfg.forward, cfg.lateral, cfg.yaw), device=raw.device)
    projected = raw.clamp(ranges[:, 0], ranges[:, 1])
    valid = (raw - projected).abs().amax(-1) <= cfg.reference_tolerance
    return projected, valid
