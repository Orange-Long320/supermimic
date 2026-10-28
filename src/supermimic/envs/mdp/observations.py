from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from supermimic.envs.mdp.commands import MultiClipMotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def goal_future_frames(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """(N, K*68) future reference-frame features (plan D5)."""
    command = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return command.goal


def fixed_box_goal(env: ManagerBasedRlEnv, box) -> torch.Tensor:
    from supermimic.envs.box_scene import box_features

    qpos = env.sim.data.qpos[:]
    return box_features(qpos[:, :3] - env.scene.env_origins, qpos[:, 3:7], box)
