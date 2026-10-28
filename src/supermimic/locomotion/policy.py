"""An additional goal encoder; the existing motion decoder remains shared."""

import torch
from torch import nn

from supermimic.policy.nets import Policy, _obs_scale_vector
from supermimic.robots.dimensions import FEAT_DIM, XI_DIM, OBS_DIM
from supermimic.world_model.dynamics import MLP


class VelocityEncoder(nn.Module):
    def __init__(self, policy_cfg, velocity_cfg):
        super().__init__()
        self.register_buffer("obs_scale", _obs_scale_vector(policy_cfg))
        self.register_buffer("command_scale", torch.tensor(velocity_cfg.command_scale))
        self.mlp = MLP(policy_cfg.history * OBS_DIM + 3, velocity_cfg.encoder_hidden, policy_cfg.latent_dim)

    def forward(self, observations, command):
        return torch.tanh(self.mlp(torch.cat((observations * self.obs_scale, command / self.command_scale), -1)))


class VelocityPolicy(Policy):
    def __init__(self, policy_cfg, velocity_cfg):
        super().__init__(policy_cfg)
        self.encoders["velocity"] = VelocityEncoder(policy_cfg, velocity_cfg)

    def velocity_action(self, observations, command):
        return self.decoder(self.encoders["velocity"](observations, command), observations)


class VelocityCritic(nn.Module):
    def __init__(self, critic_cfg, velocity_cfg):
        super().__init__()
        self.register_buffer("command_scale", torch.tensor(velocity_cfg.command_scale))
        self.mlp = MLP(FEAT_DIM + XI_DIM + 3, critic_cfg.hidden, 1)

    def forward(self, features, command, xi):
        return self.mlp(torch.cat((features, command / self.command_scale, xi), -1)).squeeze(-1)
