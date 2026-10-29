"""Learned near-zero-command action residual; moving core stays immutable."""
import types
import torch
from torch import nn
from supermimic.robots.dimensions import NJ

class StandingSkill(nn.Module):
    def __init__(self,obs_scale,radius=.1,max_residual=.5):
        super().__init__()
        self.radius=radius;self.max_residual=max_residual
        self.register_buffer('obs_scale',obs_scale.clone())
        self.net=nn.Sequential(nn.Linear(obs_scale.numel(),256),nn.Tanh(),nn.Linear(256,128),nn.Tanh(),nn.Linear(128,NJ))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,obs):return self.max_residual*torch.tanh(self.net(obs*self.obs_scale))
    def gate(self,command):return (1-command.square().sum(-1)/self.radius**2).clamp(0,1)

def velocity_action(self,observations,command):
    base=self.decoder(self.encoders['velocity'](observations,command),observations)
    return base+self.standing_skill.gate(command)[:,None]*self.standing_skill(observations)

def attach_standing_skill(policy,radius=.1,max_residual=.5):
    if not hasattr(policy,'velocity_action'):raise ValueError('Velocity policy required')
    policy.standing_skill=StandingSkill(policy.encoders['velocity'].obs_scale,radius,max_residual).to(next(policy.parameters()).device)
    policy.velocity_action=types.MethodType(velocity_action,policy)
    return policy
