"""Learned near-zero-command action residual; moving core stays immutable."""
import types
import copy
import torch
from torch import nn
from supermimic.robots.dimensions import NJ

class StandingSkill(nn.Module):
    def __init__(self,obs_scale,radius=.1,max_residual=.5,mode="residual",baseline_action=None,encoder=None,decoder=None):
        super().__init__()
        self.radius=radius;self.max_residual=max_residual;self.mode=mode
        if mode not in ("residual","standalone","expert"):raise ValueError(mode)
        if mode=="standalone":
            self.register_buffer("baseline_action",torch.as_tensor(baseline_action,device=obs_scale.device,dtype=obs_scale.dtype).clone())
            if self.baseline_action.shape!=(NJ,) or not torch.isfinite(self.baseline_action).all():raise ValueError("Invalid standing action")
        self.register_buffer('obs_scale',obs_scale.clone())
        if mode=='expert':
            self.encoder=copy.deepcopy(encoder);self.decoder=copy.deepcopy(decoder)
            return
        self.net=nn.Sequential(nn.Linear(obs_scale.numel(),256),nn.Tanh(),nn.Linear(256,128),nn.Tanh(),nn.Linear(128,NJ))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,obs):
        if self.mode=='expert':
            command=torch.zeros(len(obs),3,device=obs.device,dtype=obs.dtype)
            return self.decoder(self.encoder(obs,command),obs)
        correction=self.max_residual*torch.tanh(self.net(obs*self.obs_scale))
        return self.baseline_action+correction if self.mode=="standalone" else correction
    def gate(self,command):return (1-command.square().sum(-1)/self.radius**2).clamp(0,1)

def velocity_action(self,observations,command):
    base=self.decoder(self.encoders['velocity'](observations,command),observations)
    gate=self.standing_skill.gate(command)[:,None]
    standing=self.standing_skill(observations)
    return (1-gate)*base+gate*standing if self.standing_skill.mode in ('standalone','expert') else base+gate*standing

def attach_standing_skill(policy,radius=.1,max_residual=.5,mode="residual",baseline_action=None):
    if not hasattr(policy,'velocity_action'):raise ValueError('Velocity policy required')
    policy.standing_skill=StandingSkill(policy.encoders['velocity'].obs_scale,radius,max_residual,mode,baseline_action,policy.encoders['velocity'],policy.decoder).to(next(policy.parameters()).device)
    policy.velocity_action=types.MethodType(velocity_action,policy)
    return policy
