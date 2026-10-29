"""Task-state inputs for retained standing and locomotion experts."""
import types
import copy
import torch
from torch import nn
from supermimic.robots.dimensions import NJ

class StandingSkill(nn.Module):
    def __init__(self,obs_scale,radius=.1,max_residual=.5,mode="residual",baseline_action=None,encoder=None,decoder=None,feedback_dim=0):
        super().__init__()
        self.radius=radius;self.max_residual=max_residual;self.mode=mode;self.feedback_dim=0
        if mode not in ("residual","standalone","expert"):raise ValueError(mode)
        if mode=="standalone":
            self.register_buffer("baseline_action",torch.as_tensor(baseline_action,device=obs_scale.device,dtype=obs_scale.dtype).clone())
            if self.baseline_action.shape!=(NJ,) or not torch.isfinite(self.baseline_action).all():raise ValueError("Invalid standing action")
        self.register_buffer('obs_scale',obs_scale.clone())
        if mode=='expert':
            self.encoder=copy.deepcopy(encoder);self.decoder=copy.deepcopy(decoder)
            if feedback_dim:self.add_feedback(feedback_dim)
            return
        self.net=nn.Sequential(nn.Linear(obs_scale.numel(),256),nn.Tanh(),nn.Linear(256,128),nn.Tanh(),nn.Linear(128,NJ))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def add_feedback(self,dim=6):
        if self.mode!='expert' or dim!=6 or self.feedback_dim:raise ValueError('Expected an existing expert and six new task inputs')
        self.feedback_dim=dim
        self.feedback_weight=nn.Parameter(torch.zeros(self.encoder.mlp.net[0].out_features,dim,device=self.obs_scale.device))
        self.register_buffer('feedback_scale',torch.tensor([5.,5.,2.,10.,10.,5.],device=self.obs_scale.device))
    def forward(self,obs,feedback=None):
        if self.mode=='expert':
            command=torch.zeros(len(obs),3,device=obs.device,dtype=obs.dtype)
            if not self.feedback_dim:return self.decoder(self.encoder(obs,command),obs)
            if feedback is None or feedback.shape!=(len(obs),self.feedback_dim):raise ValueError('Standing policy requires six task-state observations')
            latent=task_latent(self.encoder,obs,command,feedback,self.feedback_weight,self.feedback_scale)
            return self.decoder(latent,obs)
        correction=self.max_residual*torch.tanh(self.net(obs*self.obs_scale))
        return self.baseline_action+correction if self.mode=="standalone" else correction
    def gate(self,command):return (1-command.square().sum(-1)/self.radius**2).clamp(0,1)

def task_latent(encoder,observations,command,feedback,weight,scale):
    h=torch.cat((observations*encoder.obs_scale,command/encoder.command_scale),-1)
    for i,layer in enumerate(encoder.mlp.net):
        h=layer(h)
        if i==0:h=h+torch.nn.functional.linear(feedback*scale,weight)
    return torch.tanh(h)


def add_velocity_feedback(policy):
    if hasattr(policy,'velocity_feedback_weight'):raise ValueError('Movement feedback already attached')
    if not policy.standing_skill.feedback_dim:raise ValueError('Standing task context required')
    encoder=policy.encoders['velocity']
    policy.register_parameter('velocity_feedback_weight',nn.Parameter(torch.zeros(encoder.mlp.net[0].out_features,6,device=next(policy.parameters()).device)))


def velocity_action(self,observations,command,standing_feedback=None,disable_standing_feedback=False,disable_moving_feedback=False):
    skill=self.standing_skill;gate=skill.gate(command)[:,None]
    feedback=torch.zeros_like(standing_feedback) if disable_standing_feedback and standing_feedback is not None else standing_feedback
    if skill.feedback_dim and (gate==1).all():return skill(observations,feedback)
    if hasattr(self,'velocity_feedback_weight') and not disable_moving_feedback:
        if standing_feedback is None:raise ValueError('Movement policy requires task-state feedback')
        error=torch.cat((standing_feedback[:,:3]-command,standing_feedback[:,3:]),-1)
        latent=task_latent(self.encoders['velocity'],observations,command,error,self.velocity_feedback_weight,skill.feedback_scale)
        base=self.decoder(latent,observations)
    else:base=self.decoder(self.encoders['velocity'](observations,command),observations)
    if skill.feedback_dim and not (gate>0).any():return base
    standing=skill(observations,feedback)
    return (1-gate)*base+gate*standing if skill.mode in ('standalone','expert') else base+gate*standing

def attach_standing_skill(policy,radius=.1,max_residual=.5,mode="residual",baseline_action=None,feedback_dim=0,moving_feedback=False):
    if not hasattr(policy,'velocity_action'):raise ValueError('Velocity policy required')
    policy.standing_skill=StandingSkill(policy.encoders['velocity'].obs_scale,radius,max_residual,mode,baseline_action,policy.encoders['velocity'],policy.decoder,feedback_dim).to(next(policy.parameters()).device)
    if moving_feedback:add_velocity_feedback(policy)
    policy.velocity_action=types.MethodType(velocity_action,policy)
    return policy


class StandingContext:
    """Observable stop target and measured motion; never computes actions."""
    def __init__(self,command,env,radius=.1):
        self.command=command;self.env=env;self.radius=radius
        self.anchor_xy=self.anchor_yaw=self.previous_episode=None
    @torch.no_grad()
    def reset(self,env_ids):
        if self.anchor_xy is None:return
        from supermimic.kinematics.rotations import heading_yaw
        state=self.command.state_io.read();self.anchor_xy[env_ids]=state[env_ids,:2]
        self.anchor_yaw[env_ids]=heading_yaw(state[env_ids,3:7]);self.previous_episode[env_ids]=self.env.episode_length_buf[env_ids]
    @torch.no_grad()
    def features(self,command):
        from supermimic.kinematics.rotations import heading_yaw
        from supermimic.locomotion.task import measured_velocity
        state=self.command.state_io.read();yaw=heading_yaw(state[:,3:7]);age=self.env.episode_length_buf
        if self.anchor_xy is None:
            self.anchor_xy=state[:,:2].clone();self.anchor_yaw=yaw.clone();self.previous_episode=age.clone()
        reset=(age<self.previous_episode)|(command.square().sum(-1)>=self.radius**2)
        self.anchor_xy[reset]=state[reset,:2];self.anchor_yaw[reset]=yaw[reset];self.previous_episode=age.clone()
        delta=self.anchor_xy-state[:,:2];c=torch.cos(yaw);s=torch.sin(yaw)
        position=torch.stack((c*delta[:,0]+s*delta[:,1],-s*delta[:,0]+c*delta[:,1]),-1)
        heading=torch.atan2(torch.sin(self.anchor_yaw-yaw),torch.cos(self.anchor_yaw-yaw))[:,None]
        return torch.cat((measured_velocity(state),position,heading),-1)


def standing_action(policy,observations,command,context=None,disable_feedback=False,disable_moving_feedback=False):
    skill=getattr(policy,'standing_skill',None)
    if skill is not None and skill.feedback_dim:
        if context is None:raise ValueError('No standing task-state context supplied')
        feedback=context.features(command)
        return policy.velocity_action(observations,command,feedback,disable_standing_feedback=disable_feedback,disable_moving_feedback=disable_moving_feedback)
    return policy.velocity_action(observations,command)
