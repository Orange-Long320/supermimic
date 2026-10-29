"""Train a private standing expert; the existing moving policy stays immutable."""
from pathlib import Path
import argparse,json,copy,hashlib,time

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['checkpoint','dataset','out']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--iterations',type=int,default=80)
 p.add_argument('--reference',type=Path,required=True)
 p.add_argument('--resume-state',action='store_true')
 p.add_argument('--sigma',type=float,default=.08)
 p.add_argument('--num-envs',type=int,default=1024)
 p.add_argument('--yaw-cost',type=float,default=40.)
 p.add_argument('--move-teacher',type=Path)
 p.add_argument('--standing-pose',type=Path)
 p.add_argument('--hours',type=float,default=1.)
 a=p.parse_args();a.out.mkdir(exist_ok=False)
 import torch
 from torch import nn
 from supermimic.eval.benchmark import load_policy_from_checkpoint
 from supermimic.envs import SuperMimicEnvCfgArgs,make_env
 from supermimic.locomotion.config import VelocityCfg
 from supermimic.locomotion.task import measured_velocity
 from supermimic.robots.dimensions import PROFILE,NQ,XI_DIM,STATE_DIM
 from supermimic.world_model.heads import analytic_observation,flatten_history_term_major
 import numpy as np
 from supermimic.kinematics.rotations import heading_yaw,projected_gravity
 torch.set_num_threads(2);torch.manual_seed(29)
 policy,saved=load_policy_from_checkpoint(a.checkpoint,'cuda:0');policy.requires_grad_(False)
 from supermimic.policy.standing import attach_standing_skill,StandingContext
 if not hasattr(policy,'standing_skill'):attach_standing_skill(policy,mode='expert')
 if policy.standing_skill.mode!='expert':raise ValueError('Standing expert required')
 actor=policy.standing_skill
 if not actor.feedback_dim:actor.add_feedback(6)
 actor.requires_grad_(False);actor.feedback_weight.requires_grad_(True);trainable=[actor.feedback_weight]
 cfg=SuperMimicEnvCfgArgs(npz_files=[str(a.dataset/'walk.npz')],num_envs=a.num_envs,seed=29,
  future_frames=saved['env']['future_frames'],episode_length_s=16.,action_noise_std=0.,obs_noise=False,
  reset_noise=False,domain_randomization=False,record_transitions=False,anchor_obs=saved['env']['anchor_obs'],velocity=VelocityCfg(**saved['velocity']))
 env=make_env(cfg,device='cuda:0');env.cfg.auto_reset=False;cmd=env.command_manager.get_term('motion')
 cmd.is_velocity[:]=True;cmd.force_standing_reset=True;cmd.set_manual_command((0.,0.,0.))
 x,_=env.reset();x=x['policy'];goal=cmd.velocity_command.clone();context=StandingContext(cmd,env)
 initial=cmd.state_io.read().clone();anchor_xy=initial[:,:2].clone();anchor_yaw=heading_yaw(initial[:,3:7]).clone();height=initial[:,2].clone()
 value=nn.Sequential(nn.Linear(x.shape[-1]+12,256),nn.Tanh(),nn.Linear(256,128),nn.Tanh(),nn.Linear(128,1)).cuda()
 opt=torch.optim.Adam(trainable,lr=3e-4);vopt=torch.optim.Adam(value.parameters(),lr=3e-4)
 reference=np.load(a.reference);qref=torch.tensor(reference['joint_pos'][0],device='cuda');refpos=torch.tensor(reference['body_pos_w'][0,0],device='cuda');refquat=torch.tensor(reference['body_quat_w'][0,0],device='cuda')
 assert len(reference['joint_pos'])>=500 and np.max(np.abs(reference['joint_vel']))<1e-6
 inv=np.load(a.reference.parent/'standing_inverse_target.npz');refstate=torch.zeros(1,STATE_DIM,device='cuda');refstate[0,:3]=refpos;refstate[0,3:7]=refquat;refstate[0,7:NQ]=qref
 act=env.action_manager.get_term('joint_pos');refaction=(torch.tensor(inv['position_control_target'],device='cuda',dtype=x.dtype)[None]-act._offset[:1])/act._scale[:1]
 xi=torch.zeros(1,XI_DIM,device='cuda');xi[:,0]=1.;refobs=analytic_observation(refstate,refaction,xi,cmd.library.default_joint_pos);refhistory=flatten_history_term_major(refobs.expand(saved['policy']['history'],-1,-1))
 height[:]=refpos[2];stop_age=torch.zeros(a.num_envs,device='cuda');was_zero=torch.ones(a.num_envs,dtype=torch.bool,device='cuda');mix_ids=torch.arange(a.num_envs,device='cuda')%4==0
 choices=torch.tensor([[.3,0,0],[-.2,0,0],[0,0,.4],[0,0,-.4],[.6,0,0]],device='cuda');move_goal=choices[(torch.arange(a.num_envs,device='cuda')//4)%5];rates=torch.tensor([1.,1.,1.5],device='cuda')*env.step_dt
 arm_ids=[i for i,n in enumerate(PROFILE['joint_names']) if any(k in n for k in ('shoulder','elbow','wrist'))]
 arm_target=qref[arm_ids].clone()
 if a.standing_pose:
  pose=json.loads(a.standing_pose.read_text())['joint_positions_rad'];arm_target=torch.tensor([pose[PROFILE['joint_names'][i]] for i in arm_ids],device='cuda')
 def value_features(xx):
  state=cmd.state_io.read()
  return torch.cat((xx*actor.obs_scale,10*measured_velocity(state),state[:,2:3],(env.episode_length_buf*env.step_dt/16)[:,None],state[:,:2]-anchor_xy,torch.atan2(torch.sin(heading_yaw(state[:,3:7])-anchor_yaw),torch.cos(heading_yaw(state[:,3:7])-anchor_yaw))[:,None],goal,(stop_age/16)[:,None]),-1)
 sigma=a.sigma;trace=[];beg=time.monotonic();total_falls=total_timeouts=0;parent_iterations=0
 if a.resume_state:
  source=torch.load(a.checkpoint,map_location='cpu',weights_only=False);assert source['standing_skill'].get('feedback_dim')==6;assert source['standing_reference']['sha256']==hashlib.sha256(a.reference.read_bytes()).hexdigest();state=source['standing_ppo_state'];assert state['sigma']==sigma
  opt.load_state_dict(state['actor_optimizer']);value.load_state_dict(state['value']);vopt.load_state_dict(state['value_optimizer']);parent_iterations=source['physical_stop']['iterations']
  print(json.dumps(dict(resumed_training_state=True,parent_iterations=parent_iterations,simulation_state='fresh original initialization')),flush=True)
 try:
  for it in range(a.iterations):
   batch=[]
   with torch.no_grad():
    for step in range(32):
     age=env.episode_length_buf*env.step_dt;desired=torch.where((mix_ids&(age<3.))[:,None],move_goal,torch.zeros_like(move_goal));goal=goal+(desired-goal).clamp(-rates,rates);cmd.manual_command=goal;cmd._refresh_velocity()
     zero=goal.square().sum(-1)<1e-6;stop_age=torch.where(zero,stop_age+env.step_dt,torch.zeros_like(stop_age));was_zero=zero.clone()
     feedback=context.features(goal);anchor_xy=context.anchor_xy;anchor_yaw=context.anchor_yaw
     oldx=x.clone();oldgoal=goal.clone();mean=policy.velocity_action(x,goal,feedback);dist=torch.distributions.Normal(mean,sigma)
     sampled=dist.sample();z=torch.where((goal.square().sum(-1)<.01)[:,None],sampled,mean);lp=dist.log_prob(z).sum(-1);vf=value_features(x);v=value(vf).squeeze(-1)
     action=z
     nxt,_,term,tout,_=env.step(action);term=term.clone();tout=tout.clone();done=term|tout
     state=cmd.state_io.read();speed=measured_velocity(state)
     dq=state[:,NQ+6:];q=state[:,7:NQ]
     arm_motion=dq[:,arm_ids].square().mean(-1)
     arm_pose=(q[:,arm_ids]-arm_target).square().mean(-1)
     yaw_error=torch.atan2(torch.sin(heading_yaw(state[:,3:7])-anchor_yaw),torch.cos(heading_yaw(state[:,3:7])-anchor_yaw))
     r=2-30*(speed[:,:2]-oldgoal[:,:2]).square().sum(-1)-a.yaw_cost*(speed[:,2]-oldgoal[:,2]).square()
     r-=(20*(state[:,:2]-anchor_xy).square().sum(-1)+10*yaw_error.square()+2*projected_gravity(state[:,3:7])[:,:2].square().sum(-1)+10*(state[:,2]-height).square())*zero
     entry=stop_age<.8
     motion_weight=torch.where(entry,.1,10.);pose_weight=torch.where(entry,10.,.5)
     r-=(motion_weight*arm_motion+pose_weight*arm_pose+2*(q-qref).square().mean(-1)+torch.where(entry,.01,.1)*dq.square().mean(-1))*zero+20*term.float()
     # Read the terminal physical state before resetting; bootstrap time limits.
     if tout.any():r+=.99*value(value_features(nxt['policy'])).squeeze(-1)*tout
     goal[done]=0;stop_age[done]=0;was_zero[done]=True
     if done.any():
      ids=done.nonzero().squeeze(-1);nxt,_=env.reset(env_ids=ids);context.reset(ids)
     batch.append((oldx,z,lp,v,r,done,oldgoal,term.clone(),tout.clone(),vf,mean.clone(),feedback.clone()));x=nxt['policy']
    tail=value(value_features(x)).squeeze(-1);adv=[];gae=torch.zeros_like(tail)
    for b in reversed(batch):
     delta=b[4]+.99*tail*(~b[5])-b[3]
     gae=delta+.99*.95*(~b[5])*gae;adv.append(gae);tail=b[3]
    adv=torch.stack(adv[::-1]);ret=(adv+torch.stack([b[3] for b in batch])).flatten()
    adv=adv.flatten();adv=(adv-adv.mean())/adv.std().clamp_min(1e-6)
    cc=torch.cat([b[6] for b in batch]);xx=torch.cat([b[0] for b in batch]);zz=torch.cat([b[1] for b in batch]);old=torch.cat([b[2] for b in batch])
   vv=torch.cat([b[9] for b in batch]);old_mean=torch.cat([b[10] for b in batch]);ff=torch.cat([b[11] for b in batch])
   actor_kls=[];accepted_steps=0
   for epoch in range(4):
    for ids in torch.randperm(len(xx),device='cuda').split(2048):
     c=cc[ids];mean=policy.velocity_action(xx[ids],c,ff[ids])
     lp=torch.distributions.Normal(mean,sigma).log_prob(zz[ids]).sum(-1)
     ratio=(lp-old[ids]).exp();eligible=(c.square().sum(-1)<.01).float();loss=-(torch.minimum(ratio*adv[ids],ratio.clamp(.8,1.2)*adv[ids])*eligible).sum()/eligible.sum().clamp_min(1)
     # The reference-trained base is frozen; learn only observable stop feedback.
     with torch.no_grad():current_kl=float((mean-old_mean[ids]).square().sum(-1).mean()/(2*sigma*sigma))
     if current_kl<.02:
      opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(trainable,.5)
      params=[t.detach().clone() for t in trainable];moments=copy.deepcopy(opt.state_dict());rate=opt.param_groups[0]['lr'];accepted=False
      for attempt in range(5):
       if attempt:
        with torch.no_grad():
         for param,previous in zip(trainable,params):param.copy_(previous)
        opt.load_state_dict(copy.deepcopy(moments))
       for group in opt.param_groups:group['lr']=rate/(2**attempt)
       opt.step()
       with torch.no_grad():
        newmean=policy.velocity_action(xx[ids],c,ff[ids])
        kl=float((newmean-old_mean[ids]).square().sum(-1).mean()/(2*sigma*sigma))
       if kl<=.03:
        accepted=True;accepted_steps+=1;actor_kls.append(kl);break
      if not accepted:
       with torch.no_grad():
        for param,previous in zip(trainable,params):param.copy_(previous)
       opt.load_state_dict(moments)
       for group in opt.param_groups:group['lr']=rate/32

     vl=(value(vv[ids]).squeeze(-1)-ret[ids]).square().mean()
     vopt.zero_grad(set_to_none=True);vl.backward();torch.nn.utils.clip_grad_norm_(value.parameters(),1.);vopt.step()
   # Recover from a one-off rollback instead of shrinking the learning rate forever.
   if actor_kls and sum(actor_kls)/len(actor_kls)<.01:
    for group in opt.param_groups:group['lr']=min(3e-4,max(1e-6,group['lr']*1.2))
   total_falls+=int(torch.stack([b[7] for b in batch]).sum());total_timeouts+=int(torch.stack([b[8] for b in batch]).sum())
   if it%5==0:
    row=dict(feedback_weight_norm=float(actor.feedback_weight.detach().norm()),total_falls=total_falls,total_timeouts=total_timeouts,cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,reference_action_rmse=float((actor(refhistory,torch.zeros(1,6,device=refhistory.device))-refaction).detach().square().mean().sqrt()),iteration=it,seconds=time.monotonic()-beg,arm_velocity_rms=float(arm_motion.mean().sqrt()),actor_kl_max=max(actor_kls,default=0),actor_lr=opt.param_groups[0]['lr'],accepted_actor_steps=accepted_steps,reward=float(torch.stack([b[4] for b in batch]).mean()),vx=float(speed[:,0].mean()),falls=int(torch.stack([b[7] for b in batch]).sum()),timeouts=int(torch.stack([b[8] for b in batch]).sum()))
    trace.append(row);print(json.dumps(row),flush=True)
   if (it+1)%20==0:
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    for k,t in policy.state_dict().items():
     if not k.startswith('standing_skill.'):assert torch.equal(t.cpu(),ck['policy'][k])
    for k,t in policy.state_dict().items():
     if k.startswith(('standing_skill.encoder.','standing_skill.decoder.')):assert torch.equal(t.cpu(),ck['policy'][k]),k
    ck['policy']={k:v.cpu() for k,v in policy.state_dict().items()};ck.pop('velocity_training',None)
    ck['standing_reference']=dict(path=str(a.reference),sha256=hashlib.sha256(a.reference.read_bytes()).hexdigest(),validation='static_feasibility_passed_active_tracking_pending',moving_core_frozen=True,transition_env_fraction=.25,episode_seconds=16)
    ck['standing_skill']=dict(radius=.1,max_residual=.5,mode='expert',feedback_dim=6)
    ck['standing_feedback']=dict(inputs=['vx_heading','vy_heading','world_yaw_rate','stop_dx_heading','stop_dy_heading','stop_heading_error'],trainable='standing_skill.feedback_weight',standing_base_frozen=True,simulation_state_feedback=True)
    ck['standing_ppo_state']=dict(actor_optimizer=opt.state_dict(),value=value.state_dict(),value_optimizer=vopt.state_dict(),sigma=sigma,privileged_value=True)
    with torch.no_grad():
     base_stand=actor.decoder(actor.encoder(x,torch.zeros(len(x),3,device=x.device)),x)
     assert torch.equal(base_stand,actor(x,torch.zeros(len(x),6,device=x.device)))
     for query in [[.3,0,0],[-.2,0,0],[0,0,.4],[0,0,-.4],[.6,0,0]]:
      cc=torch.tensor(query,device='cuda').expand(len(x),-1)
      baseline=policy.decoder(policy.encoders['velocity'](x,cc),x)
      assert torch.equal(baseline,policy.velocity_action(x,cc))
    ck['physical_stop']=dict(iterations=parent_iterations+it+1,parent_iterations=parent_iterations,source_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),trace=trace,scope='Learn only task-feedback input projection; original moving and standing base frozen; fresh physical starts; independent physical acceptance required.')
    target=a.out/f'candidate_{it+1}.pt';torch.save(ck,target);target.with_suffix('.ready').write_text(str(it+1))
   if time.monotonic()-beg>=a.hours*3600:break
  (a.out/'report.json').write_text(json.dumps(trace,indent=2)+'\n')
 finally:env.close()
if __name__=='__main__':main()
