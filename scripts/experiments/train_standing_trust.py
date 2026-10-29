"""Learn quiet zero-command posture with physical joint-motion costs and move distillation."""
from pathlib import Path
import argparse,json,copy,hashlib,time

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['checkpoint','dataset','out']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--iterations',type=int,default=80)
 p.add_argument('--yaw-cost',type=float,default=40.)
 p.add_argument('--move-teacher',type=Path)
 p.add_argument('--standing-pose',type=Path)
 p.add_argument('--conditional',action='store_true')
 a=p.parse_args();a.out.mkdir(exist_ok=False)
 import torch
 from torch import nn
 from supermimic.eval.benchmark import load_policy_from_checkpoint
 from supermimic.envs import SuperMimicEnvCfgArgs,make_env
 from supermimic.locomotion.config import VelocityCfg
 from supermimic.locomotion.task import measured_velocity
 from supermimic.robots.dimensions import PROFILE,NQ
 torch.set_num_threads(2);torch.manual_seed(29)
 policy,saved=load_policy_from_checkpoint(a.checkpoint,'cuda:0');policy.requires_grad_(False)
 from supermimic.policy.standing import attach_standing_skill
 attach_standing_skill(policy);actor=policy.standing_skill;actor.requires_grad_(True)
 cfg=SuperMimicEnvCfgArgs(npz_files=[str(a.dataset/'backward.npz')],num_envs=256,seed=29,
  future_frames=saved['env']['future_frames'],episode_length_s=3.,action_noise_std=0.,obs_noise=False,
  reset_noise=False,domain_randomization=False,record_transitions=False,anchor_obs=saved['env']['anchor_obs'],velocity=VelocityCfg(**saved['velocity']))
 env=make_env(cfg,device='cuda:0');cmd=env.command_manager.get_term('motion')
 cmd.is_velocity[:]=True;cmd.force_standing_reset=True;cmd.set_manual_command((0.,0.,0.))
 x,_=env.reset();x=x['policy'];goal=cmd.velocity_command.clone()
 value=nn.Sequential(nn.Linear(x.shape[-1],256),nn.Tanh(),nn.Linear(256,128),nn.Tanh(),nn.Linear(128,1)).cuda()
 opt=torch.optim.Adam(actor.parameters(),lr=3e-5);vopt=torch.optim.Adam(value.parameters(),lr=3e-4)
 arm_ids=[i for i,n in enumerate(PROFILE['joint_names']) if any(k in n for k in ('shoulder','elbow','wrist'))]
 arm_target=cmd.library.default_joint_pos[arm_ids].clone()
 if a.standing_pose:
  pose=json.loads(a.standing_pose.read_text())['joint_positions_rad'];arm_target=torch.tensor([pose[PROFILE['joint_names'][i]] for i in arm_ids],device='cuda')
 sigma=.015;trace=[];beg=time.monotonic()
 try:
  for it in range(a.iterations):
   if a.conditional and it % 2 == 0:
    choices=torch.tensor([-.2,0.,.3],device='cuda')
    goal[:,0]=choices[torch.randint(3,(256,),device='cuda')]
    cmd.manual_command=goal
    cmd._refresh_velocity()
   batch=[]
   with torch.no_grad():
    for step in range(32):
     mean=policy.velocity_action(x,goal);dist=torch.distributions.Normal(mean,sigma)
     z=dist.sample();lp=dist.log_prob(z).sum(-1);v=value(x).squeeze(-1)
     action=z
     nxt,_,term,tout,_=env.step(action);done=term|tout
     state=cmd.state_io.read();speed=measured_velocity(state)
     dq=state[:,NQ+6:];q=state[:,7:NQ]
     arm_motion=dq[:,arm_ids].square().mean(-1)
     arm_pose=(q[:,arm_ids]-arm_target).square().mean(-1)
     r=2-30*speed[:,:2].square().sum(-1)-a.yaw_cost*speed[:,2].square()
     entry=env.episode_length_buf*env.step_dt<.8
     motion_weight=torch.where(entry,.1,10.);pose_weight=torch.where(entry,10.,.5)
     r-=motion_weight*arm_motion+pose_weight*arm_pose+torch.where(entry,.01,.1)*dq.square().mean(-1)+20*term.float()
     batch.append((x.clone(),z,lp,v,r,done,goal.clone(),term.clone(),tout.clone()));x=nxt['policy']
    tail=value(x).squeeze(-1);adv=[];gae=torch.zeros_like(tail)
    for b in reversed(batch):
     delta=b[4]+.99*tail*(~b[5])-b[3]
     gae=delta+.99*.95*(~b[5])*gae;adv.append(gae);tail=b[3]
    adv=torch.stack(adv[::-1]);ret=(adv+torch.stack([b[3] for b in batch])).flatten()
    adv=adv.flatten();adv=(adv-adv.mean())/adv.std().clamp_min(1e-6)
    cc=torch.cat([b[6] for b in batch]);xx=torch.cat([b[0] for b in batch]);zz=torch.cat([b[1] for b in batch]);old=torch.cat([b[2] for b in batch])
   actor_kls=[];accepted_steps=0
   for epoch in range(4):
    for ids in torch.randperm(len(xx),device='cuda').split(1024):
     c=cc[ids];mean=policy.velocity_action(xx[ids],c)
     lp=torch.distributions.Normal(mean,sigma).log_prob(zz[ids]).sum(-1)
     ratio=(lp-old[ids]).exp();loss=-torch.minimum(ratio*adv[ids],ratio.clamp(.8,1.2)*adv[ids]).mean()
     loss+=.01*actor(xx[ids]).square().mean()
     with torch.no_grad():current_kl=float(((lp-old[ids]).exp()-1-(lp-old[ids])).mean())
     if current_kl<.02:
      opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),.5)
      params=[t.detach().clone() for t in actor.parameters()];moments=copy.deepcopy(opt.state_dict());rate=opt.param_groups[0]['lr'];accepted=False
      for attempt in range(5):
       if attempt:
        with torch.no_grad():
         for param,previous in zip(actor.parameters(),params):param.copy_(previous)
        opt.load_state_dict(copy.deepcopy(moments))
       for group in opt.param_groups:group['lr']=rate/(2**attempt)
       opt.step()
       with torch.no_grad():
        newlp=torch.distributions.Normal(policy.velocity_action(xx[ids],c),sigma).log_prob(zz[ids]).sum(-1)
        change=newlp-old[ids];kl=float((change.exp()-1-change).mean())
       if kl<=.03:
        accepted=True;accepted_steps+=1;actor_kls.append(kl);break
      if not accepted:
       with torch.no_grad():
        for param,previous in zip(actor.parameters(),params):param.copy_(previous)
       opt.load_state_dict(moments)
       for group in opt.param_groups:group['lr']=rate/32

     vl=(value(xx[ids]).squeeze(-1)-ret[ids]).square().mean()
     vopt.zero_grad(set_to_none=True);vl.backward();torch.nn.utils.clip_grad_norm_(value.parameters(),1.);vopt.step()
   if it%5==0:
    row=dict(iteration=it,seconds=time.monotonic()-beg,arm_velocity_rms=float(arm_motion.mean().sqrt()),actor_kl_max=max(actor_kls,default=0),actor_lr=opt.param_groups[0]['lr'],accepted_actor_steps=accepted_steps,reward=float(torch.stack([b[4] for b in batch]).mean()),vx=float(speed[:,0].mean()),falls=int(torch.stack([b[7] for b in batch]).sum()),timeouts=int(torch.stack([b[8] for b in batch]).sum()))
    trace.append(row);print(json.dumps(row),flush=True)
   if (it+1)%20==0:
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    for k,t in policy.state_dict().items():
     if not k.startswith('standing_skill.'):assert torch.equal(t.cpu(),ck['policy'][k])
    ck['policy']={k:v.cpu() for k,v in policy.state_dict().items()};ck.pop('velocity_training',None)
    ck['standing_skill']=dict(radius=.1,max_residual=.5)
    with torch.no_grad():
     for query in [[.3,0,0],[-.2,0,0],[0,0,.4],[0,0,-.4],[.6,0,0]]:
      cc=torch.tensor(query,device='cuda').expand(len(x),-1)
      baseline=policy.decoder(policy.encoders['velocity'](x,cc),x)
      assert torch.equal(baseline,policy.velocity_action(x,cc))
    ck['physical_stop']=dict(iterations=it+1,source_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),trace=trace,scope='Zero-command joint-motion PPO; original moving queries distilled; physical acceptance required; not resumable.')
    torch.save(ck,a.out/f'candidate_{it+1}.pt')
  (a.out/'report.json').write_text(json.dumps(trace,indent=2)+'\n')
 finally:env.close()
if __name__=='__main__':main()
