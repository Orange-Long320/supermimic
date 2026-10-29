"""Bounded physical PPO of a frozen-decoder velocity encoder (server only)."""
from pathlib import Path
import argparse,json,copy,hashlib,time

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['checkpoint','dataset','out']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--iterations',type=int,default=80)
 a=p.parse_args();a.out.mkdir(exist_ok=False)
 import torch
 from torch import nn
 from supermimic.eval.benchmark import load_policy_from_checkpoint
 from supermimic.envs import SuperMimicEnvCfgArgs,make_env
 from supermimic.locomotion.config import VelocityCfg
 from supermimic.locomotion.task import measured_velocity
 torch.set_num_threads(2);torch.manual_seed(29)
 policy,saved=load_policy_from_checkpoint(a.checkpoint,'cuda:0');policy.requires_grad_(False)
 actor=policy.encoders['velocity'];actor.requires_grad_(True)
 prior=copy.deepcopy(actor).requires_grad_(False)
 cfg=SuperMimicEnvCfgArgs(npz_files=[str(a.dataset/'backward.npz')],num_envs=256,seed=29,
  future_frames=saved['env']['future_frames'],episode_length_s=10.,action_noise_std=0.,obs_noise=False,
  reset_noise=False,domain_randomization=False,record_transitions=False,anchor_obs=saved['env']['anchor_obs'],velocity=VelocityCfg(**saved['velocity']))
 env=make_env(cfg,device='cuda:0');cmd=env.command_manager.get_term('motion')
 cmd.is_velocity[:]=True;cmd.force_standing_reset=True;cmd.set_manual_command((-.2,0.,0.))
 x,_=env.reset();x=x['policy'];goal=cmd.velocity_command.clone()
 value=nn.Sequential(nn.Linear(x.shape[-1],256),nn.Tanh(),nn.Linear(256,128),nn.Tanh(),nn.Linear(128,1)).cuda()
 opt=torch.optim.Adam(actor.parameters(),lr=5e-5);vopt=torch.optim.Adam(value.parameters(),lr=3e-4)
 sigma=.08;trace=[];beg=time.monotonic()
 try:
  for it in range(a.iterations):
   batch=[]
   with torch.no_grad():
    for step in range(32):
     mean=actor(x,goal);dist=torch.distributions.Normal(mean,sigma)
     z=dist.sample();lp=dist.log_prob(z).sum(-1);v=value(x).squeeze(-1)
     action=policy.decoder(z,x)
     nxt,_,term,tout,_=env.step(action);done=term|tout
     state=cmd.state_io.read();speed=measured_velocity(state)
     # Physical directional tracking resolves the standing solution directly.
     r=3*torch.exp(-((speed[:,0]+.2)/.15).square())+torch.exp(-(speed[:,2]/.3).square())
     r-=2*speed[:,1].square()+.02*(z-prior(x,goal)).square().mean(-1)+20*term.float()
     batch.append((x.clone(),z,lp,v,r,done));x=nxt['policy']
    tail=value(x).squeeze(-1);adv=[];gae=torch.zeros_like(tail)
    for b in reversed(batch):
     delta=b[4]+.99*tail*(~b[5])-b[3]
     gae=delta+.99*.95*(~b[5])*gae;adv.append(gae);tail=b[3]
    adv=torch.stack(adv[::-1]);ret=(adv+torch.stack([b[3] for b in batch])).flatten()
    adv=adv.flatten();adv=(adv-adv.mean())/adv.std().clamp_min(1e-6)
    xx=torch.cat([b[0] for b in batch]);zz=torch.cat([b[1] for b in batch]);old=torch.cat([b[2] for b in batch])
   for epoch in range(4):
    for ids in torch.randperm(len(xx),device='cuda').split(1024):
     c=goal[:1].expand(len(ids),-1);mean=actor(xx[ids],c)
     lp=torch.distributions.Normal(mean,sigma).log_prob(zz[ids]).sum(-1)
     ratio=(lp-old[ids]).exp();loss=-torch.minimum(ratio*adv[ids],ratio.clamp(.8,1.2)*adv[ids]).mean()
     # Preserve the other command interfaces on the same physical histories.
     other=c.clone();other[:,0]=.3;other[:len(ids)//2]=0
     with torch.no_grad():target=prior(xx[ids],other)
     loss+=10*(actor(xx[ids],other)-target).square().mean()
     opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),.5);opt.step()
     vl=(value(xx[ids]).squeeze(-1)-ret[ids]).square().mean()
     vopt.zero_grad(set_to_none=True);vl.backward();torch.nn.utils.clip_grad_norm_(value.parameters(),1.);vopt.step()
   if it%5==0:
    row=dict(iteration=it,seconds=time.monotonic()-beg,reward=float(torch.stack([b[4] for b in batch]).mean()),vx=float(speed[:,0].mean()),falls=int(torch.stack([b[5] for b in batch]).sum()))
    trace.append(row);print(json.dumps(row),flush=True)
   if (it+1)%20==0:
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    for k,t in policy.state_dict().items():
     if not k.startswith('encoders.velocity.'):assert torch.equal(t.cpu(),ck['policy'][k])
    ck['policy']={k:v.cpu() for k,v in policy.state_dict().items()};ck.pop('velocity_training',None)
    ck['physical_backward']=dict(iterations=it+1,source_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),trace=trace,scope='Backward-specific physical PPO; evaluation required; not resumable.')
    torch.save(ck,a.out/f'candidate_{it+1}.pt')
  (a.out/'report.json').write_text(json.dumps(trace,indent=2)+'\n')
 finally:env.close()
if __name__=='__main__':main()
