"""Merge moving, left-turn, and quiet-stop teachers into one velocity encoder.

Teacher switching occurs only in training collection/labels. Deployed policy has
one velocity encoder and the unchanged decoder, with no action overwrite.
"""
import argparse,copy,json,time,hashlib
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['checkpoint','stop_teacher','dataset','out']:p.add_argument('--'+k.replace('_','-'),type=Path,required=True)
 p.add_argument('--updates',type=int,default=10000)
 p.add_argument('--left-teacher',type=Path,required=True)
 p.add_argument('--move-weight',type=float,default=4.)
 a=p.parse_args();a.out.mkdir(exist_ok=False)
 import torch
 from supermimic.eval.benchmark import load_policy_from_checkpoint
 from supermimic.envs import SuperMimicEnvCfgArgs,make_env
 from supermimic.locomotion.config import VelocityCfg
 from supermimic.locomotion.keyboard import CommandMailbox
 from supermimic.locomotion.task import measured_velocity
 torch.set_num_threads(2);torch.manual_seed(43)
 policy,cfg=load_policy_from_checkpoint(a.checkpoint,'cuda:0');quiet,_=load_policy_from_checkpoint(a.stop_teacher,'cuda:0')
 for k,v in quiet.state_dict().items():
  if not k.startswith('encoders.velocity.'):assert torch.equal(v,policy.state_dict()[k]),k
 left_policy,_=load_policy_from_checkpoint(a.left_teacher,'cuda:0')
 for k,v in left_policy.state_dict().items():
  if not k.startswith('encoders.velocity.'):assert torch.equal(v,policy.state_dict()[k]),k
 left_policy.requires_grad_(False);left=left_policy.encoders['velocity']
 policy.requires_grad_(False);quiet.requires_grad_(False)
 move=copy.deepcopy(policy.encoders['velocity']).requires_grad_(False)
 stop=quiet.encoders['velocity']
 student=policy.encoders['velocity'];student.requires_grad_(True)
 env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(a.dataset/'walk.npz')],num_envs=64,seed=43,
  future_frames=cfg['env']['future_frames'],episode_length_s=72.,action_noise_std=0.,obs_noise=False,
  reset_noise=False,domain_randomization=False,record_transitions=False,anchor_obs=cfg['env']['anchor_obs'],velocity=VelocityCfg(**cfg['velocity'])),device='cuda:0')
 env.cfg.auto_reset=False;cmd=env.command_manager.get_term('motion');cmd.is_velocity[:]=True
 cmd.force_standing_reset=True;cmd.set_manual_command((0.,0.,0.))
 schedule=[(0.,(0.,0.,0.)),(3.,(.3,0.,0.)),(10.,(0.,0.,0.)),(15.,(-.2,0.,0.)),(22.,(0.,0.,0.)),(27.,(0.,0.,.4)),(34.,(0.,0.,0.)),(39.,(0.,0.,-.4)),(46.,(0.,0.,0.)),(51.,(.6,0.,0.)),(58.,(0.,0.,0.))]
 mailbox=CommandMailbox();rows=[];trace=[];failed=torch.zeros(64,dtype=torch.bool,device='cuda')
 try:
  obs,_=env.reset()
  with torch.no_grad():
   for step in range(3300):
    elapsed=step*env.step_dt;_,target=next(s for s in reversed(schedule) if elapsed>=s[0])
    mailbox.put(target,now=elapsed);cmd.set_manual_command(mailbox.read(env.step_dt,now=elapsed))
    c=cmd.velocity_command.clone();x=obs['policy'];zero=c.square().sum(-1)<1e-6
    z=torch.where(zero[:,None],stop(x,c),torch.where((c[:,2]>0)[:,None],left(x,c),move(x,c)));action=policy.decoder(z,x)
    if step%2==0:rows.append(x.clone())
    obs,_,term,tout,_=env.step(action);failed|=term|tout
    if step%25==0:trace.append(dict(time=(step+1)*env.step_dt,command=c.cpu().tolist(),velocity=measured_velocity(cmd.state_io.read()).cpu().tolist()))
    if failed.any():
     (a.out/'teacher_failure.json').write_text(json.dumps(dict(step=step+1,failed=failed.cpu().tolist(),trace=trace),indent=2)+'\n')
     raise RuntimeError('Combined teacher trajectory failed; preserve log, do not distill')
 finally:env.close()
 x=torch.stack(rows);train=x[:,:48].reshape(-1,x.shape[-1]);val=x[:,48:].reshape(-1,x.shape[-1])
 torch.save(dict(observations=x.cpu(),teacher_trace=trace),a.out/'teacher_states.pt')
 opt=torch.optim.Adam(student.parameters(),lr=2e-4)
 def label(xx,cc):
  with torch.no_grad():return torch.where((cc.square().sum(-1)<1e-6)[:,None],stop(xx,cc),torch.where((cc[:,2]>0)[:,None],left(xx,cc),move(xx,cc)))
 def queries(n):
  c=torch.zeros(n,3,device='cuda')
  choices=torch.tensor([[.3,0,0],[-.2,0,0],[0,0,.4],[0,0,-.4],[.6,0,0]],device='cuda')
  c[n//2:]=choices[torch.arange(n-n//2,device='cuda')%5]
  return c
 valx=val[torch.randint(len(val),(2048,),device='cuda')];valc=queries(2048);valz=label(valx,valc)
 history=[];started=time.monotonic();best=None;best_step=None;best_score=float('inf')
 for step in range(a.updates):
  xx=train[torch.randint(len(train),(1024,),device='cuda')];c=queries(1024);target=label(xx,c)
  pred=student(xx,c);weight=torch.where(c.square().sum(-1)>0,a.move_weight,1.)
  loss=((pred-target).square().mean(-1)*weight).mean()+20*((policy.decoder(pred,xx)-policy.decoder(target,xx)).square().mean(-1)*weight).mean()
  opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(student.parameters(),1.);opt.step()
  if (step+1)%200==0:
   with torch.no_grad():
    vz=student(valx,valc);ve=(vz-valz).square().mean(-1);ae=(policy.decoder(vz,valx)-policy.decoder(valz,valx)).square().mean(-1)
    result=dict(step=step+1,stop_latent_mse=float(ve[:1024].mean()),move_latent_mse=float(ve[1024:].mean()),stop_action_rmse=float(ae[:1024].mean().sqrt()),move_action_rmse=float(ae[1024:].mean().sqrt()))
   history.append(result);print(json.dumps(result),flush=True)
   score=float((ae*torch.where(valc.square().sum(-1)>0,a.move_weight,1.)).mean())
   if score<best_score:best_score=score;best=copy.deepcopy(student.state_dict());best_step=step+1
 student.load_state_dict(best)
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
 for k,v in policy.state_dict().items():
  if not k.startswith('encoders.velocity.'):assert torch.equal(v.cpu(),ck['policy'][k]),k
 ck['policy']={k:v.cpu() for k,v in policy.state_dict().items()};ck.pop('velocity_training',None)
 meta=dict(source_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),stop_sha256=hashlib.sha256(a.stop_teacher.read_bytes()).hexdigest(),updates=a.updates,selected_step=best_step,teacher_failures=0,teacher_seconds=66,move_weight=a.move_weight,action_loss_weight=20,left_sha256=hashlib.sha256(a.left_teacher.read_bytes()).hexdigest(),train_envs=48,validation_envs=16,history=history,seconds=time.monotonic()-started,scope='Single encoder and unchanged decoder; physical acceptance required; not resumable.')
 ck['quiet_stop_distillation']=meta;torch.save(ck,a.out/'candidate.pt');(a.out/'report.json').write_text(json.dumps(meta,indent=2)+'\n')
if __name__=='__main__':main()
