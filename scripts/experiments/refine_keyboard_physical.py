"""Bounded command-balanced physical refinement of a frozen-decoder policy.

All rollouts run in the dedicated server environment. Only the velocity encoder
is updated. Commands condition both actor and critic. Candidate checkpoints must
pass independent continuous physical evaluation before they are adopted.
"""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'dataset', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--iterations', type=int, default=240)
    p.add_argument('--num-envs', type=int, default=512)
    p.add_argument('--seed', type=int, default=71)
    p.add_argument('--lr', type=float, default=3e-5)
    a = p.parse_args()
    if Path(sys.prefix).resolve() != Path('/home/2025-Long.xh/HHD/venvs/supermimic').resolve():
        raise RuntimeError('Use the dedicated server interpreter')
    if not 1 <= a.iterations <= 1200 or not 2 <= a.num_envs <= 1024:
        raise ValueError('Bounded run required')
    a.out.mkdir(exist_ok=False)
    import torch
    from torch import nn
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.locomotion.config import VelocityCfg
    from supermimic.locomotion.task import measured_velocity
    from supermimic.robots.dimensions import NQ, PROFILE
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    policy, saved = load_policy_from_checkpoint(a.checkpoint, 'cuda:0')
    policy.requires_grad_(False)
    actor = policy.encoders['velocity']
    prior = copy.deepcopy(actor).requires_grad_(False)
    actor.requires_grad_(True)
    vc = VelocityCfg(**saved['velocity'])
    env = make_env(SuperMimicEnvCfgArgs(npz_files=[str(a.dataset/'walk.npz')], num_envs=a.num_envs,
        seed=a.seed, future_frames=saved['env']['future_frames'], episode_length_s=24,
        action_noise_std=0, obs_noise=False, reset_noise=False, domain_randomization=False,
        record_transitions=False, anchor_obs=saved['env']['anchor_obs'], velocity=vc), device='cuda:0')
    cmd = env.command_manager.get_term('motion')
    cmd.is_velocity[:] = True
    cmd.force_standing_reset = True
    commands = torch.tensor([[0,0,0],[.3,0,0],[-.2,0,0],[0,0,.4],[0,0,-.4],[.6,0,0]], device='cuda')
    n = a.num_envs
    goal = torch.zeros(n,3,device='cuda')
    target = commands[torch.arange(n,device='cuda') % 6].clone()
    # Batched commands use the existing command term's internal tensor interface.
    cmd.manual_command = goal
    cmd._refresh_velocity()
    obs,_ = env.reset()
    x = obs['policy']
    value = nn.Sequential(nn.Linear(x.shape[-1]+3,256), nn.Tanh(), nn.Linear(256,128), nn.Tanh(), nn.Linear(128,1)).cuda()
    scale = actor.obs_scale
    cscale = actor.command_scale
    def val(xx,cc):
        return value(torch.cat((xx*scale,cc/cscale),-1)).squeeze(-1)
    opt = torch.optim.Adam(actor.parameters(),lr=a.lr)
    vopt = torch.optim.Adam(value.parameters(),lr=3e-4)
    arms = [i for i,name in enumerate(PROFILE['joint_names']) if any(s in name for s in ('shoulder','elbow','wrist'))]
    sigma = .05
    phase_age = torch.randint(0,150,(n,),device='cuda')
    previous = torch.zeros(n, len(PROFILE['joint_names']), device='cuda')
    trace=[]
    started=time.monotonic()
    source_hash=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    torch.cuda.reset_peak_memory_stats()
    try:
        for it in range(a.iterations):
            batch=[]
            with torch.no_grad():
                for step in range(32):
                    switch = phase_age >= 150
                    target[switch] = commands[torch.randint(6,(int(switch.sum()),),device='cuda')]
                    phase_age[switch] = 0
                    goal += (target-goal).clamp(-.02,.02)
                    cmd._refresh_velocity()
                    c = goal.clone()
                    mean=actor(x,c)
                    dist=torch.distributions.Normal(mean,sigma)
                    z=dist.sample()
                    lp=dist.log_prob(z).sum(-1)
                    v=val(x,c)
                    action=policy.decoder(z,x)
                    nxt,_,term,tout,_=env.step(action)
                    done=term|tout
                    state=cmd.state_io.read()
                    speed=measured_velocity(state)
                    err=speed-c
                    stop=c.square().sum(-1)<1e-6
                    dq=state[:,NQ+6:]
                    reward=(1-12*err[:,0].square()-8*err[:,1].square()-4*err[:,2].square()
                        -.005*dq.square().mean(-1)-.01*(action-previous).square().sum(-1)
                        -stop*(.5*dq[:,arms].square().mean(-1)+.03*dq.square().mean(-1))
                        -10*term.float())
                    # No bootstrap through any reset. Timeouts are logged separately;
                    # the reset observation is never treated as a continuation.
                    batch.append((x.clone(),c,z,lp,v,reward,done,term,tout,speed))
                    x=nxt['policy'];previous=action;phase_age+=1
                    phase_age[done]=150
                tail=val(x,goal)
                advantages=[];gae=torch.zeros_like(tail)
                for b in reversed(batch):
                    delta=b[5]+.99*tail*(~b[6])-b[4]
                    gae=delta+.99*.95*(~b[6])*gae
                    advantages.append(gae);tail=b[4]
                advantages=torch.stack(advantages[::-1])
                returns=(advantages+torch.stack([b[4] for b in batch])).flatten()
                advantages=advantages.flatten()
                advantages=(advantages-advantages.mean())/advantages.std().clamp_min(1e-6)
                xx=torch.cat([b[0] for b in batch]);cc=torch.cat([b[1] for b in batch])
                zz=torch.cat([b[2] for b in batch]);old=torch.cat([b[3] for b in batch])
            kl_values=[]
            for epoch in range(4):
                for ids in torch.randperm(len(xx),device='cuda').split(2048):
                    mean=actor(xx[ids],cc[ids])
                    lp=torch.distributions.Normal(mean,sigma).log_prob(zz[ids]).sum(-1)
                    ratio=(lp-old[ids]).exp()
                    loss=-torch.minimum(ratio*advantages[ids],ratio.clamp(.8,1.2)*advantages[ids]).mean()
                    # Counterfactual queries prevent one command from overwriting others.
                    other=commands[torch.arange(len(ids),device='cuda')%6]
                    with torch.no_grad():
                        teacher=prior(xx[ids],other)
                        teacher_action=policy.decoder(teacher,xx[ids])
                    latent=actor(xx[ids],other)
                    loss += .1*(latent-teacher).square().mean()+.1*(policy.decoder(latent,xx[ids])-teacher_action).square().mean()
                    opt.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(),.5);opt.step()
                    vl=(val(xx[ids],cc[ids])-returns[ids]).square().mean()
                    vopt.zero_grad(set_to_none=True);vl.backward()
                    torch.nn.utils.clip_grad_norm_(value.parameters(),1.);vopt.step()
                    kl_values.append(float(((ratio-1)-(lp-old[ids])).mean().detach()))
                if sum(kl_values[-8:])/min(8,len(kl_values)) > .03:
                    break
            if (it+1)%10 == 0:
                speeds=torch.cat([b[9] for b in batch]);error=speeds-cc
                steady=(cc[:,None,:]-commands[None,:,:]).abs().amax(-1)<1e-5
                row=dict(iteration=it+1,seconds=time.monotonic()-started,reward=float(torch.stack([b[5] for b in batch]).mean()),
                    error_rmse=error.square().mean(0).sqrt().tolist(),terminated=int(torch.stack([b[7] for b in batch]).sum()),
                    timeouts=int(torch.stack([b[8] for b in batch]).sum()),kl=sum(kl_values)/len(kl_values),
                    commands=[dict(target=commands[j].tolist(),n=int(steady[:,j].sum()),actual=speeds[steady[:,j]].mean(0).tolist()) for j in range(6)])
                trace.append(row);print(json.dumps(row),flush=True)
            if (it+1)%40 == 0 or it+1 == a.iterations:
                ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
                for key,t in policy.state_dict().items():
                    if not key.startswith('encoders.velocity.'):
                        assert torch.equal(t.cpu(),ck['policy'][key]),key
                ck['policy']={k:v.cpu() for k,v in policy.state_dict().items()}
                ck.pop('velocity_training',None)
                meta=dict(source=str(a.checkpoint),source_sha256=source_hash,iteration=it+1,
                    config=vars(a),sigma=sigma,commands=commands.tolist(),trace=trace,
                    peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                    scope='Physical correction only; frozen decoder/motion/world model; fresh optimizers; independent acceptance required.')
                ck['physical_refinement']=meta
                dest=a.out/f'candidate_{it+1:04d}.pt'
                torch.save(ck,dest)
                (a.out/'report.json').write_text(json.dumps(meta,indent=2,default=str)+'\n')
    finally:
        env.close()

if __name__=='__main__':
    main()
