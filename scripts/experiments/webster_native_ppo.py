"""Bounded native-feedback PPO fine-tuning of an existing Webster policy.

Uses the installed RSL-RL optimizer and unchanged project simulation. This is
an explicitly different experimental optimizer, not a SuperMimic reproduction
result. No learned world-model dynamics or gradients enter PPO updates.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.storage import RolloutStorage

from supermimic.data.motion_convert import load_g1_model
from supermimic.envs import SuperMimicEnvCfgArgs,make_env
from supermimic.envs.state_io import StateIO
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.robots.dimensions import NJ,PROFILE
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.physical_reward import PhysicalRewardCfg,install_physical_reward
from supermimic.world_model.state_spec import StateSpec,StateSpecCfg


class WarmActor(MLPModel):
    """Expose the existing mean policy through RSL-RL's model interface."""
    def __init__(self,policy,std):
        torch.nn.Module.__init__(self)
        self.policy=policy;self.obs_normalization=False
        self.distribution=GaussianDistribution(NJ,init_std=std,learn_std=False)

    def forward(self,obs,masks=None,hidden_state=None,stochastic_output=False):
        if masks is not None or hidden_state is not None:raise ValueError('Only non-recurrent batches are supported')
        mean=self.policy(obs['policy'],obs['goal'])
        if stochastic_output:
            self.distribution.update(mean);return self.distribution.sample()
        return mean


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--physical-audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--shared-pids',nargs='+',type=int,required=True)
    p.add_argument('--envs',type=int,default=256);p.add_argument('--horizon',type=int,default=64)
    p.add_argument('--updates',type=int,default=4);p.add_argument('--max-seconds',type=float,default=240.)
    p.add_argument('--actor-lr',type=float,default=1e-6);p.add_argument('--critic-lr',type=float,default=3e-4)
    p.add_argument('--action-std',type=float,default=.03);p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--minibatches',type=int,default=4);p.add_argument('--snapshot-every',type=int,default=4)
    args=p.parse_args()
    if args.out.exists() or args.ckpt.name=='ckpt_latest.pt':p.error('Use a fresh output and immutable source')
    if not 1<=args.envs<=1024 or not 16<=args.horizon<=128 or not 1<=args.updates<=256 or not 0<args.actor_lr<=1e-5:
        p.error('Invalid bounded native-feedback experiment')
    if not 0<args.action_std<=.1 or args.epochs<1 or args.minibatches<1 or args.snapshot_every<1:
        p.error('Invalid PPO configuration')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.gpu):raise ValueError('Explicit physical GPU selection is required')
    def query(kind,fields):
        return subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-'+kind+'='+fields,'--format=csv,noheader,nounits'],text=True).strip()
    free=int(query('gpu','memory.free'));owners=[int(line) for line in query('compute-apps','pid').splitlines() if line.strip().isdigit()]
    if free<6500 or not owners or not set(owners)<=set(args.shared_pids):raise ValueError('Unapproved GPU occupants or insufficient headroom')
    if any(Path(f'/proc/{pid}').stat().st_uid!=os.getuid() for pid in owners):raise ValueError('Only explicitly selected project-user processes may be shared')
    audit=json.loads(args.physical_audit.read_text());reference_sha=hashlib.sha256(args.source.read_bytes()).hexdigest()
    if not audit.get('passed') or audit.get('static_manager_fixture_maximum_error',1.)>=1e-6 or reference_sha not in audit['sha256'].values():
        raise ValueError('Matching physical reward calibration is required')
    args.out.mkdir(parents=True);torch.set_num_threads(1);torch.manual_seed(0)
    torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.15);device='cuda:0'
    policy,config=load_policy_from_checkpoint(args.ckpt,device)
    if config['robot']!='x2_ultra':raise ValueError('This experiment is for X2 Ultra')
    initial_weights={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}
    env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(args.source)],num_envs=args.envs,
        future_frames=config['env']['future_frames'],action_noise_std=0.,obs_noise=False,reset_noise=False,
        domain_randomization=False,sampling_mode='start',record_transitions=False,
        anchor_obs=config['env'].get('anchor_obs',False),seed=0),device)
    try:
        model=load_g1_model();fk=G1Kinematics(model,device);robot=env.scene['robot']
        heads=AnalyticHeads(fk,tuple(PROFILE['tracked_bodies']),PROFILE['anchor_body'],PROFILE['root_body'],
            robot.data.soft_joint_pos_limits[0].clone(),robot.data.default_joint_pos[0].clone())
        heads.reward_cfg.w_joint_limit=-10.
        setup=SimpleNamespace(env=env,wm=SimpleNamespace(heads=heads),device=device,max_step_reward=heads.reward_cfg.max_step_reward)
        physical_cfg=PhysicalRewardCfg();install_physical_reward(setup,model,physical_cfg)
        io=StateIO(env);spec=StateSpec(StateSpecCfg(**config['world_model']['state_spec']),device)
        actor=WarmActor(policy,args.action_std).to(device)
        def observation(raw):
            # Clone before env.step mutates observation-manager buffers.
            return TensorDict({'policy':raw['policy'].clone(),'goal':raw['goal'].clone(),
                'critic':torch.cat([spec.canonical(io.read()),raw['policy']*policy.decoder.scale,
                    raw['goal']*policy.encoders['motion'].scale],-1)},batch_size=[args.envs],device=device)
        with torch.no_grad():
            raw,_=env.reset();obs=observation(raw)
            mean_error=float(abs(actor(obs)-policy(raw['policy'],raw['goal'])).max())
        if mean_error!=0 or not all(torch.equal(v.cpu(),initial_weights[k]) for k,v in policy.state_dict().items()):
            raise ValueError('PPO adapter changed the initial deterministic policy')
        critic=MLPModel(obs,{'critic':['critic']},'critic',1,hidden_dims=(256,128),obs_normalization=False).to(device)
        storage=RolloutStorage('rl',args.envs,args.horizon,obs,[NJ],device)
        algo=PPO(actor,critic,storage,num_learning_epochs=args.epochs,num_mini_batches=args.minibatches,
            clip_param=.1,gamma=.99,lam=.95,entropy_coef=0.,learning_rate=args.actor_lr,
            use_clipped_value_loss=False,schedule='fixed',device=device)
        # The installed optimizer supports separate fixed parameter-group rates.
        algo.optimizer=torch.optim.Adam([{'params':actor.parameters(),'lr':args.actor_lr},
            {'params':critic.parameters(),'lr':args.critic_lr}])
        package_files={name:Path(inspect.getfile(cls)) for name,cls in {'ppo':PPO,'storage':RolloutStorage,'distribution':GaussianDistribution}.items()}
        metadata={'algorithm':'Installed RSL-RL PPO with the existing SuperMimic mean policy',
            'rsl_rl_version':importlib.metadata.version('rsl-rl-lib'),
            'installed_algorithm_sha256':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in package_files.items()},
            'source_checkpoint_sha256':hashlib.sha256(args.ckpt.read_bytes()).hexdigest(),'reference_sha256':reference_sha,
            'gpu_occupants_at_start':owners,'free_gpu_mib_at_start':free,'torch_memory_fraction':.15,
            'initial_mean_adapter_error':mean_error,'world_model_dynamics_used':False,'critic_initialized_fresh':True,
            'clip_end_is_finite_episode_terminal_no_timeout_bootstrap':True,
            'joint_weight':float(env.reward_manager.get_term_cfg('joint_limit').weight),
            'physical_reward':vars(physical_cfg),'arguments':vars(args),
            'physical_candidate_passed':False,'approved_for_training':False}
        (args.out/'metadata.json').write_text(json.dumps(metadata,indent=2,default=str))
        (args.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
        (args.out/'physical_reward-as-run.py').write_bytes(Path('src/supermimic/world_model/physical_reward.py').read_bytes())
        history=[];episode_counts=torch.zeros(3,dtype=torch.int64,device=device)
        completed_episodes=0;physical_terminations=0;clip_completions=0;started=time.monotonic();stopped_for_kl=False
        def snapshot(name,update):
            target=args.out/name
            if target.exists():raise ValueError('Evaluation snapshots must be immutable')
            evaluation_config=deepcopy(config)
            evaluation_config['env'].update(num_envs=args.envs,action_noise_std=0.,obs_noise=False,
                reset_noise=False,domain_randomization=False,sampling_mode='start')
            checkpoint={'config':evaluation_config,'policy':{k:v.detach().cpu() for k,v in policy.state_dict().items()},
                'ppo_updates':update,'local_environment_steps':update*args.horizon,
                'source_checkpoint_sha256':metadata['source_checkpoint_sha256'],'optimizer_algorithm':'rsl_rl.PPO',
                'policy_evaluation_only':True,'approved_for_training':False}
            temporary=target.with_suffix('.tmp');torch.save(checkpoint,temporary);temporary.replace(target)
        for update in range(1,args.updates+1):
            rollout_started=time.monotonic()
            with torch.no_grad():
                for _ in range(args.horizon):
                    action=algo.act(obs);raw,reward,terminated,timeout,_=env.step(action)
                    next_obs=observation(raw);done=terminated|timeout
                    algo.process_env_step(next_obs,reward,done,{})
                    episode_counts+=torch.stack([done.sum(),terminated.sum(),(timeout&~terminated).sum()])
                    obs=next_obs
                if storage.step!=args.horizon:raise ValueError('Incomplete PPO rollout')
                algo.compute_returns(obs)
            rollout_s=time.monotonic()-rollout_started;update_started=time.monotonic()
            losses=algo.update()
            completed_episodes,physical_terminations,clip_completions=episode_counts.tolist()
            with torch.no_grad():
                flat=storage.observations.flatten(0,1);old_mean=storage.distribution_params[0].reshape(-1,NJ)
                kl=[]
                for start in range(0,len(old_mean),1024):
                    current=actor(flat[start:start+1024]);kl.append(((current-old_mean[start:start+1024])/args.action_std).square().sum(-1)/2)
                measured_kl=float(torch.cat(kl).mean())
            finite=all(torch.isfinite(v).all().item() for module in [actor,critic] for v in module.state_dict().values())
            if not finite or not all(torch.isfinite(torch.tensor(v)) for v in losses.values()):raise ValueError('Non-finite PPO update')
            record={'update':update,'local_environment_steps':update*args.horizon,'losses':losses,
                'measured_policy_kl':measured_kl,'completed_episodes':completed_episodes,
                'physical_terminations':physical_terminations,'clip_completions':clip_completions,
                'rollout_s':rollout_s,'update_s':time.monotonic()-update_started,'elapsed_s':time.monotonic()-started}
            history.append(record);(args.out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(record),flush=True)
            if update%args.snapshot_every==0:snapshot(f'policy_update{update:04d}.pt',update)
            if measured_kl>.2:stopped_for_kl=True;break
            if time.monotonic()-started>args.max_seconds:break
        snapshot('policy_evaluation.pt',update)
        torch.save({'actor':actor.state_dict(),'critic':critic.state_dict(),'optimizer':algo.optimizer.state_dict(),
            'metadata':metadata,'ppo_updates':update},args.out/'ppo_state.pt')
        result={'updates_completed':update,'local_environment_steps':update*args.horizon,'elapsed_s':time.monotonic()-started,
            'completed_episodes':completed_episodes,'physical_terminations':physical_terminations,'clip_completions':clip_completions,
            'stopped_for_kl':stopped_for_kl,'parameters_finite':True,
            'torch_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'torch_peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
            'evaluation_checkpoint_sha256':hashlib.sha256((args.out/'policy_evaluation.pt').read_bytes()).hexdigest(),
            'physical_evaluation_pending':True,'physical_candidate_passed':False,'approved_for_training':False}
        (args.out/'completed.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    finally:env.close()


if __name__=='__main__':main()
