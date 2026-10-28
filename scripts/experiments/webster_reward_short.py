"""Bounded shared-GPU reward comparison, using the project Trainer.

Overrides only this process's real and imagined joint-limit reward weights.
The existing long jobs and their inputs remain untouched. This is an experiment,
not approval of the imperfect teacher or a motion-success certificate.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch

from supermimic.config import TrainConfig,DataCfg,EnvCfg,ScheduleCfg
from supermimic.imagination.policy_trainer import PolicyTrainerCfg
from supermimic.policy.nets import PolicyCfg,CriticCfg
from supermimic.training.loop import Trainer
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModelCfg
from supermimic.world_model.wm_trainer import WMTrainerCfg


def construct(cls, values):
    return cls(**{k:tuple(v) if isinstance(v,list) else v for k,v in values.items()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--shared-pid',type=int,required=True)
    p.add_argument('--joint-weight',type=float,required=True)
    p.add_argument('--steps',type=int,default=120)
    p.add_argument('--envs',type=int,default=256)
    p.add_argument('--batch',type=int,default=128)
    p.add_argument('--max-seconds',type=float,default=240)
    p.add_argument('--memory-fraction',type=float,default=.18)
    p.add_argument('--sampling-mode',choices=['return','start'],default='return')
    p.add_argument('--learning-rate',type=float,default=2e-4)
    p.add_argument('--foot-weight',type=float,default=0.)
    p.add_argument('--nonfoot-weight',type=float,default=0.)
    p.add_argument('--air-rotation-weight',type=float,default=0.)
    p.add_argument('--physical-audit',type=Path)
    p.add_argument('--snapshot-local-steps',type=int,nargs='+',default=[],help='Save immutable evaluation-only policies after these local steps')
    args=p.parse_args()
    if args.out.exists() or args.ckpt.name=='ckpt_latest.pt':p.error('Use a fresh output directory and immutable checkpoint')
    if not -50<=args.joint_weight<=-2 or not 0<args.memory_fraction<=.2 or args.steps<45:
        p.error('Invalid bounded comparison settings')
    if not 0<args.learning_rate<=2e-4 or not 0<=args.foot_weight<=100 or not 0<=args.nonfoot_weight<=100 or not 0<=args.air_rotation_weight<=.1:
        p.error('Invalid experimental physical reward or learning rate')
    if any(s<45 or s>args.steps for s in args.snapshot_local_steps) or len(set(args.snapshot_local_steps))!=len(args.snapshot_local_steps):
        p.error('Snapshot steps must be distinct and inside the actual learning interval')
    physical_enabled=any([args.foot_weight,args.nonfoot_weight,args.air_rotation_weight])
    audit=None
    if physical_enabled:
        if args.physical_audit is None:p.error('Physical rewards require a matching completed audit')
        audit=json.loads(args.physical_audit.read_text())
        if not audit.get('passed') or audit['native_geometry_maximum_error_m']>=1e-5 or audit.get('static_manager_fixture_maximum_error',1.)>=1e-6:
            p.error('Physical reward audit failed or lacks real/imagined manager validation')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.gpu):raise ValueError('Explicit physical GPU selection is required')
    query=lambda fields:subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu='+fields,'--format=csv,noheader,nounits'],text=True).strip()
    free=int(query('memory.free'))
    if free<7000:raise ValueError(f'Insufficient shared-card headroom: {free} MiB free')
    listed=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    owners=[int(line.strip()) for line in listed.splitlines() if line.strip().isdigit()]
    if owners!=[args.shared_pid] or Path(f'/proc/{args.shared_pid}').stat().st_uid!=os.getuid():
        raise ValueError('Only the explicitly selected existing process owned by this project user may be shared')
    torch.set_num_threads(1);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(args.memory_fraction,0)
    checkpoint=torch.load(args.ckpt,map_location='cpu',weights_only=False);c=checkpoint['config']
    if c['robot']!='x2_ultra':raise ValueError('This comparison is for X2 Ultra')
    cfg=TrainConfig();cfg.robot='x2_ultra';cfg.device='cuda:0';cfg.resume=args.ckpt
    cfg.log_root=args.out.parent;cfg.run_name=args.out.name;cfg.seed=0
    cfg.data=construct(DataCfg,c['data'])
    cfg.data.npz_dir=Path(cfg.data.npz_dir)
    if cfg.data.manifest is not None:cfg.data.manifest=Path(cfg.data.manifest)
    cfg.env=construct(EnvCfg,c['env']);cfg.env.num_envs=args.envs;cfg.env.replay_capacity=128
    cfg.env.sampling_mode=args.sampling_mode
    cfg.env.domain_randomization=False;cfg.env.obs_noise=False;cfg.env.reset_noise=False;cfg.env.action_noise_std=0.
    world=dict(c['world_model']);world['state_spec']=construct(StateSpecCfg,world['state_spec'])
    cfg.world_model=construct(WorldModelCfg,world)
    cfg.policy=construct(PolicyCfg,c['policy']);cfg.critic=construct(CriticCfg,c['critic'])
    cfg.wm_trainer=construct(WMTrainerCfg,c['wm_trainer']);cfg.wm_trainer.batch_size=args.batch;cfg.wm_trainer.batches_per_update=2;cfg.wm_trainer.compile=None
    cfg.policy_trainer=construct(PolicyTrainerCfg,c['policy_trainer']);cfg.policy_trainer.batch_size=args.batch;cfg.policy_trainer.compile=None;cfg.policy_trainer.obs_noise=False
    cfg.wm_trainer.lr=cfg.policy_trainer.lr=cfg.policy_trainer.critic_lr=args.learning_rate
    cfg.schedule=construct(ScheduleCfg,c['schedule']);cfg.schedule.total_iterations=checkpoint['iteration']+args.steps
    cfg.schedule.warmup_iterations=40;cfg.schedule.log_interval=10;cfg.schedule.checkpoint_interval=1_000_000
    cfg.schedule.max_hours=args.max_seconds/3600;cfg.schedule.lr_final_ratio=1.
    trainer=Trainer(cfg)
    try:
        if not all(torch.equal(v.cpu(),checkpoint['policy'][k]) for k,v in trainer.policy.state_dict().items()):
            raise ValueError('Short-run initial policy differs from its source checkpoint')
        term=trainer.env.reward_manager.get_term_cfg('joint_limit')
        old_real=float(term.weight);old_imagined=float(trainer.wm.heads.reward_cfg.w_joint_limit)
        term.weight=args.joint_weight;trainer.wm.heads.reward_cfg.w_joint_limit=args.joint_weight
        if trainer.env.reward_manager.get_term_cfg('joint_limit').weight!=trainer.wm.heads.reward_cfg.w_joint_limit:
            raise ValueError('Real and imagined reward weights differ')
        physical_cfg=None
        if physical_enabled:
            from supermimic.data.motion_convert import load_g1_model
            from supermimic.world_model.physical_reward import PhysicalRewardCfg,install_physical_reward
            reference=cfg.data.npz_dir/'webster.npz'
            if hashlib.sha256(reference.read_bytes()).hexdigest() not in audit['sha256'].values():
                raise ValueError('Physical audit does not cover this training reference')
            physical_cfg=PhysicalRewardCfg(foot_weight=args.foot_weight,nonfoot_weight=args.nonfoot_weight,
                air_rotation_weight=args.air_rotation_weight)
            install_physical_reward(trainer,load_g1_model(),physical_cfg)
            (args.out/'physical_reward-as-run.py').write_bytes(Path('src/supermimic/world_model/physical_reward.py').read_bytes())
        metadata={'experiment_only':True,'approved_for_training':False,'physical_candidate_passed':False,
            'source_checkpoint':str(args.ckpt),'source_sha256':hashlib.sha256(args.ckpt.read_bytes()).hexdigest(),
            'initial_iteration':trainer.iteration,'initial_updates':trainer.num_updates,
            'physical_gpu':args.gpu,'shared_pid':args.shared_pid,'gpu_uuid':query('uuid'),'free_gpu_mib_at_start':free,
            'torch_memory_fraction':args.memory_fraction,'torch_memory_cap_excludes_native_Warp_allocations':True,
            'physical_reward':asdict(physical_cfg) if physical_cfg else None,
            'physical_audit_sha256':hashlib.sha256(args.physical_audit.read_bytes()).hexdigest() if physical_enabled else None,
            'physical_reward_in_imagined_total_but_excluded_from_original_eight_term_breakdown':physical_enabled,
            'reward_override':{'real_old':old_real,'imagined_old':old_imagined,'real_new':float(term.weight),
                'imagined_new':float(trainer.wm.heads.reward_cfg.w_joint_limit)},
            'scope':'Existing imperfect exploratory teacher, bounded reward test; source long jobs unmodified',
            'arguments':vars(args)}
        (args.out/'runtime-reward-overrides.json').write_text(json.dumps(metadata,indent=2,default=str))
        (args.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
        def save_evaluation(destination):
            if destination.exists():raise ValueError('Evaluation snapshots must be immutable')
            evaluation={'config':asdict(cfg),'policy':{k:v.cpu() for k,v in trainer.policy.state_dict().items()},
                'iteration':trainer.iteration,'num_updates':trainer.num_updates,
                'reward_override':metadata['reward_override'],'physical_reward':metadata['physical_reward'],
                'source_checkpoint_sha256':metadata['source_sha256'],
                'local_environment_steps':trainer.iteration-metadata['initial_iteration'],
                'policy_evaluation_only':True,'approved_for_training':False}
            temporary=destination.with_suffix('.tmp');torch.save(evaluation,temporary);temporary.replace(destination)
        pending=set(args.snapshot_local_steps)
        original_act=trainer.act
        def act_with_snapshot(obs):
            local=trainer.iteration-metadata['initial_iteration']
            if local in pending:
                save_evaluation(args.out/f'policy_local{local:05d}.pt');pending.remove(local)
            return original_act(obs)
        if pending:trainer.act=act_with_snapshot
        initial_updates=trainer.num_updates;started=time.monotonic()
        trainer.run()
        finite=all(torch.isfinite(v).all().item() for module in [trainer.wm,trainer.policy,trainer.critic]
                   for v in module.state_dict().values() if torch.is_tensor(v))
        if trainer.num_updates<=initial_updates or not finite:raise ValueError('No valid learning update completed')
        local=trainer.iteration-metadata['initial_iteration']
        if local in pending:save_evaluation(args.out/f'policy_local{local:05d}.pt');pending.remove(local)
        save_evaluation(args.out/'policy_evaluation.pt')
        summary={'elapsed_s':time.monotonic()-started,'final_iteration':trainer.iteration,
            'completed_updates':trainer.num_updates-initial_updates,'parameters_finite':finite,
            'torch_peak_allocated_mib':torch.cuda.max_memory_allocated(0)/2**20,
            'torch_peak_reserved_mib':torch.cuda.max_memory_reserved(0)/2**20,
            'evaluation_checkpoint_sha256':hashlib.sha256((args.out/'policy_evaluation.pt').read_bytes()).hexdigest(),
            'requested_snapshot_steps_not_reached':sorted(pending),
            'approved_for_training':False,'physical_evaluation_pending':True}
        (args.out/'completed.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    finally:
        trainer.env.close()
        if trainer.writer is not None:trainer.writer.close()


if __name__=='__main__':main()
