"""Reviewed Omni continuation on one GPU or existing two-rank gradient averaging.

Each GPU owns its simulator/replay; both ranks update one synchronized policy.
Source-start and return sampling remain50/50. No robot physics or acceptance
thresholds change. User authorized two GPUs for three hours at about12GiB each.
"""
import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint','dataset','out']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--phase',choices=['capacity','optimize'],required=True)
    p.add_argument('--proof',type=Path)
    p.add_argument('--minutes',type=float,default=180.)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--package-commit',required=True)
    p.add_argument('--num-envs',type=int,default=2048)
    p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--source-world-size',type=int,default=1,help='Normalize checkpoint LR before the existing trainer applies its world-size scaling')
    a=p.parse_args()
    if not 0<a.minutes<=180 or a.num_envs<1 or a.batch_size<1:p.error('Bounded three-hour run required')
    rank,local,world=int(os.environ.get('RANK','0')),int(os.environ.get('LOCAL_RANK','0')),int(os.environ.get('WORLD_SIZE','1'))
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if world not in [1,2] or len(visible)!=world:raise ValueError('Explicit one GPU per rank required')
    physical=visible[local]
    os.environ['CUDA_VISIBLE_DEVICES']=physical
    os.environ['SM_GPU_PINNED']='1'
    if hasattr(os,'sched_getaffinity'):
        cores=sorted(os.sched_getaffinity(0));size=len(cores)//world
        if size<2:raise ValueError('Two CPU cores per rank required')
        os.sched_setaffinity(0,cores[local*size:(local+1)*size])
    import torch
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra',check_assets=True)
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    manifest=json.loads((a.dataset/'manifest.json').read_text())
    if (not manifest.get('exploratory_training_authorized') or manifest['distinct_source_families']!=3
        or set(manifest['task_categories'])!={'box_stand','vault_land','step_up_down_land'}
        or sha(manifest['terrain'])!=manifest['terrain_sha256']):raise ValueError('Immutable reviewed three-task dataset required')
    for clip in manifest['clips']:
        if sha(a.dataset/'motions'/(clip['name']+'.npz'))!=clip['motion_sha256']:raise ValueError('Motion changed')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.4,0)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    def restore(obj,data):
        for k,v in data.items():
            current=getattr(obj,k)
            if is_dataclass(current):restore(current,v)
            else:setattr(obj,k,tuple(v) if isinstance(current,tuple) else v)
    cfg=TrainConfig();restore(cfg,ck['config'])
    incoming_world=json.loads(a.proof.read_text())['world_size'] if a.phase=='optimize' and a.proof else a.source_world_size
    if incoming_world not in [1,2]:raise ValueError('Explicit known checkpoint world size required')
    cfg.wm_trainer.lr/=incoming_world**.5
    cfg.policy_trainer.lr/=incoming_world**.5
    cfg.policy_trainer.critic_lr/=incoming_world**.5
    base_learning_rates={'wm':cfg.wm_trainer.lr,'policy':cfg.policy_trainer.lr,'critic':cfg.policy_trainer.critic_lr}
    if cfg.robot!='x2_ultra' or cfg.env.box is not None or cfg.env.sole_profile is not None:raise ValueError('Unchanged X2 terrain interface required')
    if not cfg.env.terrain_manifest or Path(cfg.env.terrain_manifest).read_bytes()!=Path(manifest['terrain']).read_bytes():
        raise ValueError('Full continuation must use the same terrain')
    initial_iteration,initial_updates=ck['iteration'],ck['num_updates']
    del ck
    cfg.data.npz_dir=a.dataset/'motions';cfg.resume=a.checkpoint
    cfg.env.terrain_manifest=Path(manifest['terrain']);cfg.world_model.state_spec.terrain_manifest=cfg.env.terrain_manifest
    cfg.run_name=a.out.name;cfg.log_root=a.out.parent
    cfg.env.num_envs=a.num_envs;cfg.env.replay_capacity=256
    cfg.env.sampling_mode='return';cfg.env.action_noise_std=.05
    cfg.env.reset_noise=cfg.env.obs_noise=cfg.env.domain_randomization=False
    cfg.policy_trainer.obs_noise=False
    cfg.wm_trainer.batch_size=cfg.policy_trainer.batch_size=a.batch_size
    cfg.wm_trainer.compile=cfg.policy_trainer.compile=None
    cfg.schedule.total_iterations=initial_iteration+100 if a.phase=='capacity' else 100_000_000
    cfg.schedule.max_hours=.15 if a.phase=='capacity' else a.minutes/60
    cfg.schedule.log_interval=20;cfg.schedule.checkpoint_interval=100 if a.phase=='capacity' else 200
    if a.phase=='optimize':
        if a.proof is None:p.error('Optimization requires measured dual capacity proof')
        proof=json.loads(a.proof.read_text())
        if (not proof['passed'] or proof['world_size']!=world or proof['checkpoint_sha256']!=sha(a.checkpoint)
            or proof['terrain_sha256']!=manifest['terrain_sha256'] or proof['num_envs_per_rank']!=a.num_envs
            or proof['batch_size_per_rank']!=a.batch_size):raise ValueError('Dual capacity proof mismatch')
    a.out.mkdir(parents=True,exist_ok=True)
    record_path=a.out/f'rank-{rank}.json'
    with record_path.open('x') as f:json.dump({'status':'initializing','pid':os.getpid()},f)
    record={'status':'initializing','phase':a.phase,'rank':rank,'world_size':world,'physical_gpu':physical,
        'pid':os.getpid(),'source_commit':a.source_commit,'package_commit':a.package_commit,
        'initial_checkpoint_sha256':sha(a.checkpoint),'initial_iteration':initial_iteration,'initial_updates':initial_updates,
        'terrain_sha256':manifest['terrain_sha256'],'num_envs_per_rank':a.num_envs,'batch_size_per_rank':a.batch_size,
        'incoming_checkpoint_world_size':incoming_world,'base_learning_rates':base_learning_rates,
        'cpu_affinity':sorted(os.sched_getaffinity(0)),'authorization':'User: three hours/about12GiB per GPU; latest instruction: use one vacant GPU first',
        'scope':__doc__,'final_validation_approved':False}
    def save():record_path.write_text(json.dumps(record,indent=2,default=str))
    trainer=None;started=time.monotonic()
    def parameter_hashes():
        result={}
        for key,module in [('wm',trainer.wm),('policy',trainer.policy),('critic',trainer.critic)]:
            digest=hashlib.sha256()
            for name,tensor in module.named_parameters():
                digest.update(name.encode());digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            result[key]=digest.hexdigest()
        return result
    try:
        trainer=Trainer(cfg)
        count=round(cfg.env.num_envs*.5)
        clips=(torch.arange(count,device=cfg.device)+rank*count)%trainer.lib.num_clips
        trainer.command.forced_frame[:count]=trainer.lib.clip_start[clips]
        torch.save({'env_ids':torch.arange(count),'clip_ids':clips.cpu(),
            'reference_frames':trainer.command.forced_frame[:count].cpu()},a.out/f'start-reset-rank-{rank}.pt')
        record.update(status='running',configuration=asdict(cfg),pinned_start_environments=count,
            return_sampling_environments=cfg.env.num_envs-count,
            distributed_backend=torch.distributed.get_backend() if torch.distributed.is_initialized() else 'single-process',
            initial_parameter_hashes=parameter_hashes(),
            initial_parameter_checksums={key:float(sum(t.detach().double().sum() for t in module.parameters()))
                for key,module in [('wm',trainer.wm),('policy',trainer.policy),('critic',trainer.critic)]})
        save();trainer.run()
        record.update(status='completed',iteration=trainer.iteration,updates=trainer.num_updates,
            new_updates=trainer.num_updates-initial_updates,
            final_parameter_hashes=parameter_hashes(),
            final_parameter_checksums={key:float(sum(t.detach().double().sum() for t in module.parameters()))
                for key,module in [('wm',trainer.wm),('policy',trainer.policy),('critic',trainer.critic)]})
        if rank==0:
            checkpoint=a.out/'ckpt_latest.pt';ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if not all(torch.isfinite(t).all() for key in ['wm','policy','critic'] for t in ck[key].values() if torch.is_tensor(t)):
                raise RuntimeError('Nonfinite checkpoint')
            record.update(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint))
    finally:
        if trainer is not None:
            trainer.env.close()
            if trainer.writer:trainer.writer.close()
        if record['status']!='completed':record['status']='failed'
        record.update(elapsed_seconds=time.monotonic()-started,
            torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            torch_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        save()


if __name__=='__main__':main()
