"""Bounded start-of-motion curriculum after the failed first Omni hour.

Keep the same two clips and scene. Concentrate resets on their true start and
use small execution noise for diverse real transitions. Capacity is measured
before a twenty-minute optimization; independent full-clip evaluation remains
mandatory. No servo protection or physics/acceptance changes are introduced.
"""
import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=3)
    p.add_argument('--phase',choices=['capacity','optimize'],required=True)
    p.add_argument('--proof',type=Path)
    p.add_argument('--source-commit',required=True)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(a.gpu):raise ValueError('Explicit physical GPU required')
    busy=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if busy:raise RuntimeError('GPU occupied: '+busy)
    import torch
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra',check_assets=True)
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    manifest=json.loads((a.dataset/'manifest.json').read_text())
    if not manifest.get('exploratory_training_authorized') or sha(manifest['terrain'])!=manifest['terrain_sha256']:raise ValueError('Dataset is not the authorized immutable pilot')
    for clip in manifest['clips']:
        if sha(a.dataset/'motions'/(clip['name']+'.npz'))!=clip['motion_sha256']:raise ValueError('Motion changed')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.65,0)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    def restore(obj,data):
        for k,v in data.items():
            current=getattr(obj,k)
            if is_dataclass(current):restore(current,v)
            else:setattr(obj,k,tuple(v) if isinstance(current,tuple) else v)
    cfg=TrainConfig();restore(cfg,ck['config'])
    if cfg.robot!='x2_ultra' or Path(cfg.env.terrain_manifest).read_bytes()!=Path(manifest['terrain']).read_bytes():raise ValueError('Wrong checkpoint scene')
    initial_iteration=ck['iteration'];initial_updates=ck['num_updates'];del ck
    cfg.data.npz_dir=a.dataset/'motions';cfg.resume=a.checkpoint
    cfg.run_name=a.out.name;cfg.log_root=a.out.parent
    cfg.env.num_envs=2048;cfg.env.replay_capacity=256
    cfg.env.sampling_mode='start';cfg.env.action_noise_std=.05
    cfg.env.reset_noise=cfg.env.obs_noise=cfg.env.domain_randomization=False
    cfg.policy_trainer.obs_noise=False
    cfg.wm_trainer.batch_size=cfg.policy_trainer.batch_size=512
    cfg.wm_trainer.compile=cfg.policy_trainer.compile=None
    cfg.schedule.total_iterations=initial_iteration+100 if a.phase=='capacity' else 100_000_000
    cfg.schedule.max_hours=.1 if a.phase=='capacity' else 1/3
    cfg.schedule.log_interval=20 if a.phase=='capacity' else 100
    cfg.schedule.checkpoint_interval=100 if a.phase=='capacity' else 200
    if a.phase=='optimize':
        if a.proof is None:p.error('Optimization requires capacity proof')
        proof=json.loads(a.proof.read_text())
        if (proof['status']!='completed' or proof['new_updates']<20 or proof['checkpoint_sha256']!=sha(a.checkpoint)
            or proof['terrain_sha256']!=manifest['terrain_sha256'] or proof['configuration']['env']['num_envs']!=2048
            or proof['configuration']['policy_trainer']['batch_size']!=512):raise ValueError('Capacity proof mismatch')
    a.out.mkdir(parents=True,exist_ok=False)
    record={'phase':a.phase,'status':'initializing','pid':os.getpid(),'physical_gpu':a.gpu,
            'source_commit':a.source_commit,'package_commit':'3bc46dc','configuration':asdict(cfg),
            'terrain_sha256':manifest['terrain_sha256'],'initial_checkpoint_sha256':sha(a.checkpoint),
            'initial_iteration':initial_iteration,'initial_updates':initial_updates,
            'final_validation_approved':False,'authorization':'User: first one hour, then continue optimizing',
            'scope':__doc__}
    def save(): (a.out/'capacity.json').write_text(json.dumps(record,indent=2,default=str))
    save();trainer=None;started=time.monotonic()
    try:
        trainer=Trainer(cfg);record['status']='running';save();trainer.run()
        checkpoint=a.out/'ckpt_latest.pt';ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if not all(torch.isfinite(t).all() for key in ['wm','policy','critic'] for t in ck[key].values() if torch.is_tensor(t)):raise RuntimeError('Nonfinite checkpoint')
        record.update(status='completed',iteration=ck['iteration'],updates=ck['num_updates'],
                      new_updates=ck['num_updates']-initial_updates,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint))
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
