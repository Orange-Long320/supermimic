"""Bounded update throughput probe; never publishes a trained checkpoint.

Use the same frozen checkpoint, scene and replay collection in separate eager
and compiled processes. Only the existing ImagStep compilation option changes.
"""
import argparse
from dataclasses import is_dataclass
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['eager','default'],required=True)
    p.add_argument('--gpu',type=int,default=3)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(a.gpu):raise ValueError('Explicit physical GPU required')
    a.out.mkdir(parents=True,exist_ok=False)
    import torch
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra',check_assets=True)
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.65,0)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    def restore(obj,data):
        for k,v in data.items():
            current=getattr(obj,k)
            if is_dataclass(current):restore(current,v)
            else:setattr(obj,k,tuple(v) if isinstance(current,tuple) else v)
    cfg=TrainConfig();restore(cfg,ck['config']);del ck
    cfg.resume=a.checkpoint;cfg.run_name='trainer';cfg.log_root=a.out
    cfg.wm_trainer.compile=None
    cfg.policy_trainer.compile=None if a.mode=='eager' else 'default'
    record={'mode':a.mode,'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
            'physical_gpu':a.gpu,'num_envs':cfg.env.num_envs,'batch_size':cfg.policy_trainer.batch_size,
            'status':'initializing','pid':os.getpid(),'checkpoint_written':False,'updates':[]}
    def save(): (a.out/'progress.json').write_text(json.dumps(record,indent=2))
    save();trainer=None;started=time.monotonic()
    try:
        trainer=Trainer(cfg)
        with torch.no_grad():
            obs,_=trainer.env.reset()
            for _ in range(80):obs,*_=trainer.env.step(trainer.act(obs))
        torch.cuda.synchronize();record['status']='updating';save()
        for i in range(8):
            torch.manual_seed(1000+i)
            torch.cuda.synchronize();start=time.monotonic()
            metrics=trainer.policy_trainer.update(trainer.buffer)
            torch.cuda.synchronize()
            if not all(torch.isfinite(t).all() for t in trainer.policy.parameters()):raise RuntimeError('Nonfinite weights')
            record['updates'].append({'index':i,'seconds':time.monotonic()-start,'metrics':metrics});save()
        record['median_update_seconds']=sorted(r['seconds'] for r in record['updates'][3:])[2]
        record['status']='completed'
    finally:
        if trainer is not None:
            trainer.env.close()
            if trainer.writer:trainer.writer.close()
        record['elapsed_seconds']=time.monotonic()-started
        record['peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
        record['peak_reserved_mib']=torch.cuda.max_memory_reserved()/2**20
        save()


if __name__=='__main__':main()
