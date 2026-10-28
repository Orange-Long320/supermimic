"""Paired CUDA-graph throughput/numerical probe on one frozen real replay.

No trained checkpoint is written. Restore identical policy, critic, optimizers
and RNG seeds before each backend; compare updates on the same replay buffer.
Uses the installed Torch cudagraphs backend, without Inductor kernel compilation.
"""
import argparse
import copy
from dataclasses import is_dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(a.gpu):raise ValueError('Explicit physical GPU required')
    busy=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if busy:raise RuntimeError('GPU occupied: '+busy)
    a.out.mkdir(parents=True,exist_ok=False)
    import torch
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra',check_assets=True)
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.65,0)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    def restore(obj,data):
        for k,v in data.items():
            current=getattr(obj,k)
            if is_dataclass(current):restore(current,v)
            else:setattr(obj,k,tuple(v) if isinstance(current,tuple) else v)
    cfg=TrainConfig();restore(cfg,ck['config']);del ck
    cfg.resume=a.checkpoint;cfg.run_name='trainer';cfg.log_root=a.out
    cfg.wm_trainer.compile=cfg.policy_trainer.compile=None
    record={'status':'initializing','physical_gpu':a.gpu,'pid':os.getpid(),
            'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
            'checkpoint_written':False,'same_replay_for_both_modes':True,'modes':{},'scope':__doc__}
    def save(): (a.out/'progress.json').write_text(json.dumps(record,indent=2))
    save();trainer=None;started=time.monotonic()
    try:
        trainer=Trainer(cfg)
        with torch.no_grad():
            obs,_=trainer.env.reset()
            for _ in range(80):obs,*_=trainer.env.step(trainer.act(obs))
        initial={k:copy.deepcopy(x.state_dict()) for k,x in [('policy',trainer.policy),('critic',trainer.critic),
                 ('policy_opt',trainer.policy_trainer.opt),('critic_opt',trainer.policy_trainer.critic_opt)]}
        step=trainer.policy_trainer.step_fn
        for mode in ['eager','cudagraphs']:
            trainer.policy.load_state_dict(initial['policy']);trainer.critic.load_state_dict(initial['critic'])
            trainer.policy_trainer.opt.load_state_dict(copy.deepcopy(initial['policy_opt']))
            trainer.policy_trainer.critic_opt.load_state_dict(copy.deepcopy(initial['critic_opt']))
            step._fn=step._step if mode=='eager' else torch.compile(step._step,backend='cudagraphs',dynamic=False)
            record.update(status='updating',phase=mode);record['modes'][mode]={'updates':[]};save()
            for i in range(8):
                torch.manual_seed(1000+i);torch.compiler.cudagraph_mark_step_begin()
                torch.cuda.synchronize();start=time.monotonic()
                metrics=trainer.policy_trainer.update(trainer.buffer)
                torch.cuda.synchronize();elapsed=time.monotonic()-start
                if not all(torch.isfinite(t).all() for t in trainer.policy.parameters()):raise RuntimeError('Nonfinite policy')
                record['modes'][mode]['updates'].append({'index':i,'seconds':elapsed,'metrics':metrics});save()
            record['modes'][mode]['median_update_seconds']=sorted(x['seconds'] for x in record['modes'][mode]['updates'][3:])[2]
            if mode=='eager':eager_weights={k:v.detach().cpu().clone() for k,v in trainer.policy.state_dict().items()}
        errors={k:abs(record['modes']['eager']['updates'][0]['metrics'][k]-record['modes']['cudagraphs']['updates'][0]['metrics'][k]) for k in ['imag_objective','policy_grad_norm','imag_return_undisc']}
        difference=max(float((v.detach().cpu()-eager_weights[k]).abs().max()) for k,v in trainer.policy.state_dict().items())
        record.update(status='completed',phase='awaiting_review',first_update_metric_absolute_differences=errors,
                      final_policy_maximum_weight_difference=difference,
                      measured_speedup=record['modes']['eager']['median_update_seconds']/record['modes']['cudagraphs']['median_update_seconds'])
    except Exception as error:
        record.update(status='failed',reason=str(error));raise
    finally:
        if trainer is not None:
            trainer.env.close()
            if trainer.writer:trainer.writer.close()
        record.update(elapsed_seconds=time.monotonic()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                      peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        save()


if __name__=='__main__':main()
