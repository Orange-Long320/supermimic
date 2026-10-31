"""Stop at the first nonfinite replay input, loss or gradient and save evidence.

Server-only bounded reproduction. Normal updates, losses and samples are unchanged.
"""
import argparse
from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time


class NonfiniteCapture(RuntimeError):
    pass


class Trace:
    def __init__(self, out):
        self.out=out
        self.trainer=None
        self.owner={}
        self.wm_batch=None
        self.start_batch=None
        self.rollout=None
        self.rng=None
        self.captured=False

    def attach(self, trainer):
        self.trainer=trainer
        for name in ('wm','policy','critic'):
            for key,p in getattr(trainer,name).named_parameters():
                self.owner[id(p)]=name+'.'+key

    def check(self, stage, values):
        import torch
        bad={k:v for k,v in values.items() if torch.is_tensor(v) and not torch.isfinite(v).all()}
        if not bad:
            return
        if self.captured:
            raise NonfiniteCapture(stage)
        self.captured=True
        target=self.out/'first-nonfinite'
        target.mkdir()
        def cpu(value):
            if torch.is_tensor(value):return value.detach().cpu().clone()
            if is_dataclass(value):return {k:cpu(v) for k,v in vars(value).items()}
            if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
            if isinstance(value,(list,tuple)):return [cpu(v) for v in value]
            return value
        stats={}
        for name,v in bad.items():
            good=v[torch.isfinite(v)]
            stats[name]=dict(shape=list(v.shape),nonfinite=int((~torch.isfinite(v)).sum()),
                             finite_abs_max=float(good.abs().max()) if good.numel() else None)
        metadata=dict(stage=stage,iteration=self.trainer.iteration if self.trainer else None,
                      updates=self.trainer.num_updates if self.trainer else None,bad=stats)
        (target/'stage.json').write_text(json.dumps(metadata,indent=2)+'\n')
        torch.save(cpu(dict(values=values,wm_batch=self.wm_batch,start_batch=self.start_batch,
                            rollout=self.rollout,rng=self.rng)),target/'batch.pt')
        if self.trainer:
            self.trainer.save('pre-nonfinite.pt')
        print(json.dumps(metadata),flush=True)
        raise NonfiniteCapture(stage)

    def install(self):
        import torch
        from supermimic.data.replay_buffer import ReplayBuffer
        from supermimic.world_model.wm_trainer import WorldModelTrainer
        from supermimic.imagination import policy_trainer as pi
        trace=self
        add=ReplayBuffer.add
        def checked_add(buffer,env_ids,**values):
            trace.check('replay_write',dict(env_ids=env_ids,**values))
            return add(buffer,env_ids,**values)
        ReplayBuffer.add=checked_add
        loss_on=WorldModelTrainer.loss_on
        def checked_loss(owner,batch):
            trace.wm_batch=batch
            trace.check('wm_input',vars(batch))
            result=loss_on(owner,batch)
            trace.check('wm_loss',dict(loss=result[0],**result[1]))
            return result
        WorldModelTrainer.loss_on=checked_loss
        starts=ReplayBuffer.sample_starts
        def checked_starts(buffer,n,hist):
            batch=starts(buffer,n,hist)
            trace.start_batch=batch
            trace.rng=dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
            trace.check('policy_input',vars(batch))
            return batch
        ReplayBuffer.sample_starts=checked_starts
        objective=pi.policy_objective
        def checked_objective(rollout,*args,**kwargs):
            trace.rollout=rollout
            trace.check('imagined_rollout',{k:v for k,v in vars(rollout).items() if torch.is_tensor(v)})
            value=objective(rollout,*args,**kwargs)
            trace.check('policy_objective',{'objective':value})
            return value
        pi.policy_objective=checked_objective
        update=pi.PolicyTrainer.update
        def checked_update(owner,*args,**kwargs):
            try:return update(owner,*args,**kwargs)
            finally:trace.rollout=None
        pi.PolicyTrainer.update=checked_update
        clip=torch.nn.utils.clip_grad_norm_
        def checked_clip(parameters,*args,**kwargs):
            params=[parameters] if torch.is_tensor(parameters) else list(parameters)
            trace.check('raw_gradient',{trace.owner.get(id(p),str(i)):p.grad
                                        for i,p in enumerate(params) if p.grad is not None})
            return clip(params,*args,**kwargs)
        torch.nn.utils.clip_grad_norm_=checked_clip


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','dataset','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--seconds',type=float,default=600)
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--compiled',action='store_true')
    args=parser.parse_args()
    if args.out.exists() or args.seconds<=0 or args.steps<=40:
        raise ValueError('New output and bounded diagnostic after replay warmup required')
    if Path(sys.prefix).resolve()!=Path('/home/2025-Long.xh/HHD/venvs/supermimic').resolve():
        raise ValueError('Dedicated server interpreter required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.gpu):
        raise ValueError('Explicit single GPU required')
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra',check_assets=True)
    import torch
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer
    torch.set_num_threads(2)
    total,free=map(int,subprocess.check_output(['nvidia-smi','-i',str(args.gpu),
        '--query-gpu=memory.total,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(','))
    if free<8192:raise ValueError('Insufficient initial shared memory headroom')
    torch.cuda.set_per_process_memory_fraction(5.5*1024/total,0)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if not all(torch.isfinite(v).all() for key in ('wm','policy','critic') for v in ck[key].values() if torch.is_tensor(v)):
        raise ValueError('Diagnostic source already nonfinite')
    def restore(obj,values):
        for k,v in values.items():
            current=getattr(obj,k)
            if is_dataclass(current):restore(current,v)
            else:setattr(obj,k,tuple(v) if isinstance(current,tuple) else v)
    cfg=TrainConfig();restore(cfg,ck['config'])
    if cfg.robot!='x2_ultra' or cfg.env.anchor_obs or cfg.policy.goal_dim!=720:
        raise ValueError('Expected the unchanged native no-anchor candidate')
    cfg.data.npz_dir=args.dataset
    cfg.resume=args.checkpoint;cfg.warmstart_world_model=None
    cfg.log_root=args.out.parent;cfg.run_name=args.out.name
    cfg.schedule.total_iterations=int(ck['iteration'])+args.steps
    cfg.schedule.max_hours=args.seconds/3600
    cfg.schedule.checkpoint_interval=500
    cfg.wm_trainer.compile=cfg.policy_trainer.compile='default' if args.compiled else None
    args.out.mkdir(parents=True)
    record=dict(status='initializing',pid=os.getpid(),gpu=args.gpu,source_iteration=ck['iteration'],
                argv=sys.argv,compiled=args.compiled,torch_memory_gib=5.5,free_memory_interrupt_enabled=False)
    del ck
    save=lambda:(args.out/'diagnostic.json').write_text(json.dumps(record,indent=2)+'\n')
    save();trace=Trace(args.out);trace.install();trainer=None;started=time.time()
    try:
        trainer=Trainer(cfg);trace.attach(trainer)
        record.update(status='running',training_started_epoch=time.time());save()
        trainer.run();record.update(status='completed_without_reproduction',iteration=trainer.iteration,updates=trainer.num_updates)
    except NonfiniteCapture as error:
        record.update(status='captured_nonfinite',stage=str(error),iteration=trainer.iteration if trainer else None)
    finally:
        if trainer:
            trainer.env.close()
            if trainer.writer:trainer.writer.close()
        record.update(elapsed_s=time.time()-started,torch_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        if record['status'] in ('initializing','running'):record['status']='failed'
        save()


if __name__=='__main__':main()
