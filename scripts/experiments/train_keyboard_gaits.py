"""Initialize existing keyboard interface from a verified five-gait tracking model.

Run in an isolated snapshot of the keyboard worktree, never in its live checkout.
"""
import argparse
from dataclasses import asdict,is_dataclass,fields
import hashlib,json,os,sys,time,subprocess,threading
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['source','checkpoint','dataset','out']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--phase',choices=['capacity','train'],required=True)
    p.add_argument('--proof',type=Path)
    p.add_argument('--hours',type=float,default=.5)
    p.add_argument('--quiet-stop-cost',action='store_true')
    p.add_argument('--weights-only',action='store_true',help='Import repaired velocity weights with fresh optimizers')
    p.add_argument('--paired-supervision', action=argparse.BooleanOptionalAction, default=True, help='Use tracking replay for persistent teacher alignment')
    a=p.parse_args()
    if not 0 < a.hours <= 24:raise ValueError('Invalid bounded duration')
    sys.path.insert(0,str(a.source/'src'))
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if world!=2 or len(devices)!=2:raise ValueError('Exactly two selected GPUs required')
    gpu=devices[local];os.environ['CUDA_VISIBLE_DEVICES']=gpu;os.environ['SM_GPU_PINNED']='1'
    if hasattr(os,'sched_getaffinity'):
        cores=sorted(os.sched_getaffinity(0));half=len(cores)//2;os.sched_setaffinity(0,cores[local*half:(local+1)*half])
    import torch
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra',check_assets=True)
    from supermimic.config import TrainConfig
    if a.paired_supervision:
        from keyboard_supervision import install
        install()
    if a.quiet_stop_cost:
        from keyboard_supervision import install_quiet_stop_cost
        install_quiet_stop_cost()
    from supermimic.training.loop import Trainer
    from supermimic.training.distributed import broadcast_params
    if Path(sys.prefix).resolve()!=Path('/home/2025-Long.xh/HHD/venvs/supermimic').resolve():raise ValueError('Dedicated server interpreter required')
    manifest=json.loads((a.dataset/'manifest.json').read_text())
    if {c['name'] for c in manifest['clips']}!={'walk','backward','turn_left_inplace','turn_right_inplace','jog'}:raise ValueError('Five-gait dataset required')
    for c in manifest['clips']:
        if sha(a.dataset/(c['name']+'.npz'))!=c['sha256']:raise ValueError('Dataset changed')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.8,0)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    cfg=TrainConfig();omitted=[]
    def restore(obj,d,prefix=''):
        for k,v in d.items():
            if not hasattr(obj,k):
                if k in ['box','terrain_manifest','sole_profile'] and v is None:omitted.append(prefix+k);continue
                raise ValueError('Unknown source configuration '+prefix+k)
            current=getattr(obj,k)
            if is_dataclass(current):restore(current,v,prefix+k+'.')
            else:setattr(obj,k,tuple(v) if isinstance(current,tuple) else v)
    restore(cfg,ck['config'])
    if cfg.robot!='x2_ultra':raise ValueError('Robot mismatch')
    cfg.data.npz_dir=a.dataset;cfg.data.clips=[c['name'] for c in manifest['clips']];cfg.data.manifest=None;cfg.data.library=None;cfg.data.limit=None
    cfg.env.num_envs=8192;cfg.env.replay_capacity=256;cfg.env.anchor_obs=True;cfg.env.episode_length_s=12
    cfg.env.action_noise_std=.1;cfg.env.obs_noise=cfg.env.reset_noise=cfg.env.domain_randomization=True
    cfg.wm_trainer.batch_size=cfg.policy_trainer.batch_size=2048;cfg.wm_trainer.batches_per_update=2
    cfg.wm_trainer.compile=cfg.policy_trainer.compile=None
    cfg.wm_trainer.lr=cfg.policy_trainer.lr=cfg.policy_trainer.critic_lr=2e-4/world**.5
    cfg.velocity.enabled=True;cfg.velocity.fraction=.5
    cfg.velocity.forward=(-.5,.8);cfg.velocity.lateral=(0.,0.);cfg.velocity.yaw=(-.8,.8)
    cfg.velocity.reference_probability=.5;cfg.velocity.standing_probability=.25;cfg.velocity.stop_relabel_fraction=.25
    cfg.run_name=a.out.name if rank==0 else a.out.name+'-rank1-internal';cfg.log_root=a.out.parent
    cfg.schedule.total_iterations=100 if a.phase=='capacity' else 320000
    cfg.schedule.max_hours=.15 if a.phase=='capacity' else a.hours
    cfg.schedule.lr_final_ratio=1.;cfg.schedule.log_interval=20;cfg.schedule.checkpoint_interval=100
    cfg.resume=None;cfg.initialize_from=None;cfg.warmstart_world_model=None
    a.out.mkdir(parents=True,exist_ok=True)
    # Strip only explicitly absent scene-conditioning metadata. Tensor state is
    # unchanged; strict initialization still checks architecture and every key.
    if a.phase=='capacity' and a.weights_only:
        if not ck['config'].get('velocity',{}).get('enabled',False):raise ValueError('Repaired velocity checkpoint required')
        class RepairedTrainer(Trainer):
            def initialize(self,path):
                source=torch.load(path,map_location='cpu',weights_only=False)
                for key in ['history','latent_dim','goal_dim']:
                    if source['config']['policy'][key]!=getattr(self.cfg.policy,key):raise ValueError('Policy architecture mismatch')
                for key,module in [('wm',self.wm),('policy',self.policy),('critic',self.critic)]:
                    module.load_state_dict(source[key],strict=True)
                    if any(not torch.equal(t.cpu(),source[key][n]) for n,t in module.state_dict().items()):raise ValueError('Repaired weight changed')
                for opt in [self.wm_trainer.opt,self.policy_trainer.opt,self.policy_trainer.critic_opt,self.policy_trainer.velocity_opt,self.policy_trainer.velocity_critic_opt]:
                    if opt.state:raise ValueError('Fresh optimizer required')
                print('[INFO] imported repaired model weights exactly; fresh optimizers, velocity critic, replay and counters',flush=True)
        Trainer=RepairedTrainer
        cfg.initialize_from=a.checkpoint
    elif a.phase=='capacity':
        if ck['config'].get('velocity',{}).get('enabled',False):raise ValueError('Tracking checkpoint required')
        compat=dict(ck);compat['config']=dict(ck['config']);wm=dict(ck['config']['world_model']);ss=dict(wm['state_spec'])
        for k in ['box','terrain_manifest']:
            if k in ss:
                if ss[k] is not None:raise ValueError('Unsupported scene-conditioned model')
                ss.pop(k)
        wm['state_spec']=ss;compat['config']['world_model']=wm
        f=a.out/'tracking-compat.pt'
        if rank==0:
            temp=f.with_suffix('.tmp');torch.save(compat,temp);temp.replace(f)
        # Avoid initializing a process group twice: peer waits only for the
        # immutable file, then normal Trainer establishes NCCL.
        if rank!=0:
            while not f.exists():time.sleep(.1)
        cfg.initialize_from=f
    else:
        proof=json.loads(a.proof.read_text())
        if not proof['passed'] or proof['checkpoint_sha256']!=sha(a.checkpoint):raise ValueError('Keyboard capacity proof mismatch')
        if (proof['dataset_manifest_sha256']!=sha(a.dataset/'manifest.json') or proof['world_size']!=world
                or proof['num_envs_per_rank']!=cfg.env.num_envs or proof['batch_size_per_rank']!=cfg.policy_trainer.batch_size):
            raise ValueError('Continuation capacity configuration mismatch')
        cfg.resume=a.checkpoint
    record=dict(status='initializing',phase=a.phase,rank=rank,physical_gpu=gpu,pid=os.getpid(),checkpoint=str(a.checkpoint),checkpoint_sha256=sha(a.checkpoint),dataset_manifest_sha256=sha(a.dataset/'manifest.json'),omitted_unused_metadata=omitted,keyboard_validated=False,paired_supervision=a.paired_supervision,quiet_stop_cost=a.quiet_stop_cost)
    def save():(a.out/f'rank-{rank}.json').write_text(json.dumps(record,indent=2,default=str)+'\n')
    def hashes():
        result={}
        for k,m in [('wm',trainer.wm),('policy',trainer.policy),('critic',trainer.critic),('velocity_critic',trainer.policy_trainer.velocity_critic)]:
            h=hashlib.sha256()
            for n,v in m.state_dict().items():h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
            result[k]=h.hexdigest()
        return result
    save();trainer=None;stop=threading.Event();samples=[];start=time.monotonic()
    def sample():
        while not stop.is_set():
            try:samples.append(int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=5).strip()))
            except (ValueError,subprocess.SubprocessError):pass
            stop.wait(1)
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    try:
        trainer=Trainer(cfg)
        # New velocity encoder and critic have rank-specific seeds. Synchronize
        # them before their first averaged-gradient optimizer update.
        for m in [trainer.wm,trainer.policy,trainer.critic,trainer.policy_trainer.velocity_critic]:broadcast_params(m)
        if a.phase=='capacity':
            for k,m in [('wm',trainer.wm),('critic',trainer.critic)]:
                if any(not torch.equal(t.cpu(),ck[k][n]) for n,t in m.state_dict().items()):raise ValueError('Tracking weights changed on initialization')
            for n,t in ck['policy'].items():
                if not torch.equal(trainer.policy.state_dict()[n].cpu(),t):raise ValueError('Tracking policy not preserved')
        del ck
        record.update(status='running',configuration=asdict(cfg),initial_parameter_hashes=hashes(),training_started_epoch=time.time());save()
        trainer.run();record.update(status='completed',iteration=trainer.iteration,updates=trainer.num_updates,final_parameter_hashes=hashes())
        if rank==0:
            f=a.out/'ckpt_latest.pt';z=torch.load(f,map_location='cpu',weights_only=False)
            if not all(torch.isfinite(v).all() for k in ['wm','policy','critic'] for v in z[k].values() if torch.is_tensor(v)):raise ValueError('Nonfinite checkpoint')
            record.update(checkpoint=str(f),checkpoint_sha256=sha(f))
    finally:
        stop.set();thread.join(timeout=6)
        if trainer is not None:
            trainer.env.close()
            if trainer.writer:trainer.writer.close()
        if record['status']!='completed':record['status']='failed'
        record.update(elapsed_s=time.monotonic()-start,device_peak_mib=max(samples,default=0),torch_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20);save()

if __name__=='__main__':main()
