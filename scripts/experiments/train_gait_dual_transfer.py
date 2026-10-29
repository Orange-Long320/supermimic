"""Two-rank gait finetuning: exact LAFAN70000 weights, fresh transfer state."""
import argparse
from dataclasses import asdict,is_dataclass
import hashlib,json,os,sys,time,threading,subprocess
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['checkpoint','dataset','out']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--phase',choices=['capacity','train'],required=True)
    p.add_argument('--proof',type=Path)
    a=p.parse_args()
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if world!=2 or len(devices)!=2:raise ValueError('Exactly two physical GPUs required')
    physical=devices[local];os.environ['CUDA_VISIBLE_DEVICES']=physical;os.environ['SM_GPU_PINNED']='1'
    if hasattr(os,'sched_getaffinity'):
        cores=sorted(os.sched_getaffinity(0));half=len(cores)//2;os.sched_setaffinity(0,cores[local*half:(local+1)*half])
    import torch
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra',check_assets=True)
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer
    if Path(sys.prefix).resolve()!=Path('/home/2025-Long.xh/HHD/venvs/supermimic').resolve():raise ValueError('Dedicated interpreter required')
    manifest=json.loads((a.dataset/'manifest.json').read_text())
    if not manifest.get('exploratory_training_authorized') or {c['name'] for c in manifest['clips']}!={'walk','backward','turn_left_inplace','jog'}:raise ValueError('Authorized four-gait dataset required')
    for c in manifest['clips']:
        if sha(a.dataset/(c['name']+'.npz'))!=c['sha256']:raise ValueError('Motion changed')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.8,0)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    def restore(obj,d):
        for k,v in d.items():
            current=getattr(obj,k)
            if is_dataclass(current):restore(current,v)
            else:setattr(obj,k,tuple(v) if isinstance(current,tuple) else v)
    cfg=TrainConfig();restore(cfg,ck['config'])
    if cfg.robot!='x2_ultra' or cfg.env.box or cfg.env.terrain_manifest or cfg.env.sole_profile:raise ValueError('Expected unchanged flat-ground X2 checkpoint')
    if a.phase=='capacity' and ck['iteration']!=70000:raise ValueError('Expected original LAFAN70000')
    if a.phase=='train':
        proof=json.loads(a.proof.read_text())
        if not proof['passed'] or proof['checkpoint_sha256']!=sha(a.checkpoint) or proof['dataset_manifest_sha256']!=sha(a.dataset/'manifest.json'):raise ValueError('Dual proof mismatch')
    cfg.data.npz_dir=a.dataset;cfg.data.clips=['walk','backward','turn_left_inplace','jog'];cfg.data.manifest=None;cfg.data.library=None;cfg.data.limit=None
    cfg.env.num_envs=8192;cfg.env.replay_capacity=256;cfg.env.anchor_obs=True;cfg.env.episode_length_s=10
    cfg.env.action_noise_std=.1;cfg.env.reset_noise=cfg.env.obs_noise=cfg.env.domain_randomization=True
    cfg.env.sampling_mode='return';cfg.policy_trainer.obs_noise=True
    cfg.wm_trainer.batch_size=cfg.policy_trainer.batch_size=2048
    cfg.wm_trainer.batches_per_update=10
    cfg.wm_trainer.compile=cfg.policy_trainer.compile=None
    # Trainer applies sqrt(world); retain actual2e-4 per optimizer without
    # inheriting the pretraining optimizer's terminal decayed LR.
    cfg.wm_trainer.lr=cfg.policy_trainer.lr=cfg.policy_trainer.critic_lr=2e-4/world**.5
    cfg.resume=None if a.phase=='capacity' else a.checkpoint;cfg.warmstart_world_model=None
    cfg.run_name=a.out.name;cfg.log_root=a.out.parent
    cfg.schedule.total_iterations=100 if a.phase=='capacity' else 320000
    cfg.schedule.max_hours=.15 if a.phase=='capacity' else 1.
    cfg.schedule.lr_final_ratio=1.;cfg.schedule.log_interval=20;cfg.schedule.checkpoint_interval=100 if a.phase=='capacity' else 500
    a.out.mkdir(parents=True,exist_ok=True);record_path=a.out/f'rank-{rank}.json'
    with record_path.open('x') as f:json.dump({'status':'initializing'},f)
    record=dict(status='initializing',phase=a.phase,pid=os.getpid(),rank=rank,world_size=world,physical_gpu=physical,source_checkpoint=str(a.checkpoint),source_checkpoint_sha256=sha(a.checkpoint),source_iteration=ck['iteration'],source_updates=ck['num_updates'],weights_only_transfer=a.phase=='capacity',dataset_manifest_sha256=sha(a.dataset/'manifest.json'),num_envs_per_rank=8192,batch_size_per_rank=2048,physics_validated=False)
    def save():record_path.write_text(json.dumps(record,indent=2,default=str)+'\n')
    def hashes():
        values={}
        for key,m in [('wm',trainer.wm),('policy',trainer.policy),('critic',trainer.critic)]:
            h=hashlib.sha256()
            for name,t in m.state_dict().items():h.update(name.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
            values[key]=h.hexdigest()
        return values
    save();trainer=None;stop=threading.Event();samples=[];start=time.monotonic()
    def sample():
        while not stop.is_set():
            try:samples.append(int(subprocess.check_output(['nvidia-smi','-i',physical,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=5).strip()))
            except (ValueError,subprocess.SubprocessError):pass
            stop.wait(1)
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    try:
        trainer=Trainer(cfg)
        if a.phase=='capacity':
            for key,m in [('wm',trainer.wm),('policy',trainer.policy),('critic',trainer.critic)]:m.load_state_dict(ck[key],strict=True)
            if trainer.iteration!=0 or trainer.num_updates!=0 or any(len(o.state) for o in [trainer.wm_trainer.opt,trainer.policy_trainer.opt,trainer.policy_trainer.critic_opt]):raise ValueError('Transfer state was not fresh')
        del ck
        record.update(status='running',configuration=asdict(cfg),initial_parameter_hashes=hashes(),training_started_epoch=time.time(),backend=torch.distributed.get_backend());save()
        trainer.run()
        record.update(status='completed',iteration=trainer.iteration,updates=trainer.num_updates,final_parameter_hashes=hashes())
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
        record.update(elapsed_s=time.monotonic()-start,device_peak_mib=max(samples,default=0),torch_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        save()

if __name__=='__main__':main()
