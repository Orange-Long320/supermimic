"""Bounded exploratory motion tracking transfer from LAFAN70000; no PPO."""
import argparse
from dataclasses import asdict,is_dataclass
import hashlib,json,os,shutil,subprocess,sys,threading,time
from pathlib import Path


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('prepare','capacity','train'),required=True)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--gpu',type=int,default=0);p.add_argument('--proof',type=Path)
    a=p.parse_args()
    if Path(sys.prefix).resolve()!=Path('/home/2025-Long.xh/HHD/venvs/supermimic').resolve():raise ValueError('Dedicated server environment required')
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra',check_assets=True)
    if a.phase=='prepare':
        import numpy as np
        from supermimic.data.motion_convert import load_g1_model
        m=load_g1_model();root=Path('/home/2025-Long.xh/HHD/data/supermimic')
        base=root/'data/video/gait_five_train_20261029_r1';existing=json.loads((base/'manifest.json').read_text())
        candidates=[root/'logs/skill_user_pose_fast06_20261030_r1']
        if a.dataset.exists():raise ValueError('Dataset exists')
        a.dataset.mkdir(parents=True);clips=[]
        for c in existing['clips']:
            source=base/(c['name']+'.npz')
            if sha(source)!=c['sha256']:raise ValueError('Source hash mismatch')
            shutil.copy2(source,a.dataset/(c['name']+'.npz'));clips.append({'name':c['name'],'source':str(source),'sha256':sha(source),'type':'prior_gait'})
        for i,folder in enumerate(candidates):
            meta=json.loads((folder/'manifest.json').read_text());source=folder/'natural-stop.npz'
            if sha(source)!=meta['stop_sha256'] or meta['stop_frames']!=30 or not meta['entry_velocity_matched']:
                raise ValueError('Confirmed 0.6s reference provenance mismatch')
            name='confirmed_stop';stand=int(meta['standing_frame']);length=stand+200
            with np.load(source,allow_pickle=False) as z:
                arrays={k:z[k].copy() for k in z.files}
            if float(arrays['fps'][0])!=50 or stand>=len(arrays['joint_pos']):raise ValueError('Invalid reference length or fps')
            frame_keys=('joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w')
            for key in frame_keys:
                values=arrays[key]
                if np.max(np.abs(values[stand:]-values[stand]))>1e-5:raise ValueError('Expected stationary final reference: '+key)
                arrays[key]=np.concatenate([values[:stand],np.repeat(values[stand:stand+1],200,axis=0)],axis=0)
            q=arrays['joint_pos'];hard=float(np.maximum(np.maximum(m.jnt_range[1:,0]-q,q-m.jnt_range[1:,1]),0).max())
            if hard>1e-4 or not all(np.isfinite(v).all() for v in arrays.values() if np.issubdtype(v.dtype,np.number)):raise ValueError('Input check failed')
            out=a.dataset/(name+'.npz');np.savez(out,**arrays)
            clips.append({'name':name,'source':str(source),'source_sha256':sha(source),'sha256':sha(out),'frames':length,'brake_frames':30,'stand_frames':200,'hard_limit_excess_rad':hard,'type':'user_confirmed_exploratory_stop'})
        manifest={'clips':clips,'scope':'User requested exploratory training; reference dynamics and stop acceptance not validated','initial_checkpoint_sha256':sha(a.checkpoint)}
        (a.dataset/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2));return
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(a.gpu):raise ValueError('Explicit physical GPU required')
    busy=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if busy:raise ValueError('Selected GPU is occupied: '+busy)
    manifest=json.loads((a.dataset/'manifest.json').read_text())
    for c in manifest['clips']:
        if sha(a.dataset/(c['name']+'.npz'))!=c['sha256']:raise ValueError('Dataset changed')
    import torch
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer
    if torch.cuda.device_count()!=1:raise ValueError('Expected one visible GPU')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.8,0)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if a.phase=='capacity' and (ck['iteration']!=70000 or sha(a.checkpoint)!=manifest['initial_checkpoint_sha256']):raise ValueError('Expected recorded LAFAN70000')
    def restore(obj,d):
        for k,v in d.items():
            current=getattr(obj,k)
            if is_dataclass(current):restore(current,v)
            else:setattr(obj,k,tuple(v) if isinstance(current,tuple) else v)
    cfg=TrainConfig();restore(cfg,ck['config'])
    if cfg.robot!='x2_ultra' or cfg.env.box or cfg.env.terrain_manifest or cfg.env.sole_profile:raise ValueError('Unexpected checkpoint configuration')
    cfg.data.npz_dir=a.dataset;cfg.data.clips=[c['name'] for c in manifest['clips']];cfg.data.manifest=None;cfg.data.library=None;cfg.data.limit=None
    cfg.env.num_envs=4096;cfg.env.replay_capacity=256;cfg.env.anchor_obs=True;cfg.env.episode_length_s=10
    cfg.env.action_noise_std=.1;cfg.env.reset_noise=cfg.env.obs_noise=cfg.env.domain_randomization=True;cfg.env.sampling_mode='return';cfg.policy_trainer.obs_noise=True
    cfg.wm_trainer.batch_size=cfg.policy_trainer.batch_size=1024;cfg.wm_trainer.batches_per_update=10
    cfg.wm_trainer.compile=cfg.policy_trainer.compile=None
    cfg.wm_trainer.lr=cfg.policy_trainer.lr=cfg.policy_trainer.critic_lr=2e-4
    cfg.resume=None;cfg.warmstart_world_model=None
    if a.phase=='train':
        if a.proof is None:raise ValueError('Capacity proof required')
        proof=json.loads(a.proof.read_text())
        if proof['status']!='completed' or proof['updates']<1 or proof['device_peak_mib']>20000 or proof['dataset_sha256']!=sha(a.dataset/'manifest.json') or proof['checkpoint_sha256']!=sha(a.checkpoint):raise ValueError('Capacity proof mismatch')
        cfg.resume=a.checkpoint
    cfg.run_name=a.out.name;cfg.log_root=a.out.parent;cfg.schedule.total_iterations=100 if a.phase=='capacity' else 320000
    cfg.schedule.max_hours=.15 if a.phase=='capacity' else 1.;cfg.schedule.lr_final_ratio=1.;cfg.schedule.log_interval=20;cfg.schedule.checkpoint_interval=100 if a.phase=='capacity' else 500
    a.out.mkdir(parents=True,exist_ok=False)
    record={'status':'initializing','phase':a.phase,'pid':os.getpid(),'gpu':a.gpu,'argv':sys.argv,'dataset_sha256':sha(a.dataset/'manifest.json'),'checkpoint_input':str(a.checkpoint),'checkpoint_input_sha256':sha(a.checkpoint),'source_iteration':ck['iteration'],'configuration':asdict(cfg),'weights_only_transfer':a.phase=='capacity','physics_validated':False,'python':sys.executable,'torch':torch.__version__,'cuda_runtime':torch.version.cuda}
    def save():(a.out/'result.json').write_text(json.dumps(record,indent=2,default=str)+'\n')
    save();trainer=None;stop=threading.Event();samples=[];start=time.monotonic()
    def monitor():
        while not stop.is_set():
            try:samples.append(int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=5).strip()))
            except (ValueError,subprocess.SubprocessError):pass
            stop.wait(1)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        trainer=Trainer(cfg)
        if a.phase=='capacity':
            for key,module in [('wm',trainer.wm),('policy',trainer.policy),('critic',trainer.critic)]:module.load_state_dict(ck[key],strict=True)
            if trainer.iteration or trainer.num_updates:raise ValueError('Expected fresh counters')
        del ck
        record.update(status='running',training_started_epoch=time.time());save();trainer.run()
        path=a.out/'ckpt_latest.pt';z=torch.load(path,map_location='cpu',weights_only=False)
        if not all(torch.isfinite(v).all() for key in ('wm','policy','critic') for v in z[key].values() if torch.is_tensor(v)):raise ValueError('Non-finite checkpoint')
        record.update(status='completed',iteration=z['iteration'],updates=z['num_updates'],checkpoint=str(path),checkpoint_sha256=sha(path))
    finally:
        stop.set();thread.join(timeout=6)
        if trainer is not None:
            trainer.env.close()
            if trainer.writer:trainer.writer.close()
        if record['status']!='completed':record['status']='failed'
        record.update(elapsed_s=time.monotonic()-start,device_peak_mib=max(samples,default=0),torch_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        save()
if __name__=='__main__':main()
