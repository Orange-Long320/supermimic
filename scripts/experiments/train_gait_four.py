"""Isolated server-only four-reference exploratory training and measured capacity."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--phase',choices=['prepare','capacity','train'],required=True)
    p.add_argument('--proof',type=Path)
    p.add_argument('--num-envs',type=int,default=4096)
    p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--checkpoint',type=Path)
    a=p.parse_args()
    if Path(sys.prefix).resolve()!=Path('/home/2025-Long.xh/HHD/venvs/supermimic').resolve(): raise ValueError('Dedicated server interpreter required')
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra',check_assets=True)
    import numpy as np
    spec=json.loads(a.spec.read_text())
    if not spec['exploratory_training_authorized'] or len(spec['clips'])!=4: raise ValueError('Expected authorized four-reference spec')
    if a.phase=='prepare':
        from supermimic.data.motion_convert import load_g1_model, G1_JOINT_NAMES
        model=load_g1_model();a.dataset.mkdir(parents=True,exist_ok=False)
        reports=[]
        for c in spec['clips']:
            if sha(c['source'])!=c['sha256']: raise ValueError('Source hash changed: '+c['name'])
            with np.load(c['source']) as m:
                assert m['joint_names'].tolist()==list(G1_JOINT_NAMES)
                assert float(m['fps'][0])==50 and m['joint_pos'].shape[1]==model.nq-7
                for v in m.values():
                    if np.issubdtype(v.dtype,np.number) and not np.isfinite(v).all(): raise ValueError('Nonfinite source')
                q=m['joint_pos'];violation=float(np.maximum(model.jnt_range[1:,0]-q,q-model.jnt_range[1:,1]).clip(0).max())
                if violation>1e-4: raise ValueError('Hard joint violation: '+c['name'])
                reports.append(dict(**c,frames=len(q),hard_limit_violation_rad=violation))
            shutil.copy2(c['source'],a.dataset/(c['name']+'.npz'))
        (a.dataset/'manifest.json').write_text(json.dumps(dict(spec_sha256=sha(a.spec),clips=reports,exploratory_training_authorized=True,physics_validated=False),indent=2)+'\n')
        print(json.dumps(reports,indent=2));return
    manifest=json.loads((a.dataset/'manifest.json').read_text())
    if manifest['spec_sha256']!=sha(a.spec): raise ValueError('Spec changed')
    for c in manifest['clips']:
        if sha(a.dataset/(c['name']+'.npz'))!=c['sha256']: raise ValueError('Dataset changed')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(a.gpu): raise ValueError('Explicit single physical GPU required')
    busy=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if busy: raise RuntimeError('GPU occupied: '+busy)
    import torch
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer
    if torch.cuda.device_count()!=1: raise ValueError('Expected one visible GPU')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.8,0)
    cfg=TrainConfig();cfg.robot='x2_ultra';cfg.run_name=a.out.name;cfg.log_root=a.out.parent
    cfg.data.npz_dir=a.dataset;cfg.data.clips=[c['name'] for c in spec['clips']]
    cfg.env.num_envs=a.num_envs;cfg.env.replay_capacity=256;cfg.env.anchor_obs=True
    cfg.wm_trainer.batch_size=cfg.policy_trainer.batch_size=a.batch_size
    cfg.wm_trainer.compile=cfg.policy_trainer.compile=None
    cfg.resume=a.checkpoint
    initial=0 if a.checkpoint is None else torch.load(a.checkpoint,map_location='cpu',weights_only=False)['iteration']
    cfg.schedule.total_iterations=initial+100 if a.phase=='capacity' else 320000
    cfg.schedule.max_hours=.15 if a.phase=='capacity' else 1.
    cfg.schedule.lr_final_ratio=1.;cfg.schedule.log_interval=20
    cfg.schedule.checkpoint_interval=100 if a.phase=='capacity' else 500
    if a.phase=='train':
        if a.proof is None: p.error('Capacity proof required')
        proof=json.loads(a.proof.read_text())
        if proof['status']!='completed' or proof['updates']<20 or proof['device_peak_mib']>20000 or proof['configuration']['env']['num_envs']!=a.num_envs or proof['configuration']['policy_trainer']['batch_size']!=a.batch_size or proof['dataset_manifest_sha256']!=sha(a.dataset/'manifest.json'): raise ValueError('Capacity proof mismatch')
        if sha(proof['checkpoint'])!=proof['checkpoint_sha256']: raise ValueError('Capacity checkpoint changed')
        cfg.resume=Path(proof['checkpoint'])
    a.out.mkdir(parents=True,exist_ok=False)
    record=dict(status='initializing',phase=a.phase,pid=os.getpid(),gpu=a.gpu,argv=sys.argv,configuration=asdict(cfg),dataset_manifest_sha256=sha(a.dataset/'manifest.json'),python=sys.version,executable=sys.executable,packages={n:importlib.metadata.version(n) for n in ['torch','mujoco','mujoco-warp','warp-lang']},physics_validated=False)
    def save(): (a.out/'result.json').write_text(json.dumps(record,indent=2,default=str)+'\n')
    save();samples=[];stop=threading.Event();trainer=None;started=time.monotonic()
    def monitor():
        while not stop.is_set():
            try: samples.append(int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=5).strip()))
            except (ValueError,subprocess.SubprocessError): pass
            stop.wait(1)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        trainer=Trainer(cfg);record.update(status='running',train_started_epoch=time.time());save()
        trainer.run()
        ck=a.out/'ckpt_latest.pt';state=torch.load(ck,map_location='cpu',weights_only=False)
        if not all(torch.isfinite(v).all() for k in ('wm','policy','critic') for v in state[k].values() if torch.is_tensor(v)): raise ValueError('Nonfinite checkpoint')
        record.update(status='completed',checkpoint=str(ck),checkpoint_sha256=sha(ck),iteration=state['iteration'],updates=state['num_updates'])
    finally:
        stop.set();thread.join(timeout=6)
        if trainer is not None: trainer.env.close();trainer.writer.close()
        if record['status']!='completed':record['status']='failed'
        record.update(elapsed_s=time.monotonic()-started,device_peak_mib=max(samples,default=0),torch_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
        save()

if __name__=='__main__': main()
