"""Adapt a verified tracking checkpoint to keyboard commands for30minutes."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['source','dataset','checkpoint','out','entry','evaluator','tracking_eval']:p.add_argument('--'+n.replace('_','-'),type=Path,required=True)
    a=p.parse_args();a.out.mkdir(exist_ok=False)
    state=dict(status='initializing',pid=os.getpid(),gpus=[0,2],phases=[])
    def save():(a.out/'controller.json').write_text(json.dumps(state,indent=2)+'\n')
    save();shutil.copy2(__file__,a.out/'controller-as-run.py')
    fixed=a.out/'tracking-source.pt'
    with a.checkpoint.open('rb') as s,fixed.open('xb') as d:shutil.copyfileobj(s,d)
    state['initial_checkpoint_sha256']=hashlib.sha256(fixed.read_bytes()).hexdigest()
    previous=json.loads(a.tracking_eval.read_text())
    if previous['checkpoint_sha256']!=state['initial_checkpoint_sha256'] or len(previous['clips'])!=5 or any(c['failed'] for c in previous['clips']):raise ValueError('Matching completed five-action evaluation required')
    shutil.copy2(a.tracking_eval,a.out/'tracking-evaluation.json');save()
    env=os.environ.copy()
    for k in ['PYTHONPATH','PYTHONHOME','SM_ROBOT_SPEC','SM_SOLE_BOX_CONFIG','SM_GPU_PINNED']:env.pop(k,None)
    env.update(CUDA_VISIBLE_DEVICES='0,2',SM_ROBOT='x2_ultra',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',MUJOCO_GL='egl',XDG_CACHE_HOME='/home/2025-Long.xh/HHD/data/supermimic/cache',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    env['WARP_CACHE_PATH']=env['XDG_CACHE_HOME']+'/warp'
    py='/home/2025-Long.xh/HHD/venvs/supermimic/bin/python'
    common=[py,'-m','torch.distributed.run','--standalone','--nproc-per-node','2',str(a.entry),'--dataset',str(a.dataset)]
    def stage(name,checkpoint,proof=None):
        while True:
            busy={str(g):subprocess.check_output(['nvidia-smi','-i',str(g),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip() for g in [0,2]}
            if not any(busy.values()):break
            state.update(status='waiting_for_idle_gpus',occupants=busy);save();time.sleep(15)
        cmd=common+['--source',str(a.source),'--phase',name,'--checkpoint',str(checkpoint),'--out',str(a.out/name)]
        if proof:cmd+=['--proof',str(proof)]
        entry=dict(name=name,argv=cmd,started_epoch=time.time());state['phases'].append(entry);state['status']=name;save()
        with (a.out/(name+'.log')).open('x') as log:
            child=subprocess.Popen(cmd,cwd=a.source,env=env,stdout=log,stderr=subprocess.STDOUT);entry['pid']=child.pid;save();code=child.wait()
        entry['exit_code']=code;save()
        if code:raise RuntimeError(name+' failed; inspect preserved log')
    stage('capacity',fixed)
    records=[json.loads((a.out/'capacity'/f'rank-{r}.json').read_text()) for r in [0,1]]
    passed=all(r['status']=='completed' and r['updates']>=20 and 0<r['device_peak_mib']<=20000 for r in records)
    passed=passed and records[0]['initial_parameter_hashes']==records[1]['initial_parameter_hashes'] and records[0]['final_parameter_hashes']==records[1]['final_parameter_hashes']
    proof=a.out/'dual-proof.json';proof.write_text(json.dumps(dict(passed=passed,world_size=2,num_envs_per_rank=8192,batch_size_per_rank=2048,checkpoint_sha256=records[0]['checkpoint_sha256'],dataset_manifest_sha256=records[0]['dataset_manifest_sha256'],peaks_mib=[r['device_peak_mib'] for r in records]),indent=2)+'\n')
    if not passed:raise RuntimeError('Dual capacity or synchronization checks failed')
    stage('train',Path(records[0]['checkpoint']),proof)
    state['status']='keyboard_evaluation';save()
    evaluation_env=env.copy();evaluation_env['CUDA_VISIBLE_DEVICES']='0';evaluation_env['SM_ROBOT']='x2_ultra';evaluation_env['PYTHONPATH']=str(a.source/'src')
    command=[py,str(a.evaluator),'--ckpt',str(a.out/'train/ckpt_latest.pt'),'--npz-dir',str(a.dataset),'--out',str(a.out/'eval_66s.json'),'--seconds','66','--num-envs','8']
    with (a.out/'eval_66s.log').open('x') as log:subprocess.run(command,cwd=a.source,env=evaluation_env,stdout=log,stderr=subprocess.STDOUT,check=True)
    state['status']='completed_pending_performance_review';save();(a.out/'controller.exit').write_text('0')

if __name__=='__main__':main()
