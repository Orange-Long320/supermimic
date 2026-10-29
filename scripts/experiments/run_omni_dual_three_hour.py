"""Queue measured one/two-GPU Omni continuation, then evaluate actual full tasks.

Only vacant explicitly selected GPUs are used. Capacity and training own one
distributed policy, not two independent trials. Original physics stays fixed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','dataset','initial-checkpoint']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-checkpoint-sha256',required=True)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--gpus',type=int,nargs='+',default=[3,2])
    p.add_argument('--minutes',type=float,default=180.)
    p.add_argument('--memory-budget-mib',type=int,default=12800)
    p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--torch-memory-fraction',type=float,default=.4)
    p.add_argument('--fallback-proof',type=Path,help='Measured same-checkpoint capacity proof to resume if this larger-batch trial fails')
    p.add_argument('--fallback-batch-size',type=int,default=1024)
    p.add_argument('--previous-training-seconds',type=float,default=0.)
    p.add_argument('--include-capacity-in-budget',action='store_true',help='Count this continuation capacity fit within the same original wall-clock training budget')
    p.add_argument('--queue-seconds',type=int,default=3600)
    a=p.parse_args()
    if (len(a.gpus) not in [1,2] or len(set(a.gpus))!=len(a.gpus) or not 0<a.minutes<=180
        or not 0<a.memory_budget_mib<=22000 or a.batch_size<1 or a.previous_training_seconds<0
        or not 0<a.torch_memory_fraction<=.8):
        p.error('One/two distinct GPUs, <=3hours, bounded batch and memory with hardware headroom required')
    r=a.run;s=r/'source';py='/home/2025-Long.xh/HHD/venvs/supermimic/bin/python'
    if (r/'progress.json').exists():p.error('Run already started')
    (r/'pid').write_text(str(os.getpid()))
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2*len(a.gpus):])
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    env=os.environ.copy();env.update(SM_ROBOT='x2_ultra',PYTHONPATH=str(s/'src'),OMP_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',MUJOCO_GL='egl')
    for key in ['PYTHONHOME','SM_ROBOT_SPEC','SM_SOLE_BOX_CONFIG','SM_GPU_PINNED']:env.pop(key,None)
    def progress(**values):
        (r/'progress.json').write_text(json.dumps(dict(updated_at=time.time(),**values),indent=2))
    def occupants(gpu):
        return subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    def wait_free(gpus,seconds):
        deadline=time.monotonic()+seconds
        while True:
            busy={str(gpu):occupants(gpu) for gpu in gpus}
            if not any(busy.values()):return
            progress(status='waiting_for_vacant_gpus',gpus=gpus,occupants=busy)
            if time.monotonic()>deadline:raise TimeoutError('Selected GPUs remain occupied; no other processes changed')
            time.sleep(15)
    def stage(name,command,gpus,timeout):
        if gpus:wait_free(gpus,30)
        child_env=env.copy();child_env['CUDA_VISIBLE_DEVICES']=','.join(map(str,gpus))
        if len(gpus)==1:child_env['MUJOCO_EGL_DEVICE_ID']=str(gpus[0])
        (r/(name+'-command.json')).write_text(json.dumps(command,indent=2))
        stop=threading.Event();peaks={str(g):0 for g in gpus};samples=[];over=[False]
        with (r/(name+'.log')).open('w') as log:
            child=subprocess.Popen(command,cwd=s,env=child_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            (r/(name+'-pid')).write_text(str(child.pid));progress(status='running',phase=name,pid=child.pid,gpus=gpus)
            def monitor():
                consecutive=0
                with (r/(name+'-memory.jsonl')).open('w') as memory_log:
                    while not stop.is_set():
                        try:
                            record={'time':time.time()}
                            for gpu in gpus:
                                value=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=5).strip())
                                peaks[str(gpu)]=max(peaks[str(gpu)],value);record[str(gpu)]=value
                            samples.append(record);memory_log.write(json.dumps(record)+'\n');memory_log.flush()
                            consecutive=consecutive+1 if any(record[str(g)]>a.memory_budget_mib for g in gpus) else 0
                            if consecutive>=3:
                                over[0]=True;os.killpg(child.pid,signal.SIGTERM);return
                        except (subprocess.SubprocessError,ProcessLookupError,ValueError):pass
                        stop.wait(1)
            thread=threading.Thread(target=monitor,daemon=True) if gpus else None
            if thread:thread.start()
            try:code=child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                code=124
            stop.set()
            if thread:thread.join(timeout=8)
        (r/(name+'-exit')).write_text(str(code));(r/(name+'-memory-peak.json')).write_text(json.dumps({
            'device_peak_mib':peaks,'sample_count':len(samples),'over_budget_stopped':over[0],
            'scope':'Whole exclusive selected GPUs at1s; may miss shorter peaks'},indent=2))
        if code or over[0]:raise RuntimeError(f'{name} failed, exit={code}, memory_stop={over[0]}')
        return peaks
    def rank_pair(directory):
        ranks=[json.loads((directory/f'rank-{i}.json').read_text()) for i in range(len(a.gpus))]
        if any(x['status']!='completed' or x['new_updates']<20 for x in ranks):raise RuntimeError('Both ranks must finish real parameter updates')
        for key in ['iteration','updates','initial_parameter_hashes','final_parameter_hashes','base_learning_rates']:
            if any(ranks[0][key]!=x[key] for x in ranks[1:]):raise RuntimeError('Rank coherence mismatch: '+key)
        if ranks[0]['final_parameter_hashes']==ranks[0]['initial_parameter_hashes']:raise RuntimeError('No parameter change')
        if ranks[0]['checkpoint_sha256']!=sha(ranks[0]['checkpoint']):raise RuntimeError('Final checkpoint changed')
        return ranks
    try:
        manifest=json.loads((a.dataset/'manifest.json').read_text())
        if sha(manifest['terrain'])!=manifest['terrain_sha256']:raise ValueError('Terrain changed')
        if sha(a.initial_checkpoint)!=a.expected_checkpoint_sha256:raise ValueError('Unexpected initial weights')
        initial=r/'initial-weights.pt';shutil.copyfile(a.initial_checkpoint,initial)
        if sha(initial)!=a.expected_checkpoint_sha256:raise ValueError('Pinned weights changed')
        (r/'source-commit.txt').write_text(a.source_commit+'; package7568607 private copy\n')
        (r/'resources.txt').write_text(f'{len(a.gpus)}rank shared policy on explicit GPUs{a.gpus};2048env/{a.batch_size}batch per rank;2CPUcores/2threads per rank;device budget{a.memory_budget_mib}MiB each;original budget{a.minutes}minutes, subtract previous{a.previous_training_seconds}s, capacity charged={a.include_capacity_in_budget};50% source0/50% return;original physics unchanged. User requested continuation despite unresolved reference dynamics.\n')
        wait_free(a.gpus,a.queue_seconds)
        (r/'gpu-preflight.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
        (r/'process-preflight.txt').write_text(subprocess.check_output(['ps','-eo','pid,user,args'],text=True))
        launcher=[py,'-m','torch.distributed.run','--standalone','--nproc-per-node','2'] if len(a.gpus)==2 else [py]
        common=launcher+[
            str(s/'scripts/experiments/train_omni_dual.py'),'--dataset',str(a.dataset),
            '--source-commit',a.source_commit,'--package-commit','7568607','--batch-size',str(a.batch_size),
            '--torch-memory-fraction',str(a.torch_memory_fraction)]
        fallback_used=False
        try:
            peak=stage('capacity',common+['--checkpoint',str(initial),'--phase','capacity','--minutes',str(a.minutes),'--out',str(r/'capacity')],a.gpus,600)
            ranks=rank_pair(r/'capacity')
            if any(not 0<v<=a.memory_budget_mib for v in peak.values()):raise RuntimeError('Measured per-card capacity budget failed')
            proof=dict(passed=True,world_size=len(a.gpus),num_envs_per_rank=2048,batch_size_per_rank=a.batch_size,
                checkpoint_sha256=ranks[0]['checkpoint_sha256'],terrain_sha256=manifest['terrain_sha256'],
                device_peaks_mib=peak,rank_records=ranks)
            fit_checkpoint=ranks[0]['checkpoint']
            charged_capacity=ranks[0]['training_loop_seconds'] if a.include_capacity_in_budget else 0.
        except Exception as trial_error:
            if a.fallback_proof is None:raise
            proof=json.loads(a.fallback_proof.read_text())
            if (not proof['passed'] or proof['world_size']!=len(a.gpus) or proof['checkpoint_sha256']!=sha(initial)
                or proof['terrain_sha256']!=manifest['terrain_sha256'] or proof['num_envs_per_rank']!=2048
                or proof['batch_size_per_rank']!=a.fallback_batch_size):raise RuntimeError('Fallback proof mismatch') from trial_error
            failed_path=r/'capacity/rank-0.json'
            failed=json.loads(failed_path.read_text()) if failed_path.exists() else {}
            actual_trial_seconds=failed.get('training_loop_seconds',max(0.,time.time()-failed['training_started_at']) if 'training_started_at' in failed else 0.)
            charged_capacity=actual_trial_seconds if a.include_capacity_in_budget else 0.
            failure=dict(requested_batch_size=a.batch_size,trial_passed=False,reason=str(trial_error),
                fallback_batch_size=a.fallback_batch_size,fallback_proof=str(a.fallback_proof),
                actual_trial_training_seconds=actual_trial_seconds,charged_trial_training_seconds=charged_capacity,
                source_checkpoint_sha256=sha(initial))
            (r/'larger-batch-trial.json').write_text(json.dumps(failure,indent=2))
            common[common.index('--batch-size')+1]=str(a.fallback_batch_size)
            a.batch_size=a.fallback_batch_size;fit_checkpoint=str(initial);fallback_used=True
            proof=dict(proof,reused_from=str(a.fallback_proof),larger_batch_trial_passed=False)
        (r/'dual-capacity-proof.json').write_text(json.dumps(proof,indent=2))
        (r/'effective-training-settings.json').write_text(json.dumps(dict(batch_size=a.batch_size,
            fallback_used=fallback_used,memory_budget_mib=a.memory_budget_mib),indent=2))
        remaining=a.minutes*60-a.previous_training_seconds-charged_capacity
        (r/'training-budget.json').write_text(json.dumps(dict(original_budget_seconds=a.minutes*60,
            previous_training_seconds=a.previous_training_seconds,charged_capacity_seconds=charged_capacity,
            remaining_formal_training_seconds=max(0.,remaining)),indent=2))
        if remaining<=0:raise RuntimeError('Original training budget consumed; no additional long fit launched')
        stage('train_3h',common+['--checkpoint',fit_checkpoint,'--phase','optimize',
            '--minutes',str(remaining/60),'--proof',str(r/'dual-capacity-proof.json'),
            '--out',str(r/'train_3h')],a.gpus,remaining+600)
        final=rank_pair(r/'train_3h');checkpoint=final[0]['checkpoint'];gpu=a.gpus[0]
        stage('benchmark',[py,str(s/'scripts/evaluation/eval_benchmark.py'),'--ckpt',checkpoint,
            '--npz-dir',str(a.dataset/'motions'),'--num-envs','3','--device','cuda:0','--out',str(r/'benchmark.json')],[gpu],300)
        reports=[];(r/'videos').mkdir()
        for clip in manifest['clips']:
            name=clip['name'];trace=r/('capture-'+name);audit=r/('task-audit-'+name)
            stage('capture-'+name,[py,str(s/'scripts/evaluation/record_policy_states.py'),'--ckpt',checkpoint,
                '--source',str(a.dataset/'motions'/(name+'.npz')),'--out',str(trace),'--seconds','18',
                '--hold-last-reference','--record-contact-forces','--device','cuda:0'],[gpu],360)
            stage('task-audit-'+name,[py,str(s/'scripts/evaluation/audit_omni_tasks.py'),'--trace',str(trace),
                '--dataset',str(a.dataset),'--clip',name,'--out',str(audit)],[],600)
            reports.append(json.loads((audit/'report.json').read_text()))
            stage('video-'+name,[py,str(s/'scripts/evaluation/render_policy.py'),'--robot','x2_ultra','--ckpt',checkpoint,
                '--npz-dir',str(a.dataset/'motions'),'--clip',name,'--seconds','18','--no-reference','--no-label',
                '--out',str(r/'videos'/(name+'.mp4'))],[gpu],300)
        progress(status='completed',checkpoint_sha256=final[0]['checkpoint_sha256'],reports=reports,
            actual_task_checks_passed=all(x['complete_task_verified'] for x in reports),
            final_validation_approved=False,manual_video_review_pending=True)
        (r/'exit-code').write_text('0')
    except Exception as error:
        progress(status='failed',reason=str(error));(r/'exit-code').write_text('1');raise


if __name__=='__main__':main()
