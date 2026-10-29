"""Move this task to a new batch at the next complete numbered checkpoint.

Never inject into the live interpreter, overwrite an old run, reset the original
training budget, or signal processes whose ownership/arguments do not match.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--after-iteration',type=int,required=True)
    p.add_argument('--wait-seconds',type=int,default=900)
    a=p.parse_args();r=a.run;old=a.parent
    if (r/'transition.json').exists():p.error('Transition already started')
    def state(**values):
        (r/'transition.json').write_text(json.dumps(dict(updated_at=time.time(),**values),indent=2))
    (r/'transition-pid').write_text(str(os.getpid()))
    (r/'transition-as-run.py').write_bytes(Path(__file__).read_bytes())
    original=json.loads((old/'progress.json').read_text())
    if original['phase']!='train_3h' or original['status']!='running':raise ValueError('Expected the active original fit')
    driver=int((old/'pid').read_text());child=int(original['pid'])
    def owned(pid,needle):
        proc=Path('/proc',str(pid))
        return proc.exists() and proc.stat().st_uid==os.getuid() and needle in (proc/'cmdline').read_bytes().decode().replace('\x00',' ')
    if not owned(driver,str(old/'source/scripts/experiments/run_omni_dual_three_hour.py')) or not owned(child,str(old/'train_3h')):
        raise ValueError('Original process identity mismatch')
    if os.getpgid(driver)!=driver or os.getpgid(child)!=child:raise ValueError('Expected independent owned process groups')
    state(status='waiting_for_next_numbered_checkpoint',parent_driver=driver,parent_training=child,after_iteration=a.after_iteration)
    deadline=time.monotonic()+a.wait_seconds
    while True:
        if not owned(driver,str(old/'source/scripts/experiments/run_omni_dual_three_hour.py')) or not owned(child,str(old/'train_3h')):
            raise RuntimeError('Original training became terminal while waiting; inspect, never restart blindly')
        candidates=sorted((old/'train_3h').glob('ckpt_[0-9]*.pt'))
        candidates=[x for x in candidates if int(x.stem.split('_')[1])>a.after_iteration]
        if candidates:checkpoint=candidates[0];break
        if time.monotonic()>deadline:raise TimeoutError('No new atomic numbered checkpoint; original training left running')
        time.sleep(.5)
    # Freeze only the verified training group before reading its just-published
    # immutable numbered checkpoint. No simulation states are edited.
    os.killpg(child,signal.SIGSTOP);frozen_at=time.time()
    try:
        import torch
        torch.set_num_threads(2)
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if ck['iteration']!=int(checkpoint.stem.split('_')[1]):raise ValueError('Checkpoint counter mismatch')
        if not all(torch.isfinite(t).all() for key in ['wm','policy','critic'] for t in ck[key].values() if torch.is_tensor(t)):
            raise ValueError('Checkpoint contains nonfinite tensors')
        expected=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        previous_rank=old/'train_3h/rank-0.json'
        previous=json.loads(previous_rank.read_text())
        loop_start=previous.get('training_started_at',previous_rank.stat().st_mtime)
        spent=frozen_at-loop_start
        if not 0<spent<10800:raise ValueError('Original three-hour budget not available for continuation')
    except BaseException:
        os.killpg(child,signal.SIGCONT)
        state(status='validation_failed_original_resumed')
        raise
    snapshot={'checkpoint':str(checkpoint),'checkpoint_sha256':expected,'iteration':ck['iteration'],
        'updates':ck['num_updates'],'training_seconds_already_spent':spent,'original_loop_start':loop_start,
        'loop_start_source':'record field' if 'training_started_at' in previous else 'running rank metadata mtime, saved immediately before Trainer.run',
        'frozen_at':frozen_at,'parent_driver':driver,'parent_training':child,
        'reason':'User explicitly allows higher VRAM and requests batch1024; retain original three-hour budget'}
    (r/'continuation-source.json').write_text(json.dumps(snapshot,indent=2))
    (old/'batch-change-interruption.json').write_text(json.dumps(snapshot,indent=2))
    (old/'progress-before-batch-change.json').write_text(json.dumps(original,indent=2))
    os.killpg(driver,signal.SIGTERM)
    os.killpg(child,signal.SIGTERM);os.killpg(child,signal.SIGCONT)
    deadline=time.monotonic()+20
    while time.monotonic()<deadline:
        if not owned(child,str(old/'train_3h')):break
        time.sleep(.25)
    if owned(child,str(old/'train_3h')):
        raise RuntimeError('Owned training did not terminate; no competing new GPU process launched')
    (old/'progress.json').write_text(json.dumps(dict(status='superseded_by_user_batch_change',next_run=str(r),**snapshot),indent=2))
    py='/home/2025-Long.xh/HHD/venvs/supermimic/bin/python'
    dataset=Path(previous['configuration']['data']['npz_dir']).parent
    command=[py,str(r/'source/scripts/experiments/run_omni_dual_three_hour.py'),'--run',str(r),
        '--dataset',str(dataset),'--initial-checkpoint',str(checkpoint),'--expected-checkpoint-sha256',expected,
        '--source-commit',a.source_commit,'--gpus','3','--minutes','180','--batch-size','1024',
        '--memory-budget-mib','20000','--previous-training-seconds',str(spent),
        '--include-capacity-in-budget','--queue-seconds','120']
    (r/'launch-command.json').write_text(json.dumps(command,indent=2))
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',SM_ROBOT='x2_ultra',PYTHONUNBUFFERED='1')
    for key in ['PYTHONHOME','PYTHONPATH','SM_ROBOT_SPEC','SM_SOLE_BOX_CONFIG']:env.pop(key,None)
    with (r/'controller.log').open('w') as log:
        child_process=subprocess.Popen(command,cwd=r/'source',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    state(status='continuation_driver_started',driver_pid=child_process.pid,**snapshot)


if __name__=='__main__':main()
