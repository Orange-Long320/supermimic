"""Freeze a saved gait checkpoint and record selected real, reset-free rollouts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','checkpoint','dataset','out','renderer']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--clips',nargs='+',default=['walk','backward','turn_left_inplace','jog'])
    p.add_argument('--run-label',default='')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    report=dict(status='initializing',gpu=a.gpu,pid=os.getpid(),started_epoch=time.time(),clips=[])
    def save(): (a.out/'preview.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    busy=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if busy:raise RuntimeError('Preview GPU occupied: '+busy)
    # Open once: atomic replacements of ckpt_latest cannot change this handle.
    frozen=a.out/'checkpoint.pt'
    with a.checkpoint.open('rb') as src, frozen.open('xb') as dst:shutil.copyfileobj(src,dst)
    import torch,numpy as np
    ck=torch.load(frozen,map_location='cpu',weights_only=False)
    report.update(checkpoint_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),iteration=ck['iteration'],updates=ck['num_updates'],source_checkpoint=str(a.checkpoint))
    del ck
    shutil.copy2(__file__,a.out/'controller-as-run.py');shutil.copy2(a.renderer,a.out/'renderer-as-run.py')
    env=os.environ.copy()
    for k in ['PYTHONPATH','PYTHONHOME','SM_SOLE_BOX_CONFIG','SM_ROBOT_SPEC']:env.pop(k,None)
    env.update(CUDA_VISIBLE_DEVICES=str(a.gpu),MUJOCO_GL='egl',SM_ROBOT='x2_ultra',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',PYTHONPATH=str(a.source/'src'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    env['XDG_CACHE_HOME']='/home/2025-Long.xh/HHD/data/supermimic/cache';env['WARP_CACHE_PATH']=env['XDG_CACHE_HOME']+'/warp'
    py='/home/2025-Long.xh/HHD/venvs/supermimic/bin/python'
    def run(name,cmd):
        (a.out/(name+'-command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
        with (a.out/(name+'.log')).open('x') as log:subprocess.run(cmd,cwd=a.source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    report['status']='recording';save()
    for clip in a.clips:
        trace=a.out/('trace-'+clip)
        with np.load(a.dataset/(clip+'.npz')) as d:seconds=len(d['joint_pos'])/float(d['fps'][0])
        run('capture-'+clip,[py,str(a.source/'scripts/evaluation/record_policy_states.py'),'--ckpt',str(frozen),'--source',str(a.dataset/(clip+'.npz')),'--out',str(trace),'--seconds',str(seconds),'--device','cuda:0','--hold-last-reference'])
        result=json.loads((trace/'result.json').read_text())
        failed=bool(result['failure_physics_steps']);duration=result['actions']*.02
        label=f"{a.run_label} {clip} | checkpoint {report['iteration']} | "+(f'STOPPED failure at {duration:.2f}s' if failed else 'full clip recorded')
        video=a.out/(clip+'.mp4')
        run('render-'+clip,[py,str(a.out/'renderer-as-run.py'),'--trajectory',str(trace/'baseline-trajectory.npz'),'--out',str(video),'--playback-speed','1','--azimuth','135','--distance','2.7','--label',label])
        report['clips'].append(dict(name=clip,requested_seconds=seconds,recorded_seconds=duration,failed=failed,last_reference_frame=result['last_reference_frame'],reference_frame_count=result['reference_frame_count'],video=str(video),policy_alone=result['policy_alone']))
        save()
    # Labelled panels; freeze shorter recordings for display, not simulation.
    longest=max(c['recorded_seconds'] for c in report['clips'])+1
    cmd=['ffmpeg','-v','error']
    for clip in report['clips']:cmd+=['-i',clip['video']]
    filters=[]
    for i,clip in enumerate(report['clips']):
        filters.append(f'[{i}:v]scale=640:480,fps=50,tpad=stop_mode=clone:stop_duration={longest},trim=duration={longest},setpts=PTS-STARTPTS[v{i}]')
    count=len(report['clips'])
    columns=2 if count<=4 else 3
    layout='|'.join(f'{(i%columns)*640}_{(i//columns)*480}' for i in range(count))
    filters.append(''.join(f'[v{i}]' for i in range(count))+f'xstack=inputs={count}:layout={layout}:fill=black[out]')
    cmd+=['-filter_complex',';'.join(filters),'-map','[out]','-c:v','libx264','-crf','19','-threads','2','-an','-movflags','+faststart','-n',str(a.out/('four-gaits-policy.mp4' if len(a.clips)==4 else 'five-gaits-policy.mp4'))]
    run('combine',cmd)
    report.update(status='completed',display_note='Real policy physics. Each rollout stops at original failure; panel freezes after last recorded frame. No automatic resets, no servo prefix, no reference animation substitution.');save()

if __name__=='__main__':main()
