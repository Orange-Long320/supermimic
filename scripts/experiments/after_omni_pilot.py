"""Wait for the owned one-hour run, then evaluate its immutable final policy.

No new training is started. Every evaluation uses the saved terrain manifest;
each clip gets its own actual state capture, independent audit and video.
"""
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
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--capture-tool',type=Path,required=True)
    p.add_argument('--audit-tool',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=3)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    (a.out/'pid').write_text(str(os.getpid()))
    for label,source in [('capture.py',a.capture_tool),('audit.py',a.audit_tool)]:
        (a.out/label).write_bytes(source.read_bytes())
    source=a.run/'source';python=sys.executable;sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    base=os.environ.copy();base.update(SM_ROBOT='x2_ultra',PYTHONPATH=str(source/'src'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    for key in ['PYTHONHOME','SM_ROBOT_SPEC','SM_SOLE_BOX_CONFIG']:base.pop(key,None)
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    def progress(phase,**extra):
        (a.out/'progress.json').write_text(json.dumps({'phase':phase,'time':time.time(),**extra},indent=2))
    def stage(name,command,gpu=False,timeout=600):
        env=base.copy();env['CUDA_VISIBLE_DEVICES']=str(a.gpu) if gpu else ''
        if gpu:
            release_deadline=time.monotonic()+30
            while True:
                pids=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
                if not pids:break
                if time.monotonic()>release_deadline:raise RuntimeError(f'Evaluation GPU occupied: {pids}')
                time.sleep(1)
            env['MUJOCO_EGL_DEVICE_ID']=str(a.gpu)
        progress(name,status='running');(a.out/(name+'-command.json')).write_text(json.dumps(command,indent=2))
        with (a.out/(name+'.log')).open('w') as log:
            process=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
            (a.out/(name+'-pid')).write_text(str(process.pid))
            try:code=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                code=124
        (a.out/(name+'-exit')).write_text(str(code));return code
    try:
        progress('waiting_for_owned_training',status='waiting');deadline=time.monotonic()+5400
        while not (a.run/'exit-code').exists():
            if time.monotonic()>deadline:
                progress('observation_deadline',status='waiting_timeout',training_terminal=False)
                return
            time.sleep(5)
        if (a.run/'exit-code').read_text().strip()!='0':raise RuntimeError('Owned training pipeline failed; inspect its logs')
        proof=json.loads((a.run/'train_1h/capacity.json').read_text())
        if proof['status']!='completed' or proof['updates']<=24 or proof['elapsed_seconds']<3500:raise RuntimeError('One-hour completion not established')
        train_pid=int((a.run/'train-pid').read_text());handle=Path(f'/proc/{train_pid}/cmdline')
        if handle.exists() and b'omni_pilot.py' in handle.read_bytes():raise RuntimeError('Original training process remains live')
        original=Path(proof['checkpoint']);before=original.stat();checkpoint=a.out/'evaluated-checkpoint.pt'
        shutil.copyfile(original,checkpoint);after=original.stat()
        if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns) or sha(checkpoint)!=proof['checkpoint_sha256']:raise RuntimeError('Final checkpoint changed')
        manifest=json.loads((a.run/'dataset/manifest.json').read_text())
        if sha(manifest['terrain'])!=manifest['terrain_sha256']:raise RuntimeError('Evaluation terrain changed')
        (a.out/'training-proof.json').write_text(json.dumps(proof,indent=2))
        idle_deadline=time.monotonic()+900
        while subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
            if time.monotonic()>idle_deadline:
                progress('waiting_for_evaluation_gpu',status='waiting_timeout',training_terminal=True)
                return
            time.sleep(10)
        codes={'benchmark':stage('benchmark',[python,str(source/'scripts/evaluation/eval_benchmark.py'),
            '--ckpt',str(checkpoint),'--npz-dir',str(a.run/'dataset/motions'),'--num-envs','2','--device','cuda:0','--out',str(a.out/'benchmark.json')],gpu=True)}
        summaries=[];(a.out/'videos').mkdir()
        for clip in manifest['clips']:
            name=clip['name'];capture=a.out/('capture-'+name);audit=a.out/('audit-'+name)
            code=stage('capture-'+name,[python,str(a.out/'capture.py'),'--ckpt',str(checkpoint),
                '--source',str(a.run/'dataset/motions'/f'{name}.npz'),'--out',str(capture),
                '--seconds','16','--hold-last-reference','--device','cuda:0'],gpu=True)
            codes['capture-'+name]=code
            if code==0:
                codes['audit-'+name]=stage('audit-'+name,[python,str(a.out/'audit.py'),'--trace',str(capture),
                    '--dataset',str(a.run/'dataset'),'--clip',name,'--out',str(audit)],timeout=180)
                if (audit/'report.json').exists():
                    report=json.loads((audit/'report.json').read_text());summaries.append({k:v for k,v in report.items() if k not in ['rows','scope']})
            video=a.out/'videos'/f'{name}.mp4'
            codes['video-'+name]=stage('video-'+name,[python,str(source/'scripts/evaluation/render_policy.py'),
                '--robot','x2_ultra','--ckpt',str(checkpoint),'--npz-dir',str(a.run/'dataset/motions'),
                '--clip',name,'--seconds','16','--no-reference','--no-label','--out',str(video)],gpu=True)
            if video.exists():
                probe=subprocess.run(['ffprobe','-v','error','-show_entries','stream=codec_type,nb_frames,duration','-of','json',str(video)],capture_output=True,text=True)
                (a.out/(name+'-video-probe.json')).write_text(probe.stdout or '{}')
                streams=json.loads(probe.stdout or '{}').get('streams',[])
                valid=probe.returncode==0 and any(s.get('codec_type')=='video' and s.get('duration') not in (None,'N/A') and float(s['duration'])>0 for s in streams)
                codes['video-probe-'+name]=0 if valid else 1
        summary={'clips':summaries,'exit_codes':codes,'checkpoint_sha256':sha(checkpoint),'terrain_sha256':manifest['terrain_sha256'],
            'all_complete_climbs_verified':len(summaries)==len(manifest['clips']) and all(s['complete_climb_verified'] for s in summaries),
            'further_training_started':False,'goal_completion_requires_manual_evidence_review':True}
        (a.out/'summary.json').write_text(json.dumps(summary,indent=2));progress('awaiting_review',status='completed',**summary)
        (a.out/'exit-code').write_text('0' if all(c==0 for c in codes.values()) else '1')
    except Exception as error:
        progress('needs_attention',status='failed',reason=str(error));(a.out/'exit-code').write_text('1');raise


if __name__=='__main__':main()
