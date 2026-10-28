"""Run the reviewed three-task capacity, bounded fit and immutable evaluation.

The source checkout must already be deployed privately with its assets/env
links. This driver never changes the shared checkout, model or original data.
Legacy box-standing audits are retained; task-specific landing acceptance is
explicitly pending independent review of the saved actual contact trajectories.
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
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--initial-checkpoint',type=Path,required=True)
    p.add_argument('--expected-checkpoint-sha256',required=True)
    p.add_argument('--package-commit',required=True)
    p.add_argument('--driver-commit',required=True)
    p.add_argument('--gpu',type=int,default=3)
    p.add_argument('--minutes',type=float,default=60.)
    a=p.parse_args()
    if not 0<a.minutes<=60:p.error('One bounded fit of at most60minutes')
    r=a.run;s=r/'source';py='/home/2025-Long.xh/HHD/venvs/supermimic/bin/python'
    with (r/'progress.json').open('x') as f:json.dump({'status':'initializing'},f)
    (r/'pid').write_text(str(os.getpid()))
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:])
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    env=os.environ.copy();env.update(SM_ROBOT='x2_ultra',PYTHONPATH=str(s/'src'),OMP_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID=str(a.gpu))
    for key in ['PYTHONHOME','SM_ROBOT_SPEC','SM_SOLE_BOX_CONFIG']:env.pop(key,None)
    def progress(**value): (r/'progress.json').write_text(json.dumps(value,indent=2))
    def idle():
        deadline=time.monotonic()+30
        while True:
            busy=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
            if not busy:return
            if time.monotonic()>deadline:raise RuntimeError('Physical GPU occupied: '+busy)
            time.sleep(1)
    def stage(name,command,gpu=False,timeout=180):
        child_env=env.copy();child_env['CUDA_VISIBLE_DEVICES']=str(a.gpu) if gpu else ''
        if gpu:idle()
        (r/(name+'-command.json')).write_text(json.dumps(command,indent=2))
        stop=threading.Event();peak=[0]
        def monitor():
            with (r/(name+'-gpu-memory.jsonl')).open('w') as log:
                while not stop.is_set():
                    try:
                        value=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=5).strip())
                        peak[0]=max(peak[0],value);log.write(json.dumps({'time':time.time(),'device_used_mib':value})+'\n');log.flush()
                    except Exception:pass
                    stop.wait(1)
        thread=threading.Thread(target=monitor,daemon=True) if gpu else None
        if thread:thread.start()
        with (r/(name+'.log')).open('w') as log:
            child=subprocess.Popen(command,cwd=s,env=child_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            (r/(name+'-pid')).write_text(str(child.pid));progress(status='running',phase=name,pid=child.pid)
            try:code=child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                code=124
        stop.set()
        if thread:thread.join(timeout=6)
        (r/(name+'-exit')).write_text(str(code))
        (r/(name+'-memory-peak.json')).write_text(json.dumps({'device_peak_mib':peak[0],
            'scope':'Whole exclusive physical GPU at1s; samples may miss short transients'}))
        if code:raise RuntimeError(f'{name} failed: {code}')
        return peak[0]
    try:
        deployment=json.loads((r/'deployment.json').read_text())
        if deployment['source_commit']!=a.package_commit:raise ValueError('Deployed revision mismatch')
        data=json.loads((a.dataset/'manifest.json').read_text())
        if data['distinct_source_families']!=3 or set(data['task_categories'])!={'box_stand','vault_land','step_up_down_land'}:
            raise ValueError('Expected the reviewed three-family task batch')
        if sha(data['terrain'])!=data['terrain_sha256'] or not all(c['terminal_reference_kinematics']['passed'] for c in data['clips']):
            raise ValueError('Scene or reference terminal checks changed')
        idle()
        gpu_info=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=index,uuid,name,memory.total,driver_version','--format=csv,noheader'],text=True)
        (r/'gpu-info.txt').write_text(gpu_info)
        (r/'resources.txt').write_text(f'GPU{a.gpu}; CPU2 threads; capacity2048env/512batch/100steps420s; fit{a.minutes}min/outer{a.minutes*60+600}s; Torch65%; device sampled peak<=18000MiB before fit. Source0 plus existing return sampler50/50; actual evaluation always source0.\n')
        (r/'source-commit.txt').write_text(f'driver{a.driver_commit}; entry/package{a.package_commit}; shared checkout untouched\n')
        files=[Path(__file__),s/'scripts/experiments/train_omni_prefix.py',s/'src/supermimic/kinematics/g1_fk.py',
               s/'scripts/evaluation/record_policy_states.py',s/'scripts/evaluation/audit_omni_execution.py']
        (r/'source-sha256.json').write_text(json.dumps({str(f):sha(f) for f in files},indent=2))
        profile=(s/'assets/robots/x2_ultra/profile.json').resolve()
        xml=Path(json.loads(profile.read_text())['xml_path'])
        if not xml.is_absolute():xml=profile.parent/xml
        descriptors={str(f):sha(f) for f in [profile,xml.resolve()]}
        (r/'asset-descriptor-sha256.json').write_text(json.dumps(descriptors,indent=2))
        source_stat=a.initial_checkpoint.stat()
        if sha(a.initial_checkpoint)!=a.expected_checkpoint_sha256:raise ValueError('Unexpected initial weights')
        initial=r/'initial-weights.pt'
        if initial.exists():raise ValueError('Initial pin already exists')
        shutil.copyfile(a.initial_checkpoint,initial)
        if a.initial_checkpoint.stat().st_mtime_ns!=source_stat.st_mtime_ns or sha(initial)!=a.expected_checkpoint_sha256:
            raise ValueError('Source checkpoint changed while pinning')
        probe_env=env.copy();probe_env['CUDA_VISIBLE_DEVICES']=str(a.gpu)
        probe="import sys,json,torch,numpy,mujoco;from supermimic.robots.paths import select_robot;select_robot('x2_ultra',check_assets=True);print(json.dumps(dict(python=sys.version,prefix=sys.prefix,torch=torch.__version__,cuda=torch.version.cuda,numpy=numpy.__version__,mujoco=mujoco.__version__,cuda_available=torch.cuda.is_available(),visible_devices=torch.cuda.device_count(),device=torch.cuda.get_device_name(0))))"
        tested=subprocess.run([py,'-c',probe],cwd=s,env=probe_env,capture_output=True,text=True,timeout=60)
        (r/'environment.jsonl').write_text(tested.stdout);(r/'environment.log').write_text(tested.stderr)
        if tested.returncode:raise RuntimeError('Environment preflight failed')
        common=[py,str(s/'scripts/experiments/train_omni_prefix.py'),'--dataset',str(a.dataset),'--gpu',str(a.gpu),
                '--source-commit',a.package_commit,'--package-commit',a.package_commit,'--mixed-start-fraction','0.5']
        peak=stage('capacity',common+['--checkpoint',str(initial),'--out',str(r/'capacity'),'--phase','capacity','--weights-only'],True,420)
        capacity=json.loads((r/'capacity/capacity.json').read_text())
        if (capacity['status']!='completed' or capacity['new_updates']<20 or not capacity['weights_only_transfer']
            or capacity['pinned_start_environments']!=1024 or capacity['return_sampling_environments']!=1024 or not 0<peak<=18000):
            raise RuntimeError('Capacity, curriculum or memory headroom gate failed')
        stage('train_1h',common+['--checkpoint',capacity['checkpoint'],'--out',str(r/'train_1h'),'--phase','optimize',
              '--proof',str(r/'capacity/capacity.json'),'--optimization-minutes',str(a.minutes)],True,a.minutes*60+600)
        final=json.loads((r/'train_1h/capacity.json').read_text());checkpoint=final['checkpoint']
        if final['status']!='completed' or final['new_updates']<=0:raise RuntimeError('No completed parameter optimization')
        if any(sha(f)!=value for f,value in descriptors.items()):raise RuntimeError('Robot asset descriptors changed during training')
        stage('benchmark',[py,str(s/'scripts/evaluation/eval_benchmark.py'),'--ckpt',checkpoint,'--npz-dir',str(a.dataset/'motions'),
              '--num-envs','3','--device','cuda:0','--out',str(r/'benchmark.json')],True,300)
        reports=[];(r/'videos').mkdir()
        for clip in data['clips']:
            name=clip['name'];trace=r/('capture-'+name);audit=r/('box-standing-audit-'+name)
            stage('capture-'+name,[py,str(s/'scripts/evaluation/record_policy_states.py'),'--ckpt',checkpoint,
                '--source',str(a.dataset/'motions'/(name+'.npz')),'--out',str(trace),'--seconds','16',
                '--hold-last-reference','--record-contact-forces','--device','cuda:0'],True,300)
            stage('box-standing-audit-'+name,[py,str(s/'scripts/evaluation/audit_omni_execution.py'),'--trace',str(trace),
                '--dataset',str(a.dataset),'--clip',name,'--out',str(audit)],False,180)
            report=json.loads((audit/'report.json').read_text());reports.append({'task_type':clip['task_type'],
                'box_standing_audit_only':True,**{k:v for k,v in report.items() if k not in ['rows','scope']}})
            video=r/'videos'/(name+'.mp4')
            stage('video-'+name,[py,str(s/'scripts/evaluation/render_policy.py'),'--robot','x2_ultra','--ckpt',checkpoint,
                '--npz-dir',str(a.dataset/'motions'),'--clip',name,'--seconds','16','--no-reference','--no-label','--out',str(video)],True,240)
            probe=subprocess.run(['ffprobe','-v','error','-show_entries','stream=codec_type,nb_frames,duration','-of','json',str(video)],capture_output=True,text=True)
            (r/(name+'-video-probe.json')).write_text(probe.stdout)
            if probe.returncode or not any(x.get('codec_type')=='video' and float(x.get('duration',0))>0 for x in json.loads(probe.stdout).get('streams',[])):
                raise RuntimeError('Video is not a valid nonempty stream')
        progress(status='completed',phase='awaiting_task_specific_review',reports=reports,
                 checkpoint_sha256=final['checkpoint_sha256'],task_specific_validation_pending=True,final_validation_approved=False)
        (r/'exit-code').write_text('0')
    except Exception as error:
        progress(status='failed',reason=str(error));(r/'exit-code').write_text('1');raise


if __name__=='__main__':main()
