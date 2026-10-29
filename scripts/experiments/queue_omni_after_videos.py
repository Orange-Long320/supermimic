"""Continue one completed Omni run for three hours after its actual videos exist."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    a=p.parse_args();r=a.run;previous=a.previous
    progress=r/'queue-progress.json'
    def save(**values):progress.write_text(json.dumps(dict(updated_at=time.time(),**values),indent=2))
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        save(status='waiting_for_previous_training_audits_and_videos',previous=str(previous))
        deadline=time.monotonic()+1800
        while not (previous/'exit-code').exists():
            if time.monotonic()>deadline:raise TimeoutError('Previous postprocessing did not finish within30minutes')
            time.sleep(10)
        if (previous/'exit-code').read_text().strip()!='0':raise RuntimeError('Previous pipeline failed; no duplicate training launched')
        state=json.loads((previous/'progress.json').read_text())
        if state['status']!='completed':raise ValueError('Previous pipeline is not complete')
        rank=json.loads((previous/'train_3h/rank-0.json').read_text())
        checkpoint=Path(rank['checkpoint'])
        if sha(checkpoint)!=rank['checkpoint_sha256']:raise ValueError('Final checkpoint changed')
        dataset=Path('/home/2025-Long.xh/HHD/data/supermimic/logs/omni_practical_reference_gate_20261029_r1/dataset')
        manifest=json.loads((dataset/'manifest.json').read_text())
        videos=[]
        for clip in manifest['clips']:
            name=clip['name'];video=previous/'videos'/(name+'.mp4')
            report=json.loads(video.with_suffix('.json').read_text())
            if (not video.is_file() or video.stat().st_size<1024
                or report['checkpoint_sha256']!=rank['checkpoint_sha256']
                or report['render_mode']!='policy_only'):raise ValueError('Missing actual final policy video: '+name)
            if not (previous/('task-audit-'+name)/'report.json').is_file():raise ValueError('Missing independent task audit')
            videos.append(dict(path=str(video),sha256=sha(video),report=report))
        (r/'previous-video-proof.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),
            checkpoint_sha256=rank['checkpoint_sha256'],videos=videos,
            previous_task_checks_passed=state['actual_task_checks_passed'],
            note='Video production precedes continuation; task success is separate and manual review pending'),indent=2))
        py='/home/2025-Long.xh/HHD/venvs/supermimic/bin/python'
        command=[py,str(r/'source/scripts/experiments/run_omni_dual_three_hour.py'),
            '--run',str(r),'--dataset',str(dataset),'--initial-checkpoint',str(checkpoint),
            '--expected-checkpoint-sha256',rank['checkpoint_sha256'],'--source-commit',a.source_commit,
            '--gpus','0','2','--minutes','180','--num-envs-per-rank','4096',
            '--batch-size','1024','--preserve-learning-rate','--memory-budget-mib','23500',
            '--torch-memory-fraction','0.8','--validated-resume-run',str(previous),'--queue-seconds','120']
        (r/'launch-command.json').write_text(json.dumps(command,indent=2))
        with (r/'controller.log').open('w') as log:
            child=subprocess.Popen(command,cwd=r/'source',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(status='continuation_launched',driver_pid=child.pid,previous_checkpoint_sha256=rank['checkpoint_sha256'])
    except Exception as error:
        save(status='failed',reason=str(error));raise


if __name__=='__main__':main()
