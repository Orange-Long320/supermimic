"""Execute pinned diagnostic argument lists sequentially without shell parsing."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', type=Path, required=True)
    a = p.parse_args()
    source = a.request.parent.resolve()
    record = json.loads(a.request.read_text())
    expected = Path('/home/2025-Long.xh/HHD/venvs/supermimic')
    if Path(sys.prefix).resolve() != expected.resolve():
        p.error('Dedicated server interpreter required')
    env = os.environ.copy()
    for key in ('PYTHONHOME', 'SM_ROBOT_SPEC', 'SM_SOLE_BOX_CONFIG', 'SM_GPU_PINNED'):
        env.pop(key, None)
    env.update(record['environment'])
    record.update(pid=os.getpid(), status='running')
    def save():
        (source/'diagnostic-progress.json').write_text(json.dumps(record, indent=2)+'\n')
    save()
    for job in record['jobs']:
        argv = job['argv']
        if Path(argv[0]).resolve() != (expected/'bin/python').resolve() or Path(argv[1]).resolve().parent != source:
            raise ValueError('Only diagnostic entries in this pinned source folder may run')
        gpu = env.get('CUDA_VISIBLE_DEVICES', '')
        if not gpu or len(gpu.split(',')) != 1:
            raise ValueError('Select one explicit GPU for the diagnostic sequence')
        active = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-compute-apps=pid',
                                          '--format=csv,noheader'], text=True).strip()
        job['gpu_compute_processes_before'] = active
        if active:
            job.update(status='gpu_busy_not_started', finished_epoch=time.time())
            record.update(status='gpu_busy_not_started', finished_epoch=time.time())
            save()
            raise RuntimeError('Diagnostic GPU is occupied; this job was not started')
        job.update(started_epoch=time.time(), status='running')
        with (source/(job['name']+'.log')).open('x') as output:
            child = subprocess.Popen(argv, env=env, stdout=output, stderr=subprocess.STDOUT)
            job['pid'] = child.pid; save()
            try:
                job['returncode'] = child.wait(timeout=600)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
                job.update(returncode=child.returncode, timed_out=True)
        job.update(finished_epoch=time.time(), status='completed' if job['returncode']==0 else 'failed')
        save()
    record.update(status='completed', finished_epoch=time.time()); save()


if __name__ == '__main__':
    main()
