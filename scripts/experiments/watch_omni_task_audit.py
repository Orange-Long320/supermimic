"""Bounded CPU follow-up of an existing immutable three-task training run.

Never launch or restart training. Audit only completed captures after its driver
has exited; retain failed runs and never convert missing evidence into success.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--auditor', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'pid').write_text(str(os.getpid()))
    def progress(**values):
        (a.out/'progress.json').write_text(json.dumps(values, indent=2))
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    sources = [a.auditor, a.auditor.with_name('audit_omni_execution.py'), Path(__file__)]
    hashes = {str(f):sha(f) for f in sources}
    (a.out/'source-sha256.json').write_text(json.dumps(hashes, indent=2))
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               SM_ROBOT='x2_ultra', PYTHONPATH=str(a.training/'source/src'))
    if hasattr(os, 'sched_setaffinity'): os.sched_setaffinity(0, {4,5})
    progress(status='waiting_for_existing_training', training=str(a.training))
    deadline = time.monotonic()+7200
    try:
        while not (a.training/'exit-code').exists():
            if time.monotonic()>deadline: raise TimeoutError('Existing driver did not finish within two hours')
            time.sleep(30)
        if any(sha(f)!=h for f,h in hashes.items()): raise ValueError('Auditor sources changed while waiting')
        manifest = json.loads((a.dataset/'manifest.json').read_text())
        reports = []
        for clip in manifest['clips']:
            name = clip['name']; trace = a.training/('capture-'+name)
            if not (trace/'result.json').exists():
                reports.append(dict(clip=name, complete_task_verified=False, reason='No completed actual capture'))
                continue
            command = [sys.executable, str(a.auditor), '--trace', str(trace), '--dataset', str(a.dataset),
                       '--clip', name, '--out', str(a.out/name)]
            (a.out/(name+'-command.json')).write_text(json.dumps(command,indent=2))
            progress(status='auditing', clip=name)
            with (a.out/(name+'.log')).open('w') as log:
                completed = subprocess.run(command, cwd=a.training/'source', env=env,
                    stdout=log, stderr=subprocess.STDOUT, timeout=600)
            (a.out/(name+'-exit')).write_text(str(completed.returncode))
            if completed.returncode:
                reports.append(dict(clip=name, complete_task_verified=False, reason='Auditor execution failed; inspect log'))
            else: reports.append(json.loads((a.out/name/'report.json').read_text()))
        passed = bool(len(reports)==3 and all(r['complete_task_verified'] for r in reports))
        progress(status='completed', reports=reports, task_checks_passed=passed, final_validation_approved=False,
            training_driver_exit=(a.training/'exit-code').read_text().strip(),
            scope='Task-specific actual trajectory audit; manual video review still required before final project acceptance')
    except Exception as error:
        progress(status='failed', reason=str(error)); raise


if __name__ == '__main__': main()
