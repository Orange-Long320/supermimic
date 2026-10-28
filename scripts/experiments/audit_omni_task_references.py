"""Bounded CPU inverse-dynamics diagnostics of the three dense task references.

This does not change references or training. Estimated friction cones are
necessary-condition diagnostics, never actual policy or contact certificates.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'pid').write_text(str(os.getpid()))
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{10,11})
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',SM_ROBOT='x2_ultra',PYTHONPATH=str(a.source/'src'))
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    auditor=a.source/'scripts/evaluation/audit_omni_x2_reference.py'
    profile=(a.source/'assets/robots/x2_ultra/profile.json').resolve()
    model=profile.parent/json.loads(profile.read_text())['xml_path']
    sources=[Path(__file__),auditor,a.source/'src/supermimic/data/contact_wrench.py',profile,model]
    (a.out/'source-sha256.json').write_text(json.dumps({str(f):sha(f) for f in sources},indent=2))
    def progress(**values):(a.out/'progress.json').write_text(json.dumps(values,indent=2))
    results=[]
    try:
        for name in ['step43_box_stand','step28_return_land','hand46_vault_land']:
            reference=a.tasks/name/'dense/x2-reference.npz';scene=reference.with_name('result.json')
            command=[sys.executable,str(auditor),'--reference',str(reference),'--scene',str(scene),
                '--model',str(model),'--allow-hand-support','--friction','0.8','--out',str(a.out/name)]
            (a.out/(name+'-command.json')).write_text(json.dumps(command,indent=2))
            progress(status='running',phase=name,results=results)
            with (a.out/(name+'.log')).open('w') as log:
                child=subprocess.Popen(command,cwd=a.source,env=env,stdout=log,stderr=subprocess.STDOUT)
                (a.out/(name+'-pid')).write_text(str(child.pid))
                try:code=child.wait(timeout=900)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    code=124
            (a.out/(name+'-exit')).write_text(str(code))
            record=dict(name=name,exit_code=code,reference_sha256=sha(reference),scene_sha256=sha(scene))
            if code==0:
                report=json.loads((a.out/name/'report.json').read_text())
                record.update(summary=report['summary'],necessary_diagnostics_passed=report['necessary_diagnostics_passed'])
            results.append(record)
        progress(status='completed',results=results,actual_policy_validation=False)
    except Exception as error:
        progress(status='failed',reason=str(error),results=results);raise


if __name__=='__main__':main()
