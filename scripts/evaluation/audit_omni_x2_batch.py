"""Audit completed candidates as a specific live Omni batch produces them."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-run',type=Path,required=True)
    p.add_argument('--auditor',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('New output required')
    a.out.mkdir(parents=True)
    (a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    (a.out/'auditor.py').write_bytes(a.auditor.read_bytes())
    deadline=time.monotonic()+7200
    results=[];seen=set()
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    while time.monotonic()<deadline:
        progress=a.batch_run/'batch/progress.json'
        try:queue=json.loads(progress.read_text()) if progress.is_file() else []
        except json.JSONDecodeError:
            time.sleep(1);continue
        for row in queue:
            if row['motion'] in seen:continue
            seen.add(row['motion']);source=Path(row['output'])/'result'
            folder=a.out/Path(row['output']).name;folder.mkdir()
            item={'motion':row['motion'],'retarget_exit_code':row['exit_code'],'training_approved':False}
            if row['exit_code']==0 and (source/'result.json').is_file() and (source/'x2-reference.npz').is_file():
                command=[sys.executable,str(a.out/'auditor.py'),'--reference',str(source/'x2-reference.npz'),
                    '--scene',str(source/'result.json'),'--model',str(a.model),'--out',str(folder/'audit')]
                if 'hand-supported' in row['category']:command.append('--allow-hand-support')
                (folder/'command.json').write_text(json.dumps(command,indent=2)+'\n')
                with (folder/'run.log').open('x') as log:
                    process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                    (folder/'pid').write_text(str(process.pid)+'\n')
                    try:item['audit_exit_code']=process.wait(timeout=900)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        try:process.wait(timeout=15)
                        except subprocess.TimeoutExpired:process.kill();process.wait()
                        item['audit_exit_code']=124
                report=folder/'audit/report.json'
                if report.is_file():
                    v=json.loads(report.read_text());item.update(summary=v['summary'],necessary_diagnostics_passed=v['necessary_diagnostics_passed'])
            results.append(item)
            (a.out/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps(item),flush=True)
        if (a.batch_run/'exit-code').is_file():
            # progress is written before the batch controller's exit marker.
            (a.out/'completed.json').write_text(json.dumps({'results':results,'training_approved':False},indent=2)+'\n')
            return
        time.sleep(10)
    raise TimeoutError('Bounded batch audit wait expired; preserve all partial results')


if __name__=='__main__':main()
