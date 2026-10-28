"""Run a bounded, serial CPU X2 reference preparation queue on the server."""
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
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--x2-model', type=Path, required=True)
    p.add_argument('--equilibrium', type=Path, required=True)
    p.add_argument('--source-fk-report', type=Path, required=True)
    p.add_argument('--restoration-helper', type=Path, help='Explicit private nonlinear contact-restoration source')
    p.add_argument('--timeout-seconds', type=int, default=900)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error('Output exists')
    config = json.loads(a.config.read_text())
    settings=config.get('adapter_options',{})
    if settings.get('nonlinear_restoration') and not a.restoration_helper:
        p.error('The selected adapter settings require an explicit restoration helper')
    if a.restoration_helper and not(settings.get('source_flat_foot_lock') and settings.get('collision_backtracking')):
        p.error('Restoration requires the configured flat support locks and collision backtracking')
    if settings.get('source_foot_surface_lock') and not settings.get('collision_backtracking'):
        p.error('Source surface locking requires actual-distance backtracking')
    if not json.loads(a.source_fk_report.read_text())['passed']:
        raise ValueError('Independent source FK verification required')
    source_audit = Path(config['server']['screening'])
    screening = json.loads((source_audit/'screening.json').read_text())
    source_rows = {r['motion']: r for r in screening['motions']}
    a.out.mkdir(parents=True)
    (a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    (a.out/'adapter.py').write_bytes(a.adapter.read_bytes())
    (a.out/'config.json').write_bytes(a.config.read_bytes())
    if a.restoration_helper:(a.out/'restoration-helper.py').write_bytes(a.restoration_helper.read_bytes())
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    results = []
    for i, candidate in enumerate(config['multi_motion_review_queue']):
        motion = candidate['motion']
        folder = a.out / f'{i:02d}_{Path(motion).stem}'
        folder.mkdir()
        command = [sys.executable, str(a.out/'adapter.py'), '--source', config['server']['raw'],
                   '--source-audit', str(source_audit), '--x2-model', str(a.x2_model),
                   '--equilibrium', str(a.equilibrium), '--motion', motion,
                   '--frames', str(source_rows[motion]['frames']), '--out', str(folder/'result')]
        if settings:
            command+=['--joint-limit-factor',str(settings.get('joint_limit_factor',1.)),
                '--wrist-reference-limit-deg',str(candidate.get('wrist_reference_limit_deg',settings.get('default_wrist_reference_limit_deg',15.)))]
            for flag in ['source_flat_foot_lock','source_foot_surface_lock','collision_backtracking']:
                if settings.get(flag):command+=['--'+flag.replace('_','-')]
        if a.restoration_helper:command+=['--restoration-helper',str(a.out/'restoration-helper.py')]
        (folder/'command.json').write_text(json.dumps(command, indent=2)+'\n')
        started = time.time()
        with (folder/'run.log').open('x') as log:
            try:
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                (folder/'pid').write_text(str(process.pid)+'\n')
                try:
                    code = process.wait(timeout=a.timeout_seconds)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    code = 124
            except OSError as error:
                log.write(str(error)+'\n')
                code = 125
        item = {'motion': motion, 'category': candidate['category'], 'exit_code': code,
                'elapsed_seconds': time.time()-started, 'output': str(folder),
                'source_sha256': source_rows[motion]['sha256'], 'training_approved': False,
                'adapter_sha256':hashlib.sha256((a.out/'adapter.py').read_bytes()).hexdigest(),
                'restoration_helper_sha256':hashlib.sha256((a.out/'restoration-helper.py').read_bytes()).hexdigest() if a.restoration_helper else None}
        report = folder/'result/result.json'
        if report.is_file():
            item['kinematic_result'] = json.loads(report.read_text())
            trajectory = folder/'result/x2-reference.npz'
            item['trajectory_sha256'] = hashlib.sha256(trajectory.read_bytes()).hexdigest()
        results.append(item)
        (a.out/'progress.json').write_text(json.dumps(results, indent=2)+'\n')
        print(json.dumps({k:v for k,v in item.items() if k!='kinematic_result'}), flush=True)
    (a.out/'completed.json').write_text(json.dumps({'results': results,
        'scope': 'Full-source kinematic candidates only; physical audit, continuous execution and policy training pending',
        'training_approved': False}, indent=2)+'\n')


if __name__ == '__main__':
    main()
