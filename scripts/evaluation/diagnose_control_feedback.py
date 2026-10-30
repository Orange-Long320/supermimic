"""Paired physical ablations of added movement and stop feedback (server only)."""
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
    for name in ('source', 'checkpoint', 'dataset', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--gpu', type=int, default=2)
    a = p.parse_args()
    expected = Path('/home/2025-Long.xh/HHD/venvs/supermimic').resolve()
    if Path(sys.prefix).resolve() != expected:
        raise RuntimeError('Dedicated server interpreter required')
    if subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('Diagnostic GPU must be idle before launch')
    a.out.mkdir(exist_ok=False)
    env = os.environ.copy()
    for key in ('PYTHONHOME', 'SM_ROBOT_SPEC', 'SM_SOLE_BOX_CONFIG', 'SM_GPU_PINNED'):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES=str(a.gpu), PYTHONPATH=str(a.source / 'src'),
               SM_ROBOT='x2_ultra', MUJOCO_GL='egl', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    entry = a.source / 'scripts/evaluate_standing_skill.py'
    meta = dict(pid=os.getpid(), gpu=a.gpu, source=str(a.source), checkpoint=str(a.checkpoint),
                checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
                evaluator_sha256=hashlib.sha256(entry.read_bytes()).hexdigest(),
                started_epoch=time.time(), status='running', runs=[])
    def save():
        (a.out / 'diagnosis.json').write_text(json.dumps(meta, indent=2) + '\n')
    save()
    try:
        for name, flags in [('full', []), ('no_movement_feedback', ['--disable-moving-feedback']),
                            ('no_standing_feedback', ['--disable-standing-feedback'])]:
            result = a.out / (name + '.json')
            argv = [sys.executable, str(entry), '--ckpt', str(a.checkpoint), '--npz-dir',
                    str(a.dataset), '--out', str(result), '--seconds', '66', '--num-envs', '8',
                    '--seed', '0'] + flags
            row = dict(name=name, argv=argv, started_epoch=time.time())
            meta['runs'].append(row); save()
            with (a.out / (name + '.log')).open('x') as log:
                child = subprocess.Popen(argv, env=env, cwd=a.source, stdout=log, stderr=subprocess.STDOUT)
                row['pid'] = child.pid; save()
                try:
                    row['returncode'] = child.wait(timeout=300)
                except BaseException:
                    child.terminate(); child.wait(timeout=20)
                    raise
            if row['returncode']:
                raise RuntimeError(f'{name} evaluation failed')
            import numpy as np
            d = json.loads(result.read_text()); phases = []
            for j, (start, target) in enumerate(d['schedule']):
                end = d['schedule'][j+1][0] if j+1 < len(d['schedule']) else 66
                samples = [r for r in d['samples'] if start+1 <= r['time_s'] < end]
                v = np.array([r['velocity'] for r in samples])
                pos = np.array([r['root_position'] for r in samples])
                phases.append(dict(start=start, target=target, velocity=v.mean((0, 1)).tolist(),
                    worst_planar=float(np.linalg.norm(v[:, :, :2], axis=-1).mean(0).max()),
                    worst_net_displacement=float(np.linalg.norm(pos[-1, :, :2]-pos[0, :, :2], axis=-1).max())))
            row.update(survived=d['survived_without_reset'], resets=d['resets'], phases=phases,
                       stop_arm_metrics=d['stop_arm_metrics'], finished_epoch=time.time())
            save(); print(json.dumps(row), flush=True)
        meta.update(status='completed', finished_epoch=time.time()); save()
    except BaseException as error:
        meta.update(status='failed', error=repr(error)); save(); raise


if __name__ == '__main__':
    main()
