"""Queue bounded keyboard continuation after a completed dual training run.

Run only on the project server, inside an immutable keyboard source snapshot.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['parent_run', 'source', 'dataset', 'out', 'entry', 'evaluator', 'stop_evaluator']:
        p.add_argument('--' + key.replace('_', '-'), type=Path, required=True)
    p.add_argument('--hours', type=float, default=6.)
    a = p.parse_args()
    if not 0 < a.hours <= 24:
        raise ValueError('Invalid bounded duration')
    a.out.mkdir(exist_ok=False)
    shutil.copy2(__file__, a.out / 'controller-as-run.py')
    state = dict(status='waiting_for_parent', pid=os.getpid(), parent_run=str(a.parent_run),
                 source=str(a.source), requested_train_hours=a.hours, gpus=[0, 2],
                 queued_epoch=time.time())

    def save():
        temp = a.out / 'controller.tmp'
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(a.out / 'controller.json')

    env = os.environ.copy()
    for key in ['PYTHONPATH', 'PYTHONHOME', 'SM_ROBOT_SPEC', 'SM_SOLE_BOX_CONFIG', 'SM_GPU_PINNED']:
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES='0,2', SM_ROBOT='x2_ultra', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1',
               MUJOCO_GL='egl', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               XDG_CACHE_HOME='/home/2025-Long.xh/HHD/data/supermimic/cache',
               WARP_CACHE_PATH='/home/2025-Long.xh/HHD/data/supermimic/cache/warp')
    py = '/home/2025-Long.xh/HHD/venvs/supermimic/bin/python'

    def wait_idle(gpus):
        while True:
            occupants = {str(g): subprocess.check_output(
                ['nvidia-smi', '-i', str(g), '--query-compute-apps=pid',
                 '--format=csv,noheader,nounits'], text=True).strip() for g in gpus}
            if not any(occupants.values()):
                return
            state.update(status='waiting_for_idle_gpus', occupants=occupants)
            save()
            time.sleep(15)

    def run(name, argv, run_env):
        state.update(status=name, argv=argv, stage_started_epoch=time.time())
        save()
        with (a.out / (name + '.log')).open('x') as log:
            child = subprocess.Popen(argv, cwd=a.source, env=run_env,
                                     stdout=log, stderr=subprocess.STDOUT)
            state['child_pid'] = child.pid
            save()
            code = child.wait()
        state['last_exit_code'] = code
        save()
        if code:
            raise RuntimeError(f'{name} failed with exit code {code}')

    save()
    try:
        deadline = time.monotonic() + 3 * 3600
        while True:
            parent = json.loads((a.parent_run / 'controller.json').read_text())
            if parent['status'] == 'completed_pending_performance_review':
                if not (a.parent_run / 'controller.exit').exists():
                    time.sleep(1)
                    continue
                if (a.parent_run / 'controller.exit').read_text().strip() != '0':
                    raise RuntimeError('Parent controller did not complete successfully')
                break
            proc = Path(f'/proc/{parent["pid"]}/cmdline')
            if (not proc.exists() or a.parent_run.name.encode() not in proc.read_bytes()
                    or time.monotonic() > deadline):
                raise RuntimeError('Parent stopped or timed out before successful completion')
            if parent['status'].startswith(('failed', 'cancelled')):
                raise RuntimeError('Parent cancelled or failed')
            time.sleep(15)

        records = [json.loads((a.parent_run / 'train' / f'rank-{k}.json').read_text())
                   for k in [0, 1]]
        capacity = json.loads((a.parent_run / 'dual-proof.json').read_text())
        manifest_sha = digest(a.dataset / 'manifest.json')
        if not capacity['passed'] or capacity['dataset_manifest_sha256'] != manifest_sha:
            raise ValueError('Parent capacity or dataset mismatch')
        if (any(r['status'] != 'completed' for r in records)
                or records[0]['final_parameter_hashes'] != records[1]['final_parameter_hashes']):
            raise ValueError('Parent did not complete synchronized dual training')
        for record in records:
            cfg = record['configuration']
            if (cfg['env']['num_envs'] != 8192 or cfg['wm_trainer']['batch_size'] != 2048
                    or cfg['policy_trainer']['batch_size'] != 2048
                    or not record['paired_supervision'] or not record['quiet_stop_cost']
                    or record['dataset_manifest_sha256'] != manifest_sha):
                raise ValueError('Continuation configuration differs from tested parent')
        final = a.parent_run / 'train/ckpt_latest.pt'
        if digest(final) != records[0]['checkpoint_sha256']:
            raise ValueError('Parent final checkpoint hash mismatch')
        fixed = a.out / 'resume-source.pt'
        with final.open('rb') as src, fixed.open('xb') as dst:
            shutil.copyfileobj(src, dst)
        checksum = digest(fixed)
        if checksum != records[0]['checkpoint_sha256']:
            raise ValueError('Checkpoint copy mismatch')
        for name in ['eval_66s.json', 'eval_stop42.json']:
            result = json.loads((a.parent_run / name).read_text())
            if Path(result['checkpoint']).resolve() != final.resolve():
                raise ValueError('Parent evaluation used a different checkpoint')
            shutil.copy2(a.parent_run / name, a.out / ('parent_' + name))
        proof = dict(capacity, checkpoint_sha256=checksum,
                     kind='continuation_from_completed_synchronized_dual_run',
                     parent_run=str(a.parent_run), parent_iteration=records[0]['iteration'],
                     parent_updates=records[0]['updates'],
                     parent_capacity_checkpoint_sha256=capacity['checkpoint_sha256'])
        proof_path = a.out / 'continuation-proof.json'
        proof_path.write_text(json.dumps(proof, indent=2) + '\n')
        state.update(resume_checkpoint=str(fixed), resume_sha256=checksum,
                     parent_iteration=records[0]['iteration'], parent_updates=records[0]['updates'])
        save()
        wait_idle([0, 2])
        run('train', [py, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node', '2',
                      str(a.entry), '--source', str(a.source), '--dataset', str(a.dataset),
                      '--checkpoint', str(fixed), '--out', str(a.out / 'train'),
                      '--phase', 'train', '--proof', str(proof_path), '--hours', str(a.hours),
                      '--quiet-stop-cost'], env)
        wait_idle([0])
        evaluation_env = dict(env, CUDA_VISIBLE_DEVICES='0', PYTHONPATH=str(a.source / 'src'))
        common = ['--ckpt', str(a.out / 'train/ckpt_latest.pt'), '--npz-dir', str(a.dataset),
                  '--num-envs', '8']
        run('eval_66s', [py, str(a.evaluator), *common, '--out', str(a.out / 'eval_66s.json'),
                         '--seconds', '66'], evaluation_env)
        run('eval_stop42', [py, str(a.stop_evaluator), *common,
                            '--out', str(a.out / 'eval_stop42.json'), '--seconds', '42',
                            '--extended-stop'], evaluation_env)
        state.update(status='completed_pending_performance_review', completed_epoch=time.time())
        save()
        (a.out / 'controller.exit').write_text('0')
    except BaseException as error:
        state.update(status='failed', error=repr(error), failed_epoch=time.time())
        save()
        raise


if __name__ == '__main__':
    main()
