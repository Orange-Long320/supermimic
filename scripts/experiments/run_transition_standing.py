"""Run one bounded transition experiment and independently gate its final checkpoint."""

from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'checkpoint', 'dataset', 'reference', 'standing-pose', 'baseline0', 'baseline7', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--gpu', type=int, default=2)
    p.add_argument('--iterations', type=int, default=120)
    p.add_argument('--num-envs', type=int, default=512)
    p.add_argument('--hours', type=float, default=.2)
    a = p.parse_args()
    if a.gpu < 0 or a.iterations <= 0 or a.num_envs < 80 or a.hours <= 0:
        p.error('Invalid GPU or training limits')
    py = Path('/home/2025-Long.xh/HHD/venvs/supermimic/bin/python')
    if Path(sys.prefix).resolve() != py.parent.parent.resolve() or os.path.realpath(sys.executable) != os.path.realpath(py) or not py.is_file():
        p.error(f'Use the dedicated server interpreter: {py}')
    a.source = a.source.resolve()
    a.checkpoint = a.checkpoint.resolve()
    a.dataset = a.dataset.resolve()
    a.reference = a.reference.resolve()
    a.standing_pose = a.standing_pose.resolve()
    a.baseline0 = a.baseline0.resolve()
    a.baseline7 = a.baseline7.resolve()
    a.out = a.out.resolve()
    entry = a.source / 'scripts/train_transition_standing.py'
    evaluator = a.source / 'scripts/evaluate_standing_skill.py'
    prior_controller = a.source / 'scripts/run_control_feedback.py'
    for path in (entry, evaluator, prior_controller, a.checkpoint, a.reference, a.standing_pose,
                 a.baseline0, a.baseline7, a.dataset / 'walk.npz'):
        if not path.is_file():
            p.error(f'Required file missing: {path}')
    if not (a.source / 'src').is_dir():
        p.error('Source src directory is missing')
    a.out.mkdir(parents=True, exist_ok=False)
    record = {'status': 'initializing', 'controller_pid': os.getpid(), 'started_epoch': time.time(),
              'controller_argv': sys.argv, 'source': str(a.source), 'source_commit': a.source_commit,
              'gpu': a.gpu, 'evaluations': [], 'scope': 'bounded experimental validation; physical acceptance required'}
    train = None
    log = None

    def save():
        tmp = a.out / 'controller.json.tmp'
        tmp.write_text(json.dumps(record, indent=2) + '\n')
        tmp.replace(a.out / 'controller.json')

    def stop_child():
        if train is not None and train.poll() is None:
            train.terminate()
            try:
                train.wait(timeout=15)
            except subprocess.TimeoutExpired:
                train.kill()
                train.wait()

    save()
    try:
        gpu_processes = subprocess.check_output(
            ['nvidia-smi', '-i', str(a.gpu), '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        gpu_state = subprocess.check_output(
            ['nvidia-smi', '-i', str(a.gpu), '--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu',
             '--format=csv,noheader,nounits'], text=True).strip()
        gpu_details = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu)], text=True)
        record.update(gpu_query=gpu_state, gpu_compute_processes=gpu_processes, gpu_details=gpu_details)
        parts = [v.strip() for v in gpu_state.split(',')]
        if len(parts) != 7 or int(parts[0]) != a.gpu or gpu_processes or float(parts[5]) > 100 or float(parts[6]) != 0:
            raise RuntimeError('Requested GPU is not idle; no training was started')
        env = os.environ.copy()
        for key in ('PYTHONHOME', 'SM_ROBOT_SPEC', 'SM_SOLE_BOX_CONFIG', 'SM_GPU_PINNED'):
            env.pop(key, None)
        env.update(CUDA_VISIBLE_DEVICES=str(a.gpu), PYTHONPATH=str(a.source / 'src'), SM_ROBOT='x2_ultra',
                   MUJOCO_GL='egl', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', XDG_CACHE_HOME=str(a.source.parent.parent / 'cache'))
        env['SM_SOURCE_COMMIT'] = a.source_commit
        probe_code = ('import json,sys,torch; print(json.dumps({"executable":sys.executable,"python":sys.version,'
                      '"torch":torch.__version__,"cuda_runtime":torch.version.cuda,"cuda_available":torch.cuda.is_available(),'
                      '"device_count":torch.cuda.device_count(),"device_name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))')
        probe = json.loads(subprocess.check_output([str(py), '-c', probe_code], cwd=a.source, env=env, text=True, timeout=60))
        if not probe['cuda_available'] or probe['device_count'] != 1:
            raise RuntimeError('Dedicated interpreter cannot see exactly one CUDA device')
        source_files = {str(path.relative_to(a.source)): digest(path) for path in sorted((a.source / 'src').rglob('*'))
                        if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc'}
        inputs = (entry, evaluator, prior_controller, Path(__file__).resolve(), a.checkpoint, a.reference,
                  a.standing_pose, a.baseline0, a.baseline7, a.dataset / 'walk.npz')
        record.update(interpreter_probe=probe, platform=platform.platform(), cpu=platform.machine(),
                      source_sha256=source_files, input_sha256={str(path): digest(path) for path in inputs},
                      environment={key: env[key] for key in ('CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'SM_ROBOT', 'MUJOCO_GL',
                                                             'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'HF_HUB_OFFLINE',
                                                             'TRANSFORMERS_OFFLINE', 'XDG_CACHE_HOME')})
        argv = [str(py), str(entry), '--checkpoint', str(a.checkpoint), '--dataset', str(a.dataset),
                '--reference', str(a.reference), '--standing-pose', str(a.standing_pose), '--out', str(a.out / 'train'),
                '--iterations', str(a.iterations), '--num-envs', str(a.num_envs), '--hours', str(a.hours)]
        record['training_argv'] = argv
        save()
        log = (a.out / 'train.log').open('x')
        train = subprocess.Popen(argv, cwd=a.source, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record.update(status='training', train_pid=train.pid)
        save()
        try:
            code = train.wait(timeout=a.hours * 3600 + 600)
        except subprocess.TimeoutExpired:
            stop_child()
            record.update(status='training_timeout', train_returncode=train.returncode, finished_epoch=time.time())
            save()
            return
        record['train_returncode'] = code
        if code != 0:
            record.update(status='training_failed', finished_epoch=time.time())
            save()
            return
        ready = list((a.out / 'train').glob('candidate_*.ready'))
        if not ready:
            raise RuntimeError('Training completed without a ready checkpoint')
        final_ready = max(ready, key=lambda path: int(path.stem.split('_')[-1]))
        ck = final_ready.with_suffix('.pt')
        if not ck.is_file() or final_ready.read_text().strip() != ck.stem.split('_')[-1]:
            raise RuntimeError('Final ready marker does not match its checkpoint')
        checkpoint_sha = digest(ck)
        record.update(status='evaluating', final_checkpoint=str(ck), final_checkpoint_sha256=checkpoint_sha,
                      final_iteration=int(ck.stem.split('_')[-1]))
        save()
        spec = importlib.util.spec_from_file_location('transition_prior_controller', prior_controller)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        import numpy as np
        ref = np.load(a.reference)
        arm_target = np.array([ref['joint_pos'][0, list(ref['joint_names']).index(n)] for n in ref['joint_names']
                               if any(k in n for k in ('shoulder', 'elbow', 'wrist'))])
        baselines = {0: json.loads(a.baseline0.read_text()), 7: json.loads(a.baseline7.read_text())}

        def evaluate(name, seconds, seed, flags=(), baseline=None):
            if digest(ck) != checkpoint_sha:
                raise RuntimeError('Final checkpoint changed after training')
            result_path = a.out / f'{ck.stem}-{name}.json'
            args = [str(py), str(evaluator), '--ckpt', str(ck), '--npz-dir', str(a.dataset), '--out', str(result_path),
                    '--seconds', str(seconds), '--num-envs', '8', '--seed', str(seed), *flags]
            item = {'name': name, 'argv': args, 'checkpoint_sha256': checkpoint_sha, 'status': 'running'}
            record['evaluations'].append(item)
            save()
            with result_path.with_suffix('.log').open('x') as output:
                subprocess.run(args, cwd=a.source, env=env, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=600)
            if digest(ck) != checkpoint_sha:
                raise RuntimeError('Final checkpoint changed during evaluation')
            item.update(status='complete', result_sha256=digest(result_path), summary=module.summarize(result_path, baseline, arm_target))
            save()
            return item['summary']['passed']

        full0 = evaluate('full0', 66, 0, baseline=baselines[0])
        full7 = evaluate('full7', 66, 7, baseline=baselines[7])
        hold20 = evaluate('hold20', 20, 23, flags=['--standing-only'])
        passed = full0 and full7 and hold20
        if passed:
            cold0 = evaluate('cold0', 3, 0, flags=['--standing-only'])
            cold7 = evaluate('cold7', 3, 7, flags=['--standing-only'])
            extended = evaluate('extended42', 42, 23, flags=['--extended-stop'])
            direct = evaluate('direct58', 58, 41, flags=['--direct-switches'])
            passed = cold0 and cold7 and extended and direct
        if passed:
            accepted = a.out / 'accepted.pt'
            shutil.copy2(ck, accepted)
            if digest(accepted) != checkpoint_sha:
                raise RuntimeError('Accepted checkpoint copy checksum mismatch')
            record.update(status='accepted', accepted_checkpoint=str(accepted), accepted_sha256=checkpoint_sha)
        else:
            record['status'] = 'not_accepted'
        record['finished_epoch'] = time.time()
        save()
    except BaseException as error:
        stop_child()
        record.update(status='failed', error=repr(error), finished_epoch=time.time())
        if train is not None:
            record['train_returncode'] = train.returncode
        save()
        raise
    finally:
        if log is not None:
            log.close()


if __name__ == '__main__':
    main()
