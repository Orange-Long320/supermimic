"""Run bounded SuperMimic model fitting and differentiable stop training, then physical acceptance."""

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
    for key in ('source', 'checkpoint', 'dataset', 'reference', 'standing-pose', 'standing-retention', 'baseline0', 'baseline7', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--gpu', type=int, default=2)
    p.add_argument('--iterations', type=int, default=120)
    p.add_argument('--num-envs', type=int, default=256)
    p.add_argument('--hours', type=float, default=.25)
    p.add_argument('--lr', type=float, default=None, help='Override the native checkpoint policy learning rate')
    p.add_argument('--train-base', action='store_true', help='Explicit full-standing-expert control')
    p.add_argument('--ema-decay', type=float, default=.99)
    p.add_argument('--compare-ema', action='store_true', help='Evaluate online and EMA checkpoints as a pair')
    a = p.parse_args()
    if a.gpu < 0 or a.iterations <= 0 or a.num_envs < 80 or a.hours <= 0 or (a.lr is not None and a.lr <= 0) or not 0 <= a.ema_decay < 1:
        p.error('Invalid GPU or training limits')
    py = Path('/home/2025-Long.xh/HHD/venvs/supermimic/bin/python')
    if Path(sys.prefix).resolve() != py.parent.parent.resolve() or os.path.realpath(sys.executable) != os.path.realpath(py) or not py.is_file():
        p.error(f'Use the dedicated server interpreter: {py}')
    a.source = a.source.resolve()
    a.checkpoint = a.checkpoint.resolve()
    a.dataset = a.dataset.resolve()
    a.reference = a.reference.resolve()
    a.standing_pose = a.standing_pose.resolve()
    a.standing_retention = a.standing_retention.resolve()
    a.baseline0 = a.baseline0.resolve()
    a.baseline7 = a.baseline7.resolve()
    a.out = a.out.resolve()
    entry = a.source / 'scripts/train_supermimic_stop.py'
    helper = a.source / 'scripts/supermimic_stop_imagination.py'
    ema_helper = a.source / 'scripts/supermimic_stop_ema.py'
    evaluator = a.source / 'scripts/evaluate_standing_skill.py'
    prior_controller = a.source / 'scripts/run_control_feedback.py'
    for path in (entry, helper, ema_helper, evaluator, prior_controller, a.checkpoint, a.reference, a.standing_pose, a.standing_retention,
                 a.baseline0, a.baseline7, a.dataset / 'walk.npz'):
        if not path.is_file():
            p.error(f'Required file missing: {path}')
    if not (a.source / 'src').is_dir():
        p.error('Source src directory is missing')
    a.out.mkdir(parents=True, exist_ok=False)
    record = {'status': 'initializing', 'controller_pid': os.getpid(), 'started_epoch': time.time(),
              'controller_argv': sys.argv, 'source': str(a.source), 'source_commit': a.source_commit,
              'gpu': a.gpu, 'evaluations': [], 'scope': 'bounded experimental validation; physical acceptance required'}
    record['control_matmul_precision'] = 'ieee'
    record['algorithm'] = 'supermimic_bptt'
    record['ema_decay'] = a.ema_decay
    record['compare_ema'] = a.compare_ema
    record['train_base'] = a.train_base
    record['actor_lr_override'] = a.lr
    record['standing_retention'] = str(a.standing_retention)
    record['standing_retention_sha256'] = digest(a.standing_retention)
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
        inputs = (entry, helper, ema_helper, evaluator, prior_controller, Path(__file__).resolve(), a.checkpoint, a.reference,
                  a.standing_pose, a.standing_retention, a.baseline0, a.baseline7, a.dataset / 'walk.npz')
        record.update(interpreter_probe=probe, platform=platform.platform(), cpu=platform.machine(),
                      source_sha256=source_files, input_sha256={str(path): digest(path) for path in inputs},
                      environment={key: env[key] for key in ('CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'SM_ROBOT', 'MUJOCO_GL',
                                                             'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'HF_HUB_OFFLINE',
                                                             'TRANSFORMERS_OFFLINE', 'XDG_CACHE_HOME')})
        argv = [str(py), str(entry), '--checkpoint', str(a.checkpoint), '--dataset', str(a.dataset),
                '--reference', str(a.reference), '--standing-pose', str(a.standing_pose),
                '--standing-retention', str(a.standing_retention), '--out', str(a.out / 'train'),
                '--iterations', str(a.iterations), '--num-envs', str(a.num_envs), '--hours', str(a.hours),
                '--ema-decay', str(a.ema_decay)]
        if a.lr is not None:
            argv += ['--lr', str(a.lr)]
        if a.train_base:
            argv.append('--train-base')
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
        import torch
        trained = torch.load(ck, map_location='cpu', weights_only=False)
        metadata = trained.get('supermimic_stop', {})
        if metadata.get('algorithm') != 'supermimic_bptt':
            raise RuntimeError('Only SuperMimic differentiable imagination checkpoints are allowed')
        required_updates = 2 if a.compare_ema else 1
        if metadata.get('actor_updates', 0) < required_updates:
            record.update(status='insufficient_actor_updates_for_ema' if a.compare_ema else 'not_accepted',
                          reason=f'Fewer than {required_updates} effective actor updates', final_checkpoint=str(ck),
                          final_checkpoint_sha256=checkpoint_sha, finished_epoch=time.time())
            save()
            return
        checkpoints = {'online': dict(path=str(ck), sha256=checkpoint_sha)}
        record['actor_updates'] = metadata['actor_updates']
        if a.compare_ema:
            ema_ck = ck.parent / 'ema' / ck.name
            ema_sha = digest(ema_ck)
            from supermimic_stop_ema import validate_pair
            averaged = torch.load(ema_ck, map_location='cpu', weights_only=False)
            validate_pair(trained, averaged, checkpoint_sha)
            checkpoints['ema'] = dict(path=str(ema_ck), sha256=ema_sha)
            record['ema_updates'] = averaged['policy_ema']['updates']
            record['paired_checkpoints'] = checkpoints
            del averaged
        record['checkpoints'] = checkpoints
        del trained
        record.update(status='evaluating', final_checkpoint=str(ck), final_checkpoint_sha256=checkpoint_sha,
                      final_iteration=int(ck.stem.split('_')[-1]))
        save()
        spec = importlib.util.spec_from_file_location('transition_prior_controller', prior_controller)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        import numpy as np
        ref = np.load(a.reference)
        pose = json.loads(a.standing_pose.read_text())['joint_positions_rad']
        arm_target = np.array([pose[str(n)] for n in ref['joint_names']
                               if any(k in str(n) for k in ('shoulder', 'elbow', 'wrist'))])
        baselines = {0: json.loads(a.baseline0.read_text()), 7: json.loads(a.baseline7.read_text())}
        paired_conditions = {}

        def evaluate(variant, ck, checkpoint_sha, name, seconds, seed, flags=(), baseline=None):
            if digest(ck) != checkpoint_sha:
                raise RuntimeError('Final checkpoint changed after training')
            result_path = a.out / f'{ck.stem}-{variant}-{name}.json'
            args = [str(py), str(evaluator), '--ckpt', str(ck), '--npz-dir', str(a.dataset), '--out', str(result_path),
                    '--seconds', str(seconds), '--num-envs', '8', '--seed', str(seed), '--ieee', *flags]
            item = {'variant': variant, 'name': name, 'argv': args, 'checkpoint_sha256': checkpoint_sha, 'status': 'running'}
            record['evaluations'].append(item)
            save()
            with result_path.with_suffix('.log').open('x') as output:
                subprocess.run(args, cwd=a.source, env=env, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=600)
            if digest(ck) != checkpoint_sha:
                raise RuntimeError('Final checkpoint changed during evaluation')
            data = json.loads(result_path.read_text())
            condition_keys = ('initial_state_sha256', 'initial_observation_sha256', 'schedule', 'seed',
                              'duration_s', 'num_envs', 'control_matmul_precision', 'velocity_config',
                              'action_joint_names', 'teacher_action_override', 'standing_feedback_disabled',
                              'moving_feedback_disabled')
            conditions = {key: data[key] for key in condition_keys}
            if name in paired_conditions and conditions != paired_conditions[name]:
                raise RuntimeError(f'Online and EMA evaluation conditions differ: {name}')
            paired_conditions[name] = conditions
            item['conditions'] = conditions
            item.update(status='complete', result_sha256=digest(result_path), summary=module.summarize(result_path, baseline, arm_target))
            save()
            return item['summary']

        # Every variant runs every case, including when the other variant fails.
        cases = [('full0', 66, 0, (), baselines[0]), ('full7', 66, 7, (), baselines[7]),
                 ('hold20', 20, 23, ('--standing-only',), None),
                 ('cold0', 3, 0, ('--standing-only',), None),
                 ('cold7', 3, 7, ('--standing-only',), None),
                 ('extended42', 42, 23, ('--extended-stop',), None),
                 ('direct58', 58, 41, ('--direct-switches',), None)]
        variants = ('online', 'ema') if a.compare_ema else ('online',)
        comparison = dict(algorithm='supermimic_bptt', checkpoints=checkpoints,
                          compare_ema=a.compare_ema, ema_decay=a.ema_decay, ema_updates=record.get('ema_updates'),
                          cases=[dict(name=n, seconds=t, seed=s, flags=f) for n,t,s,f,_ in cases],
                          variants={})
        for variant in variants:
            comparison['variants'][variant] = dict(results={})
        for name, seconds, seed, flags, baseline in cases:
            for variant in variants:
                info = checkpoints[variant]
                summary = evaluate(variant, Path(info['path']), info['sha256'], name, seconds, seed, flags, baseline)
                comparison['variants'][variant]['results'][name] = summary
                (a.out / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
        for variant, result in comparison['variants'].items():
            result['passed'] = all(item['passed'] for item in result['results'].values())
            if result['passed']:
                info = checkpoints[variant]
                accepted = a.out / (f'accepted_{variant}.pt' if a.compare_ema else 'accepted.pt')
                shutil.copy2(info['path'], accepted)
                if digest(accepted) != info['sha256']:
                    raise RuntimeError('Accepted checkpoint copy checksum mismatch')
                result.update(accepted_checkpoint=str(accepted), sha256=info['sha256'])
        comparison['verified_conditions'] = paired_conditions
        (a.out / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
        record.update(variant_passed={name: result['passed'] for name, result in comparison['variants'].items()},
                      comparison=str(a.out / 'comparison.json'))
        if a.compare_ema:
            record['status'] = 'paired_evaluation_complete'
        else:
            online = comparison['variants']['online']
            record['status'] = 'accepted' if online['passed'] else 'not_accepted'
            if online['passed']:
                record.update(accepted_checkpoint=str(Path(online['accepted_checkpoint']).resolve()),
                              accepted_sha256=online['sha256'])
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
