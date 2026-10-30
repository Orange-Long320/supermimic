"""Run all seven existing physical gates on one saved SuperMimic candidate."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'checkpoint', 'dataset', 'reference', 'standing-pose', 'baseline0', 'baseline7', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--gpu', type=int, required=True)
    a = p.parse_args()
    py = Path('/home/2025-Long.xh/HHD/venvs/supermimic/bin/python')
    if Path(sys.prefix).resolve() != py.parent.parent.resolve() or os.path.realpath(sys.executable) != os.path.realpath(py) or not py.is_file():
        p.error(f'Use the dedicated server interpreter: {py}')
    if a.gpu < 0:
        p.error('GPU must be nonnegative')
    for key in ('source', 'checkpoint', 'dataset', 'reference', 'standing_pose', 'baseline0', 'baseline7', 'out'):
        setattr(a, key, getattr(a, key).resolve())
    evaluator = a.source / 'scripts/evaluate_standing_skill.py'
    controller = a.source / 'scripts/run_control_feedback.py'
    inputs = (evaluator, controller, a.checkpoint, a.reference, a.standing_pose, a.baseline0, a.baseline7, a.dataset / 'walk.npz')
    for path in inputs:
        if not path.is_file():
            p.error(f'Missing input: {path}')
    if not (a.source / 'src').is_dir():
        p.error('Source src directory is missing')
    if a.out.exists():
        p.error('Output directory already exists')
    a.out.mkdir(parents=True, exist_ok=False)
    record = dict(status='initializing', controller_pid=os.getpid(), controller_argv=sys.argv,
        source=str(a.source), source_commit=a.source_commit, gpu=a.gpu, started_epoch=time.time(),
        control_matmul_precision='ieee', evaluations=[], scope='Seven independent physical gates on the saved candidate')

    def save():
        temp = a.out / 'controller.json.tmp'
        temp.write_text(json.dumps(record, indent=2)+'\n')
        temp.replace(a.out / 'controller.json')

    child = None
    save()
    try:
        processes = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-compute-apps=pid',
            '--format=csv,noheader'], text=True).strip()
        gpu_state = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
            '--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True).strip()
        gpu_details = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu)], text=True)
        record.update(gpu_query=gpu_state, gpu_compute_processes=processes, gpu_details=gpu_details)
        parts = [v.strip() for v in gpu_state.split(',')]
        if len(parts) != 7 or int(parts[0]) != a.gpu or processes or float(parts[5]) > 100 or float(parts[6]) != 0:
            raise RuntimeError('Requested GPU is not idle; no evaluation started')
        env = os.environ.copy()
        for key in ('PYTHONHOME', 'SM_ROBOT_SPEC', 'SM_SOLE_BOX_CONFIG', 'SM_GPU_PINNED'):
            env.pop(key, None)
        env.update(CUDA_VISIBLE_DEVICES=str(a.gpu), PYTHONPATH=str(a.source / 'src'), SM_ROBOT='x2_ultra',
            MUJOCO_GL='egl', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', HF_HUB_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1', XDG_CACHE_HOME=str(a.source.parent.parent / 'cache'), SM_SOURCE_COMMIT=a.source_commit)
        probe_code = ('import json,sys,torch; print(json.dumps({"executable":sys.executable,"python":sys.version,'
            '"torch":torch.__version__,"cuda_runtime":torch.version.cuda,"cuda_available":torch.cuda.is_available(),'
            '"device_count":torch.cuda.device_count(),"device_name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))')
        probe = json.loads(subprocess.check_output([str(py), '-c', probe_code], cwd=a.source, env=env, text=True, timeout=60))
        if not probe['cuda_available'] or probe['device_count'] != 1:
            raise RuntimeError('Dedicated interpreter must see exactly one CUDA device')
        record.update(interpreter_probe=probe, platform=platform.platform(), cpu=platform.machine(),
            source_sha256={str(path.relative_to(a.source)): digest(path) for path in sorted((a.source / 'src').rglob('*'))
                           if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc'},
            input_sha256={str(path): digest(path) for path in (*inputs, Path(__file__).resolve())},
            environment={key: env[key] for key in ('CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'SM_ROBOT', 'MUJOCO_GL',
                'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'XDG_CACHE_HOME')})
        import torch
        import numpy as np
        ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
        eligible = [(key, ck[key]) for key in ('supermimic_stop', 'supermimic_velocity_feedback')
                    if isinstance(ck.get(key), dict) and ck[key].get('algorithm') == 'supermimic_bptt']
        if len(eligible) != 1:
            raise ValueError('Checkpoint must identify exactly one SuperMimic BPTT candidate type')
        kind, metadata = eligible[0]
        if not metadata.get('actor_updates', 0):
            raise ValueError('Candidate has no effective actor updates')
        if metadata.get('reference_sha256') != digest(a.reference):
            raise ValueError('Checkpoint standing reference mismatch')
        if metadata.get('standing_pose_sha256') and metadata['standing_pose_sha256'] != digest(a.standing_pose):
            raise ValueError('Checkpoint standing pose mismatch')
        del ck
        checkpoint_sha = digest(a.checkpoint)
        record.update(status='evaluating', algorithm='supermimic_bptt', candidate_type=kind,
                      candidate_checkpoint=str(a.checkpoint), candidate_sha256=checkpoint_sha)
        save()
        spec = importlib.util.spec_from_file_location('candidate_existing_gates', controller)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with np.load(a.reference) as ref:
            pose = json.loads(a.standing_pose.read_text())['joint_positions_rad']
            arm_target = np.array([pose[str(n)] for n in ref['joint_names']
                                   if any(k in str(n) for k in ('shoulder', 'elbow', 'wrist'))])
        baselines = {0: json.loads(a.baseline0.read_text()), 7: json.loads(a.baseline7.read_text())}
        cases = [('full0', 66, 0, (), baselines[0]), ('full7', 66, 7, (), baselines[7]),
                 ('hold20', 20, 23, ('--standing-only',), None), ('cold0', 3, 0, ('--standing-only',), None),
                 ('cold7', 3, 7, ('--standing-only',), None), ('extended42', 42, 23, ('--extended-stop',), None),
                 ('direct58', 58, 41, ('--direct-switches',), None)]
        for name, seconds, seed, flags, baseline in cases:
            if digest(a.checkpoint) != checkpoint_sha:
                raise RuntimeError('Candidate changed during evaluation')
            result = a.out / f'{name}.json'
            argv = [str(py), str(evaluator), '--ckpt', str(a.checkpoint), '--npz-dir', str(a.dataset),
                    '--out', str(result), '--seconds', str(seconds), '--num-envs', '8', '--seed', str(seed), '--ieee', *flags]
            item = dict(name=name, status='running', argv=argv, checkpoint_sha256=checkpoint_sha,
                        started_epoch=time.time(), log=str(result.with_suffix('.log')))
            record['evaluations'].append(item); save()
            with result.with_suffix('.log').open('x') as log:
                child = subprocess.Popen(argv, cwd=a.source, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                item['pid'] = child.pid; save()
                try:
                    code = child.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        child.kill(); child.wait()
                    raise RuntimeError(f'{name} evaluation timed out')
            item.update(returncode=code, finished_epoch=time.time())
            if code != 0:
                raise RuntimeError(f'{name} evaluation failed; see its log')
            if digest(a.checkpoint) != checkpoint_sha:
                raise RuntimeError('Candidate changed during evaluation')
            data = json.loads(result.read_text())
            if data.get('control_matmul_precision') != 'ieee' or any(data.get(k) is not False for k in
                    ('teacher_action_override', 'standing_feedback_disabled', 'moving_feedback_disabled')):
                raise RuntimeError(f'{name} used an unapproved runtime override or precision')
            if data.get('seed') != seed or data.get('duration_s') != seconds or data.get('num_envs') != 8:
                raise RuntimeError(f'{name} evaluation conditions do not match')
            item.update(status='complete', result_sha256=digest(result), summary=module.summarize(result, baseline, arm_target))
            save()
        passed = len(record['evaluations']) == 7 and all(item['summary']['passed'] for item in record['evaluations'])
        if passed:
            accepted = a.out / 'accepted.pt'
            shutil.copy2(a.checkpoint, accepted)
            if digest(accepted) != checkpoint_sha:
                raise RuntimeError('Accepted checkpoint checksum mismatch')
            record.update(status='accepted', accepted_checkpoint=str(accepted), accepted_sha256=checkpoint_sha)
        else:
            record['status'] = 'not_accepted'
        record['finished_epoch'] = time.time(); save()
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
        record.update(status='failed', error=repr(error), finished_epoch=time.time()); save()
        raise


if __name__ == '__main__':
    main()
