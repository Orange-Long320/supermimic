"""Compare two policies on fixed real control states, without simulation or training."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source-checkpoint', 'candidate', 'control-retention', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--chunk', type=int, default=128, choices=(128, 256))
    a = p.parse_args()
    interpreter = Path('/home/2025-Long.xh/HHD/venvs/supermimic/bin/python')
    if platform.system() != 'Linux' or Path(sys.prefix).resolve() != interpreter.parent.parent.resolve() or os.path.realpath(sys.executable) != os.path.realpath(interpreter):
        p.error(f'Use the dedicated server interpreter on Linux: {interpreter}')
    for key in ('source_checkpoint', 'candidate', 'control_retention', 'out'):
        setattr(a, key, getattr(a, key).resolve())
    for path in (a.source_checkpoint, a.candidate, a.control_retention):
        if not path.is_file():
            p.error(f'Missing input: {path}')
    if a.out.exists():
        p.error('Output directory already exists')
    a.out.mkdir(parents=True, exist_ok=False)
    source_hash, candidate_hash, retention_hash = (sha(p) for p in (a.source_checkpoint, a.candidate, a.control_retention))
    report = {'status': 'initializing', 'argv': sys.argv, 'started_epoch': time.time(),
              'source_checkpoint_sha256': source_hash, 'candidate_sha256': candidate_hash,
              'control_retention_sha256': retention_hash,
              'entry_sha256': sha(Path(__file__).resolve()), 'chunk': a.chunk,
              'scope': 'same-state action comparison; does not establish the sole cause of physical drift'}
    def save():
        tmp = a.out / 'audit.json.tmp'
        tmp.write_text(json.dumps(report, indent=2) + '\n')
        tmp.replace(a.out / 'audit.json')
    save()
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['SM_ROBOT'] = 'x2_ultra'
        for key in ('SM_ROBOT_SPEC', 'SM_SOLE_BOX_CONFIG', 'SM_GPU_PINNED'):
            os.environ.pop(key, None)
        import torch
        import supermimic
        from supermimic.eval.benchmark import load_policy_from_checkpoint
        from supermimic.robots.dimensions import PROFILE
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
        torch.backends.cudnn.fp32_precision = 'ieee'
        source_ck = torch.load(a.source_checkpoint, map_location='cpu', weights_only=False)
        candidate_ck = torch.load(a.candidate, map_location='cpu', weights_only=False)
        retention = torch.load(a.control_retention, map_location='cpu', weights_only=False)
        if retention.get('source_sha256') != source_hash:
            raise ValueError('Control retention does not originate from the supplied source checkpoint')
        if retention.get('preserve_moving_feedback') is not True:
            raise ValueError('Control retention must contain complete source-policy actions')
        meta = candidate_ck.get('supermimic_stop', {})
        if meta.get('source_sha256') != source_hash or meta.get('algorithm') != 'supermimic_bptt':
            raise ValueError('Candidate training record does not originate from source')
        reference_hash = retention.get('reference_sha256')
        pose_hash = retention.get('standing_pose_sha256')
        source_meta = source_ck.get('supermimic_composed_feedback', {})
        if (not reference_hash or not pose_hash or source_meta.get('reference_sha256') != reference_hash
                or source_meta.get('standing_pose_sha256') != pose_hash or meta.get('reference_sha256') != reference_hash
                or source_ck.get('standing_reference', {}).get('sha256') != reference_hash
                or candidate_ck.get('standing_reference', {}).get('sha256') != reference_hash):
            raise ValueError('Source, candidate and retention reference/pose provenance differs')
        if meta.get('standing_pose_sha256'):
            if meta['standing_pose_sha256'] != pose_hash:
                raise ValueError('Candidate pose hash differs from retention')
        else:
            launch_path = a.candidate.parent / 'launch.json'
            if not launch_path.is_file():
                raise ValueError('Candidate has no pose hash or associated launch.json')
            launch = json.loads(launch_path.read_text())
            if (launch.get('checkpoint_sha256') != source_hash or launch.get('reference_sha256') != reference_hash
                    or launch.get('pose_sha256') != pose_hash or not meta.get('entry_sha256')
                    or launch.get('entry_sha256') != meta['entry_sha256']):
                raise ValueError('Candidate launch provenance does not match source, pose or training entry')
            report['candidate_launch_sha256'] = sha(launch_path)
        if source_ck.get('config') != candidate_ck.get('config'):
            raise ValueError('Policy configurations differ')
        keys = ('obs', 'command', 'feedback', 'source_action', 'age', 'env_id')
        if any(k not in retention or not isinstance(retention[k], torch.Tensor) for k in keys):
            raise ValueError('Retention is missing required tensors')
        obs, command, feedback, stored, age, env_id = (retention[k] for k in keys)
        n = len(obs)
        if (obs.ndim != 2 or n == 0 or command.shape != (n, 3) or feedback.shape != (n, 6)
                or stored.ndim != 2 or len(stored) != n or age.shape != (n,) or env_id.shape != (n,)):
            raise ValueError('Retention tensor dimensions do not match')
        if any(t.device.type != 'cpu' or t.dtype != torch.float32 or not torch.isfinite(t).all()
               for t in (obs, command, feedback, stored)):
            raise ValueError('Policy inputs and stored actions must be finite CPU float32 tensors')
        if not torch.isfinite(age).all() or (age < 0).any() or (age >= 66).any():
            raise ValueError('Retention sample times are outside the 66-second protocol')
        if len(PROFILE['joint_names']) != stored.shape[1]:
            raise ValueError('Action dimensions do not match X2 joint names')
        source, _ = load_policy_from_checkpoint(a.source_checkpoint, 'cpu')
        candidate, _ = load_policy_from_checkpoint(a.candidate, 'cpu')
        source.eval(); candidate.eval()
        package_root = Path(supermimic.__file__).resolve().parent
        report.update(reference_sha256=reference_hash, standing_pose_sha256=pose_hash,
                      source_code_sha256={str(path.relative_to(package_root)): sha(path)
                                          for path in sorted(package_root.rglob('*.py'))},
                      interpreter=sys.executable, python=sys.version, torch=torch.__version__, device='cpu',
                      torch_threads=torch.get_num_threads(), matmul_precision='CPU float32; no autocast',
                      sample_count=n)
        save()
        source_actions = []
        with torch.no_grad():
            for start in range(0, n, a.chunk):
                sl = slice(start, min(start + a.chunk, n))
                source_actions.append(source.velocity_action(obs[sl], command[sl], feedback[sl]).cpu())
        source_actions = torch.cat(source_actions)
        if source_actions.shape != stored.shape or not torch.isfinite(source_actions).all():
            raise ValueError('Recomputed source action shape or values are invalid')
        source_delta = (source_actions - stored).abs()
        report['source_action_check'] = {'max_abs': float(source_delta.max()),
                                         'mean_abs': float(source_delta.mean()), 'tolerance': 1e-4,
                                         'note': 'CPU float32 recomputation compared with recorded source actions'}
        if report['source_action_check']['max_abs'] > 1e-4:
            report.update(status='source_action_mismatch', finished_epoch=time.time())
            save()
            return
        candidate_actions = []
        with torch.no_grad():
            for start in range(0, n, a.chunk):
                sl = slice(start, min(start + a.chunk, n))
                candidate_actions.append(candidate.velocity_action(obs[sl], command[sl], feedback[sl]).cpu())
        candidate_actions = torch.cat(candidate_actions)
        if candidate_actions.shape != source_actions.shape or not torch.isfinite(candidate_actions).all():
            raise ValueError('Candidate action shape or values are invalid')
        delta = candidate_actions - source_actions
        rms = delta.square().mean(-1).sqrt()
        maxabs = delta.abs().max(-1).values
        def stats(mask):
            count = int(mask.sum())
            if not count:
                return {'count': 0}
            r, m = rms[mask], maxabs[mask]
            return {'count': count, 'sample_rms': {'mean': float(r.mean()), 'p95': float(torch.quantile(r, .95)), 'max': float(r.max())},
                    'sample_max_abs': {'mean': float(m.mean()), 'p95': float(torch.quantile(m, .95)), 'max': float(m.max())},
                    'fraction_sample_rms_gt_0_03': float((r > .03).float().mean()),
                    'fraction_sample_max_abs_gt_0_15': float((m > .15).float().mean())}
        starts = (0., 3., 10., 15., 22., 27., 34., 39., 46., 51., 58.)
        labels = ('initial_stop', 'forward', 'forward_stop', 'backward', 'backward_stop', 'left',
                  'left_stop', 'right', 'right_stop', 'jog', 'jog_stop')
        phases = {}
        for i, (start, name) in enumerate(zip(starts, labels)):
            end = starts[i + 1] if i + 1 < len(starts) else 66.
            phases[name] = {'start_s': start, 'end_s': end, **stats((age >= start) & (age < end))}
        norm = command.square().sum(-1).sqrt()
        zero = norm == 0
        near = (norm > 0) & (norm < .1)
        full = source.standing_skill.gate(command) == 0
        other = ~(zero | near | full)
        modes = {'zero': stats(zero), 'near_zero': stats(near), 'full_movement': stats(full), 'other': stats(other)}
        worst_index = int(rms.argmax())
        worst_abs_index = int(maxabs.argmax())
        def sample(index):
            return {'index': index, 'env_id': int(env_id[index]), 'time_s': float(age[index]),
                    'command': command[index].tolist(), 'sample_rms': float(rms[index]),
                    'sample_max_abs': float(maxabs[index]),
                    'joint_action_deltas': dict(zip(PROFILE['joint_names'], delta[index].tolist()))}
        if sha(a.source_checkpoint) != source_hash or sha(a.candidate) != candidate_hash or sha(a.control_retention) != retention_hash:
            raise RuntimeError('An input changed during the audit')
        pure_equal = bool(torch.equal(candidate_actions[full], source_actions[full]))
        report.update(status='complete' if pure_equal else 'pure_movement_mismatch', phase_statistics=phases, mode_statistics=modes,
                      overall=stats(torch.ones(n, dtype=torch.bool)),
                      pure_movement_bitwise_equal=pure_equal,
                      pure_movement_unequal_samples=int((candidate_actions[full] != source_actions[full]).any(-1).sum()),
                      any_per_sample_rms_gt_0_03=bool((rms > .03).any()),
                      any_per_sample_max_abs_gt_0_15=bool((maxabs > .15).any()),
                      worst_by_rms=sample(worst_index), worst_by_max_abs=sample(worst_abs_index),
                      finished_epoch=time.time())
        save()
    except BaseException as error:
        report.update(status='failed', error=repr(error), finished_epoch=time.time())
        save()
        raise


if __name__ == '__main__':
    main()
