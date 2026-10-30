"""Compose compatible standing and movement feedback weights on server CPU.

This creates a candidate only. It does not train, evaluate, or grant acceptance.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import platform
import sys


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('base', 'stop', 'moving', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    interpreter = Path('/home/2025-Long.xh/HHD/venvs/supermimic/bin/python')
    if platform.system() != 'Linux' or Path(sys.prefix).resolve() != interpreter.parent.parent.resolve() or os.path.realpath(sys.executable) != os.path.realpath(interpreter):
        p.error(f'Use the dedicated server interpreter on Linux: {interpreter}')
    for key in ('base', 'stop', 'moving', 'out'):
        setattr(a, key, getattr(a, key).resolve())
    for path in (a.base, a.stop, a.moving):
        if not path.is_file():
            p.error(f'Input does not exist: {path}')
    if a.out.exists():
        p.error('Output file already exists')
    if not a.out.parent.is_dir():
        p.error('Output parent directory must exist')
    import torch
    paths = {'base': a.base, 'stop': a.stop, 'moving': a.moving}
    hashes = {key: sha(path) for key, path in paths.items()}
    base, stop, moving = (torch.load(paths[key], map_location='cpu', weights_only=False) for key in ('base', 'stop', 'moving'))
    stop_meta = stop.get('supermimic_stop')
    moving_meta = moving.get('supermimic_velocity_feedback')
    if not isinstance(stop_meta, dict) or not isinstance(moving_meta, dict):
        raise ValueError('Both component training records are required')
    for name, meta in (('stop', stop_meta), ('moving', moving_meta)):
        if meta.get('algorithm') != 'supermimic_bptt' or not isinstance(meta.get('actor_updates'), int) or meta['actor_updates'] <= 0:
            raise ValueError(f'{name} must have effective SuperMimic BPTT updates')
        if meta.get('source_sha256') != hashes['base']:
            raise ValueError(f'{name} was not initialized from the supplied base checkpoint')
    ref_sha = base.get('standing_reference', {}).get('sha256')
    if not ref_sha or any(meta.get('reference_sha256') != ref_sha for meta in (stop_meta, moving_meta)):
        raise ValueError('Base and component reference hashes differ or are missing')
    if any(ck.get('standing_reference', {}).get('sha256') != ref_sha for ck in (stop, moving)):
        raise ValueError('Component checkpoint standing references do not match base')
    if not isinstance(base.get('config'), dict) or base['config'] != stop.get('config') or base['config'] != moving.get('config'):
        raise ValueError('Checkpoint model/environment configurations differ')
    # Current stop checkpoints record the pose hash in the associated launch file.
    launch_path = a.stop.parent / 'launch.json'
    if not launch_path.is_file():
        raise ValueError('The stop checkpoint associated launch.json is missing')
    launch = json.loads(launch_path.read_text())
    launch_sha = sha(launch_path)
    if (launch.get('checkpoint_sha256') != hashes['base'] or launch.get('reference_sha256') != ref_sha
            or not stop_meta.get('entry_sha256') or launch.get('entry_sha256') != stop_meta['entry_sha256']):
        raise ValueError('Stop launch provenance does not match the checkpoint')
    pose_sha = launch.get('pose_sha256')
    if (not pose_sha or moving_meta.get('standing_pose_sha256') != pose_sha
            or (stop_meta.get('standing_pose_sha256') and stop_meta['standing_pose_sha256'] != pose_sha)):
        raise ValueError('Standing pose hashes differ or are missing')
    base_skill = base.get('standing_skill')
    if not isinstance(base_skill, dict) or base_skill.get('moving_feedback', False):
        raise ValueError('Base must contain a standing expert with moving feedback disabled')
    if stop.get('standing_skill') != base_skill:
        raise ValueError('Stop standing skill configuration differs from base')
    if moving.get('standing_skill') != dict(base_skill, moving_feedback=True):
        raise ValueError('Moving standing skill configuration differs beyond enabling feedback')
    bp, sp, mp = (ck.get('policy') for ck in (base, stop, moving))
    if not all(isinstance(state, dict) for state in (bp, sp, mp)):
        raise ValueError('Every input must contain policy state')
    standing_key, moving_key = 'standing_skill.feedback_weight', 'velocity_feedback_weight'
    if moving_key in bp or standing_key not in bp or set(sp) != set(bp) or set(mp) != set(bp) | {moving_key}:
        raise ValueError('Policy parameter keys do not match the permitted composition')
    for key, tensor in bp.items():
        if not all(isinstance(state[key], torch.Tensor) for state in (bp, sp, mp)):
            raise ValueError(f'Invalid policy tensor {key}')
        if not torch.equal(tensor, mp[key]):
            raise ValueError(f'Moving checkpoint changed base tensor: {key}')
        if key != standing_key and not torch.equal(tensor, sp[key]):
            raise ValueError(f'Standing checkpoint changed frozen tensor: {key}')
    for name, tensor in ((standing_key, sp[standing_key]), (moving_key, mp[moving_key])):
        if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
            raise ValueError(f'Non-finite or invalid component tensor: {name}')
    if sp[standing_key].shape != bp[standing_key].shape or sp[standing_key].dtype != bp[standing_key].dtype:
        raise ValueError('Standing feedback tensor has changed shape or dtype')
    if mp[moving_key].shape != bp[standing_key].shape or mp[moving_key].dtype != bp[standing_key].dtype:
        raise ValueError('Moving feedback tensor has an unexpected shape or dtype')
    if not isinstance(stop.get('wm'), dict):
        raise ValueError('Stop checkpoint has no world model')
    if any(sha(path) != hashes[key] for key, path in paths.items()) or sha(launch_path) != launch_sha:
        raise RuntimeError('An input changed during validation')
    ck = copy.deepcopy(base)
    ck['policy'] = {key: tensor.clone() for key, tensor in bp.items()}
    ck['policy'][standing_key] = sp[standing_key].clone()
    ck['policy'][moving_key] = mp[moving_key].clone()
    ck['standing_skill'] = dict(base_skill, moving_feedback=True)
    ck['wm'] = copy.deepcopy(stop['wm'])
    for key in ('policy_ema', 'supermimic_stop', 'supermimic_velocity_feedback'):
        ck.pop(key, None)
    ck['composed_training'] = {'stop': copy.deepcopy(stop_meta), 'moving': copy.deepcopy(moving_meta)}
    ck['supermimic_composed_feedback'] = {
        'algorithm': 'supermimic_bptt', 'reference_sha256': ref_sha, 'standing_pose_sha256': pose_sha,
        'source_common_sha': hashes['base'], 'actor_updates': stop_meta['actor_updates'] + moving_meta['actor_updates'],
        'entry_sha256': sha(Path(__file__).resolve()),
        'components': {key: {'path': str(paths[key]), 'sha256': hashes[key]} for key in paths},
        'stop_pose_evidence': {'source': 'launch.json', 'path': str(launch_path), 'sha256': launch_sha,
                               'standing_pose_sha256': pose_sha},
        'frozen_verification': {'stop_only': standing_key, 'moving_only': moving_key,
                                'other_policy_tensors': 'bitwise_equal_to_base', 'wm_source': 'stop'},
        'validation': 'composition only; physical acceptance pending',
    }
    # Open exclusively after all checks; a partial output is removed if serialization fails.
    try:
        with a.out.open('xb') as f:
            torch.save(ck, f)
    except FileExistsError:
        raise
    except BaseException:
        a.out.unlink(missing_ok=True)
        raise
    print(json.dumps({'output': str(a.out), 'sha256': sha(a.out), 'source_common_sha': hashes['base'],
                      'stop_sha256': hashes['stop'], 'moving_sha256': hashes['moving'],
                      'world_model_source': 'stop', 'physical_acceptance': 'pending'}), flush=True)


if __name__ == '__main__':
    main()
