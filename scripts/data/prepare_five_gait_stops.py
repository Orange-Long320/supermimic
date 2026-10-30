"""Build exploratory stop-and-hold clips for five gait references on the server."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--pose', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--iteration', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    if a.out.exists():
        raise FileExistsError(a.out)
    original = json.loads((a.source / 'manifest.json').read_text())
    a.out.mkdir(parents=True)
    data = a.out / 'dataset'
    data.mkdir()
    scripts = Path(__file__).resolve().parent.parent
    clips = []
    for item in original['clips']:
        name = item['name']
        source = a.source / (name + '.npz')
        if sha(source) != item['sha256']:
            raise ValueError('Source hash mismatch: ' + name)
        shutil.copy2(source, data / source.name)
        clips.append({'name': name, 'sha256': sha(source), 'source': str(source), 'type': 'prior_reference'})
        if item['type'] != 'prior_gait':
            continue
        with np.load(source, allow_pickle=False) as z:
            frames = len(z['joint_pos'])
        release = min(frames // 2, frames - 31)
        if release < 20:
            raise ValueError('Reference too short: ' + name)
        entry = a.out / (name + '-entry')
        entry.mkdir()
        shutil.copy2(source, entry / 'candidate.npz')
        (entry / 'manifest.json').write_text(json.dumps({'release_frame': release, 'parameters': {'release_lead': 10}}))
        result = a.out / (name + '-stop')
        subprocess.run([sys.executable, str(scripts / 'data/build_natural_stop_preview.py'), '--walking', str(entry), '--pose', str(a.pose), '--out', str(result), '--stop-frames', '30', '--hold-frames', '200', '--match-entry-velocity', '--soft-entry'], check=True)
        subprocess.run([sys.executable, str(scripts / 'evaluation/audit_fast_stop_reference.py'), '--reference', str(result)], check=True)
        audit = json.loads((result / 'speed-audit.json').read_text())
        if audit['brake_max_hard_excess_rad'] > 1e-4 or audit['brake_max_speed_ratio'] > 1.0:
            raise ValueError('Stop fails hard limit or actuator speed screen: ' + name)
        meta = json.loads((result / 'manifest.json').read_text())
        with np.load(result / 'natural-stop.npz', allow_pickle=False) as z:
            q = z['joint_pos']
            i = meta['brake_start_frame']
            step = np.sqrt(np.mean(np.diff(q, axis=0)**2, axis=1))
            if step[i] > 1.6 * step[i-1] + .005:
                raise ValueError('First stop step discontinuity: ' + name)
            root = list(z['body_names']).index('pelvis')
            height = z['body_pos_w'][:, root, 2]
            dh = np.diff(height)
            if abs(dh[i] - dh[i-1]) > .005:
                raise ValueError('First stop root height velocity discontinuity: ' + name)
            audit['entry_root_height_step_m'] = float(dh[i])
            audit['previous_root_height_step_m'] = float(dh[i-1])
            audit['brake_max_root_height_step_m'] = float(np.max(np.abs(dh[i:meta['standing_frame']])))
        name_out = name + '_stop'
        shutil.copy2(result / 'natural-stop.npz', data / (name_out + '.npz'))
        clips.append({'name': name_out, 'sha256': sha(data / (name_out + '.npz')), 'source': str(source), 'source_sha256': sha(source), 'release_frame': release, 'audit': audit, 'type': 'exploratory_stop'})
    if len([c for c in clips if c['type'] == 'exploratory_stop']) != 5:
        raise ValueError('Expected five distinct gait stops')
    manifest = {'clips': clips, 'initial_checkpoint_sha256': sha(a.checkpoint), 'initial_checkpoint_iteration': a.iteration,
                'pose_sha256': sha(a.pose), 'scope': 'Kinematic references; contact forces and physical stop stability not validated'}
    (data / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'clips': len(clips), 'stops': 5, 'checkpoint_sha256': manifest['initial_checkpoint_sha256']}, indent=2))


if __name__ == '__main__':
    main()
