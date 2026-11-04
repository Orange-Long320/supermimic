"""Score official-simulator turn trajectories without changing control inputs."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def summarize(frames):
    segments = []
    starts = [0] + [i for i in range(1, len(frames))
                    if frames[i]['mode'] != frames[i-1]['mode']] + [len(frames)]
    for start, end in zip(starts[:-1], starts[1:]):
        rows = frames[start:end]
        t = np.array([r['elapsed'] for r in rows])
        xy = np.array([r['root'][:2] for r in rows])
        reference = np.array([r['reference_root_position'][:2] for r in rows])
        yaw = np.unwrap(Rotation.from_quat(
            np.array([r['q'] for r in rows])[:, [1, 2, 3, 0]]).as_euler('xyz')[:, 2])
        current = np.array([r['goal_current'] for r in rows])
        # Goal row: q, qd, gravity(3), angular velocity(3), linear velocity(3), z.
        nj = (current.shape[1]-10)//2
        entry_error = reference[0]-xy[0]
        displacement = xy[-1]-xy[0]
        entry_norm = np.linalg.norm(entry_error)
        settled = t >= t[0]+2.
        settled_xy = xy[settled]
        segments.append(dict(mode=rows[0]['mode'], start_s=float(t[0]),
            duration_s=float(t[-1]-t[0]), samples=len(rows),
            net_xy_m=displacement.tolist(), net_distance_m=float(np.linalg.norm(displacement)),
            max_radius_from_entry_m=float(np.linalg.norm(xy-xy[0], axis=1).max()),
            yaw_change_deg=float(np.degrees(yaw[-1]-yaw[0])),
            reference_net_xy_m=(reference[-1]-reference[0]).tolist(),
            reference_max_radius_m=float(np.linalg.norm(reference-reference[0], axis=1).max()),
            reference_entry_error_xy_m=entry_error.tolist(),
            reference_final_error_xy_m=(reference[-1]-xy[-1]).tolist(),
            displacement_toward_entry_reference_m=(float(displacement@entry_error/entry_norm)
                                                  if entry_norm > 1e-8 else None),
            settled_max_radius_m=(float(np.linalg.norm(settled_xy-settled_xy[0], axis=1).max())
                                   if len(settled_xy) else None),
            max_reference_horizontal_speed_m_s=float(np.linalg.norm(current[:, 2*nj+6:2*nj+8], axis=1).max()),
            root_height_min_m=min(r['root'][2] for r in rows)))
    return dict(scope='Single official physics simulation with simulator odometry; not hardware accuracy validation',
                samples=len(frames), segments=segments)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('Require a new output path')
    frames = [json.loads(line) for line in (a.policy_dir/'frames.jsonl').open()]
    if not frames:
        raise ValueError('No policy frames')
    report = summarize(frames)
    report['policy_result'] = json.loads((a.policy_dir/'result.json').read_text())
    a.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
