"""Inspect candidate walking-loop seams without changing the reference or simulator.

Run on the simulation server; results are screening evidence, not physical validation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('reference', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--first', type=int, default=60)
    p.add_argument('--last', type=int, default=203, help='Last allowed seam frame, inclusive')
    p.add_argument('--min-period', type=int, default=25)
    p.add_argument('--max-period', type=int, default=100)
    p.add_argument('--min-speed', type=float, default=0.2)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.min_period < 1 or args.max_period < args.min_period or args.first < 0 or args.last < args.first or args.min_speed < 0:
        p.error('Invalid search bounds')
    with np.load(args.reference, allow_pickle=False) as d:
        q, v = d['joint_pos'], d['joint_vel']
        pos, vel = d['body_pos_w'], d['body_lin_vel_w']
        names = list(d['body_names'])
        root = names.index('pelvis')
        feet = [names.index('left_ankle_roll_link'), names.index('right_ankle_roll_link')]
        fps = float(np.asarray(d['fps']).reshape(-1)[0])
        if args.last >= len(q) or not all(np.isfinite(x).all() for x in (q, v, pos, vel)):
            raise ValueError('Bounds exceed reference or reference contains non-finite values')
        speed = np.linalg.norm(vel[:, root, :2], axis=-1)
        seams = []
        for a in range(args.first, args.last + 1):
            for b in range(a + args.min_period, min(a + args.max_period, args.last) + 1):
                if min(speed[a], speed[b]) < args.min_speed:
                    continue
                dq = float(np.sqrt(np.mean((q[a] - q[b]) ** 2)))
                dv = float(np.sqrt(np.mean((v[a] - v[b]) ** 2)))
                foot = float(np.max(np.abs(pos[a, feet, 2] - pos[b, feet, 2])))
                ds = float(abs(speed[a] - speed[b]))
                seams.append(dict(start=a, end=b, joint_pos_rms_rad=dq, joint_vel_rms_rad_s=dv,
                                  ankle_height_max_diff_m=foot, horizontal_speed_diff_m_s=ds,
                                  screening_score=dq + .05 * dv + foot + .1 * ds))
        delta = np.sqrt(np.mean(np.diff(q, axis=0) ** 2, axis=1))
        result = dict(reference=str(args.reference), sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                      fps=fps, frames=len(q), search=dict(first=args.first, last=args.last,
                      min_period=args.min_period, max_period=args.max_period, min_speed=args.min_speed),
                      initial_horizontal_speed_m_s=float(speed[0]), final_horizontal_speed_m_s=float(speed[-1]),
                      final_joint_vel_rms_rad_s=float(np.sqrt(np.mean(v[-1] ** 2))),
                      adjacent_joint_pos_rms_median_rad=float(np.median(delta)),
                      adjacent_joint_pos_rms_p95_rad=float(np.percentile(delta, 95)),
                      candidate_count=len(seams), candidates=sorted(seams, key=lambda x: x['screening_score'])[:10],
                      note='Heuristic ranking only; contact, root orientation, physical continuity and stability are not validated.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
