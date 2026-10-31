"""Build foot-aligned plans from recorded official simulator poses (CPU only)."""
import argparse
import json
from pathlib import Path

import numpy as np

from supermimic.control.foot_aligned_idle import FootAlignedIdle
from supermimic.control.idle_velocity_feedback import RootVelocityEstimate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frames',type=Path,required=True)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    rows = [json.loads(x) for x in args.frames.read_text().splitlines()]
    planner = FootAlignedIdle(args.scene,json.loads(args.profile.read_text()))
    velocity = RootVelocityEstimate()
    previous, release, selected = 'stand', None, False
    reports = []
    for row in rows:
        v = velocity.update(row['elapsed'],row['root'])
        if row['mode'] == 'stand' and previous == 'walk':
            release, selected = row['elapsed'], False
        previous = row['mode']
        if release is None or selected or row['mode'] != 'stand' or row['elapsed'] < release+1.:
            continue
        context = planner.observe(row['root'],row['q'],row['joint_pos'],row['joint_vel'],v,row['body_angular_velocity'])
        if not planner.ready(context):
            continue
        selected = True
        item = dict(release_s=release,context_s=row['elapsed'],gap=context['gap'].tolist(),
                    root_speed=float(np.linalg.norm(v[:2])))
        try:
            plan = planner.build(row['goal_current'],context)
            item['report'] = plan.report
            np.savez(args.out/f'plan-{len(reports)}.npz',features=plan.features,
                     root_positions=plan.root_positions,yaw=plan.yaw)
        except Exception as exc:
            item['error'] = str(exc)
        reports.append(item)
        print(json.dumps(item),flush=True)
    (args.out/'report.json').write_text(json.dumps(reports,indent=2)+'\n')


if __name__ == '__main__':
    main()
