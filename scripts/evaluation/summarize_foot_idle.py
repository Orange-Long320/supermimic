"""Compare equal-time holds, and expose whether foot alignment actually ran."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def window(rows, start, duration):
    chosen = [r for r in rows if start <= r['elapsed'] < start+duration]
    if len(chosen)<2 or chosen[-1]['elapsed'] < start+duration-.06:
        return dict(complete=False,frames=len(chosen))
    pos = np.array([r['root'][:2] for r in chosen])
    qd = np.array([r['joint_vel'] for r in chosen])
    item = dict(complete=True,start_s=start,duration_s=duration,
                root_excursion_m=float(np.linalg.norm(pos-pos[0],axis=1).max()),
                root_net_displacement_m=float(np.linalg.norm(pos[-1]-pos[0])),
                joint_speed_peak_rms_rad_s=float(np.sqrt(np.mean(qd**2,axis=1)).max()))
    if all('reference_root_position' in r for r in chosen):
        ref = np.array([r['reference_root_position'][:2] for r in chosen])
        item.update(reference_range_xy_m=np.ptp(ref,axis=0).tolist(),
                    root_target_error_mean_m=float(np.linalg.norm(ref-pos,axis=1).mean()))
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    rows, reports, result = {}, {}, {}
    for name in ('context-idle','context-foot-aligned'):
        folder = args.run/name/'policy'
        rows[name] = [json.loads(line) for line in (folder/'frames.jsonl').read_text().splitlines()]
        reports[name] = json.loads((folder/'result.json').read_text())
        r = rows[name]
        releases = [r[i]['elapsed'] for i in range(1,len(r)) if r[i-1]['mode']=='walk' and r[i]['mode']=='stand']
        item = dict(failure=reports[name]['failure'],control_seconds=reports[name]['control_seconds'],
                    frames=len(r),statuses=dict(Counter(x.get('foot_idle_status','original') for x in r)),
                    events=reports[name].get('foot_idle_events',[]),stops=[])
        for j,release in enumerate(releases):
            nxt = next((x['elapsed'] for x in r if x['elapsed']>release and x['mode']!='stand'),r[-1]['elapsed']+.02)
            holds = [x for x in r if release <= x['elapsed'] < nxt and x.get('foot_idle_status')=='hold']
            item['stops'].append(dict(release_s=release,next_motion_s=nxt,
                aligned_hold_start_s=holds[0]['elapsed'] if holds else None,
                fixed_window_10s=window(r,release+5.,10.) if nxt>=release+15. else dict(complete=False),
                fixed_window_40s=window(r,release+5.,40.) if nxt>=release+45. else dict(complete=False)))
        result[name] = item
    holds = [r for r in rows['context-foot-aligned'] if r['elapsed']>=27.8 and r.get('foot_idle_status')=='hold']
    if holds:
        start = holds[0]['elapsed']+.2
        for name in rows:
            result[name]['same_time_after_alignment_40s'] = window(rows[name],start,40.)
    output = dict(trials=result,scope='One frozen-policy official simulation per variant. Fixed release+5 windows may contain an unfinished alignment; separate equal-time post-alignment metric does not replace them. No PPO or hardware.')
    with args.out.open('x') as f:
        json.dump(output,f,indent=2);f.write('\n')
    for name,item in result.items():
        print(json.dumps(dict(trial=name,failure=item['failure'],control_seconds=item['control_seconds'],statuses=item['statuses'],stops=item['stops'],post_alignment=item.get('same_time_after_alignment_40s'))))


if __name__ == '__main__':
    main()
