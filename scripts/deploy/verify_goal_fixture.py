"""Cross-platform future-reference comparison on synthetic inputs; no robot I/O."""
import argparse
import json
from pathlib import Path
import time

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--expected', type=Path)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('Output exists')
    import torch
    from supermimic.deploy.skill_goal import SkillGoal
    torch.set_num_threads(1)
    with np.load(a.dataset/'walk_stop.npz', allow_pickle=False) as z:
        standing = z['joint_pos'][-1].astype(float)
    modes = [('stand',15),('walk',60),('stand',45),('jog',60),('stand',45),
             ('backward',25),('stand',20),('turn_left_inplace',25),('stand',20),
             ('turn_right_inplace',25),('stand',20)]
    tracker = SkillGoal(a.dataset, a.profile, [0.,0.,.65], [1.,0.,0.,0.])
    outputs, elapsed = [], []
    for mode, count in modes:
        for _ in range(count):
            i = len(outputs)
            q = standing + .01*np.sin(i*.07 + np.arange(len(standing))*.13)
            pos = np.array([i*.0002, 0., .65]); start = time.perf_counter()
            goal = tracker.goal(pos, [1.,0.,0.,0.], q, mode)
            elapsed.append((time.perf_counter()-start)*1000)
            if goal.shape != (729,) or not np.isfinite(goal).all():
                raise ValueError('Invalid goal')
            outputs.append(goal.tolist()); tracker.advance()
    result = {'modes':modes, 'frames':len(outputs), 'outputs':outputs,
              'numpy_version':np.__version__, 'torch_version':torch.__version__,
              'scope':'synthetic fixed inputs, no policy loop or robot actuation'}
    if a.expected:
        expected = json.loads(a.expected.read_text())
        if expected['modes'] != result['modes']:
            raise ValueError('Fixture sequence differs')
        error = float(np.max(np.abs(np.asarray(expected['outputs'])-np.asarray(outputs))))
        order = sorted(elapsed)
        result.pop('outputs')
        result.update(max_abs_error=error, goal_ms_p50=order[len(order)//2],
                      goal_ms_p99=order[min(len(order)-1,int(.99*len(order)))],
                      goal_ms_max=max(order))
        a.out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
        if error > 1e-4:
            raise SystemExit('Future-reference mismatch')
    else:
        a.out.write_text(json.dumps(result)+'\n')
        print(json.dumps({'frames':len(outputs),'scope':result['scope']}))


if __name__ == '__main__':
    main()
