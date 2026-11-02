"""Verify a fixed world-yaw error rotates outputs without changing support decisions."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('output exists')
    r=a.results; base=np.load(r/'baseline.npz');yaw=np.load(r/'yaw_1deg.npz')
    first=json.loads((r/'baseline.json').read_text());second=json.loads((r/'yaw_1deg.json').read_text())
    if not np.array_equal(base['t'],yaw['t']):raise ValueError('timestamps differ')
    rot=Rotation.from_euler('z',1,degrees=True)
    result={'samples':len(base['t']),'timestamps_equal':True,'counters_equal':first['counters']==second['counters'],
            'max_position_rotation_residual_m':float(np.max(np.linalg.norm(rot.apply(base['p'])-yaw['p'],axis=1))),
            'max_velocity_rotation_residual_m_s':float(np.max(np.linalg.norm(rot.apply(base['v'])-yaw['v'],axis=1)))}
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if not result['counters_equal'] or result['max_position_rotation_residual_m']>1e-9 or result['max_velocity_rotation_residual_m_s']>1e-9:
        raise ValueError('Yaw equivariance verification failed')


if __name__=='__main__':main()
