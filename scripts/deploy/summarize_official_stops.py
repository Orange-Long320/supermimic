"""Summarize one uninterrupted official-simulator trace and foot geometry."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial', type=Path, required=True)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists(): raise ValueError('output exists')
    rows = [json.loads(line) for line in (a.trial/'frames.jsonl').open()]
    result = json.loads((a.trial/'result.json').read_text())
    if len(rows) < 2: raise ValueError('not enough samples')
    t = np.array([r['elapsed'] for r in rows]); xyz = np.array([r['root'] for r in rows]);
    quats = np.array([r['q'] for r in rows]); joints = np.array([r['joint_pos'] for r in rows])
    if np.any(np.diff(t) <= 0): raise ValueError('non-monotonic timestamps')
    yaw = np.unwrap(Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_euler('xyz')[:, 2])
    spec = json.loads(a.profile.read_text()); names = spec['joint_names']
    model = mujoco.MjModel.from_xml_path(str(a.scene)); data = mujoco.MjData(model)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    if any(i < 0 for i in ids): raise ValueError('joint missing from official model')
    qadr = [model.jnt_qposadr[i] for i in ids]
    foot_geom = []
    for side in ('left', 'right'):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, side+'_ankle_roll_link')
        geom = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE and model.geom_contype[i]]
        if not geom: raise ValueError('no foot collision spheres')
        foot_geom.append(geom)
    gap = np.empty((len(rows), 2))
    for i in range(len(rows)):
        data.qpos[:7] = np.r_[xyz[i], quats[i]]; data.qpos[qadr] = joints[i]
        mujoco.mj_kinematics(model, data)
        for j, ids in enumerate(foot_geom):
            gap[i,j] = np.min(data.geom_xpos[ids,2]-model.geom_size[ids,0])
    def nearest(x): return int(np.argmin(np.abs(t-x)))
    modes = np.array([r['mode'] for r in rows]); changes = np.flatnonzero((modes[:-1]=='jog') & (modes[1:]=='stand'))+1
    events = [e for e in result['phase_events'] if e.get('from') == 'jog' and e.get('to') == 'stand']
    stops=[]
    for index, event in zip(changes, events):
        start = t[index]
        item = {'time': float(start), 'next_frame_at_selection': event['next_frame_at_selection']}
        for seconds in (1, 2, 4, 5):
            if start+seconds <= t[-1]:
                end=nearest(start+seconds)
                item[f'displacement_{seconds}s_m'] = float(np.linalg.norm(xyz[end,:2]-xyz[index,:2]))
                item[f'yaw_{seconds}s_deg'] = float(np.degrees(yaw[end]-yaw[index]))
        end=nearest(min(start+5,t[-1]))+1
        item['min_root_height_m']=float(xyz[index:end,2].min())
        item['min_foot_gap_m']=gap[index:end].min(axis=0).tolist()
        stops.append(item)
    if len(events) != len(changes): raise ValueError('release event count mismatch')
    stand = None
    if len(changes):
        begin = nearest(t[changes[-1]]+5); duration = t[-1]-t[begin]
        if duration > 0:
            pos = xyz[begin:,:2]; dist = np.linalg.norm(pos-pos[0], axis=1)
            stand={'settled_seconds': float(duration), 'net_displacement_m': float(dist[-1]),
                   'max_displacement_m': float(dist.max()), 'yaw_change_deg': float(np.degrees(yaw[-1]-yaw[begin])),
                   'minimum_root_height_m': float(xyz[begin:,2].min()), 'min_foot_gap_m': gap[begin:].min(axis=0).tolist(),
                   'both_feet_near_ground_fraction': float(np.mean(np.all(gap[begin:] < .005, axis=1))),
                   'near_ground_rule': 'minimum official sphere gap below 5 mm, not force contact'}
    dt=np.diff(t)
    summary={'failure':result['failure'], 'resets_after_start':result['resets_after_start'], 'seconds':float(t[-1]-t[0]),
             'frames':len(rows), 'jog_brake_fade':result['jog_brake_fade'], 'stops':stops, 'stand':stand,
             'max_future_overlap_error':result['max_future_overlap_error'], 'dt_p99_s':float(np.percentile(dt,99)),
             'dt_max_s':float(dt.max()), 'dt_over_40ms':int(np.sum(dt>.04)),
             'scope':'one no-reset rollout; foot gaps are kinematics, not measured contact forces'}
    a.out.write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))


if __name__ == '__main__': main()
