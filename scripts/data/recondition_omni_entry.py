"""Blend a calibrated low-surface entry into an unchanged climbing source.

The original data is preserved. This only prepares a diagnostic reference;
independent geometry, dynamics and actual complete-climb checks remain required.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--source-support-report',type=Path,required=True)
    p.add_argument('--source-start-frame',type=int,required=True)
    p.add_argument('--blend-frames',type=int,default=20)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists() or a.source_start_frame<0 or a.blend_frames<2:p.error('New output and a valid supported entry window required')
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation
    with np.load(a.reference,allow_pickle=False) as z:
        q=z['qpos'].copy();fps=float(np.asarray(z['fps']).reshape(-1)[0])
    with np.load(a.calibration,allow_pickle=False) as z:entry=z['qpos'].copy()
    if q.shape[1:]!=(38,) or entry.shape!=(38,) or a.source_start_frame+a.blend_frames>=len(q):raise ValueError('Invalid full-reference window')
    support=json.loads(a.source_support_report.read_text())
    if not support['enabled']:raise ValueError('Source-confirmed support windows required')
    for side in ['left','right']:
        if not any(start<=a.source_start_frame and end>=a.source_start_frame+a.blend_frames for start,end,height in support['windows'][side]):
            raise ValueError('Entry correction must finish before either source foot leaves the initial support')
    original=q[a.source_start_frame:].copy();q=original.copy()
    translation=entry[:3]-q[0,:3];joint_delta=entry[7:]-q[0,7:]
    correction=(Rotation.from_quat(entry[[4,5,6,3]])*Rotation.from_quat(q[0,[4,5,6,3]]).inv()).as_rotvec()
    for i in range(a.blend_frames):
        t=i/a.blend_frames;weight=1-(10*t**3-15*t**4+6*t**5)
        q[i,:3]+=weight*translation;q[i,7:]+=weight*joint_delta
        q[i,3:7]=(Rotation.from_rotvec(weight*correction)*Rotation.from_quat(q[i,[4,5,6,3]])).as_quat()[[3,0,1,2]]
    np.testing.assert_allclose(q[0],entry,atol=1e-12)
    np.testing.assert_array_equal(q[a.blend_frames:],original[a.blend_frames:])
    m=mujoco.MjModel.from_binary_path(str(a.model))
    hard=float(np.maximum(np.maximum(m.jnt_range[1:,0]-q[:,7:],q[:,7:]-m.jnt_range[1:,1]),0.).max())
    middle=m.jnt_range[1:].mean(axis=1);half=np.diff(m.jnt_range[1:],axis=1).reshape(-1)*.45
    soft=float(np.maximum(np.maximum(middle-half-q[:,7:],q[:,7:]-(middle+half)),0.).max())
    if max(hard,soft)>1e-8 or not np.isfinite(q).all():raise ValueError('Reconditioning violates existing hard or 0.9 reset limits')
    a.out.mkdir(parents=True)
    np.savez_compressed(a.out/'reference.npz',qpos=q,fps=np.array([fps]),source_start_frame=np.array(a.source_start_frame))
    report={'source_start_frame':a.source_start_frame,'frames':len(q),'fps':fps,'blend_frames':a.blend_frames,
        'unchanged_from_source_frame':a.source_start_frame+a.blend_frames,
        'entry_translation_m':translation.tolist(),'entry_world_rotation_delta_rad':correction.tolist(),
        'maximum_entry_joint_adjustment_rad':float(abs(joint_delta).max()),'hard_violation_rad':hard,'reset_soft_violation_rad':soft,
        'source_sha256':hashlib.sha256(a.reference.read_bytes()).hexdigest(),
        'calibration_sha256':hashlib.sha256(a.calibration.read_bytes()).hexdigest(),
        'configuration':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        'complete_climb_verified':False,'training_approved':False,'scope':__doc__}
    (a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n');(a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
