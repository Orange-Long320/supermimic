"""Create a separate exploratory jog-stop reference with grounded brake frames."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--stop-manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new output required')
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.data.physics_motion import export_qpos_motion
    m=load_g1_model(); d=mujoco.MjData(m); file=a.dataset/'jog_stop.npz'
    with np.load(file,allow_pickle=False) as z:
        root=list(z['body_names']).index('pelvis');fps=float(z['fps'][0]);q=np.c_[z['body_pos_w'][:,root],z['body_quat_w'][:,root],z['joint_pos']].astype(float)
    meta=json.loads(a.stop_manifest.read_text());start=meta['brake_start_frame'];end=meta['standing_frame']
    feet=[]
    for side in ('left','right'):
        body=m.body(side+'_ankle_roll_link').id;meshes=[]
        for g in range(m.ngeom):
            if m.geom_bodyid[g]!=body or not(m.geom_contype[g] or m.geom_conaffinity[g]) or m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:continue
            k=m.geom_dataid[g];meshes.append((g,m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k]+m.mesh_vertnum[k]].copy()))
        if not meshes:raise ValueError('missing foot geometry')
        feet.append(meshes)
    def foot_heights(row):
        d.qpos[:]=row;mujoco.mj_kinematics(m,d)
        return [min(float((v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])[:,2].min()) for g,v in gg) for gg in feet]
    h=np.array([foot_heights(row) for row in q[start:end+1]])
    # Preserve the reviewed endpoint when the two model geometries disagree
    # slightly; never demand a tighter bound than its existing foot clearance.
    clearance_floor=min(-.001,float(h[-1].min()))
    need=np.maximum(0.,clearance_floor-h.min(axis=1))
    lift=np.maximum(need,gaussian_filter1d(maximum_filter1d(need,size=5),sigma=1.0))
    if need[-1]>1e-9:raise ValueError('standing endpoint requires correction')
    lift[-1]=0.
    correction=np.zeros(len(q));correction[start:end+1]=lift
    # Restore the existing entry gradually, without modifying the published end pose.
    for i in range(max(0,start-8),start):
        t=(i-(start-8))/8.;correction[i]=lift[0]*(3*t*t-2*t*t*t)
    modified=q.copy();modified[:,2]+=correction
    dz=np.diff(modified[:,2]);dd=np.diff(dz)
    if np.max(np.abs(dd[max(0,start-9):end+1]))>.005:raise ValueError('root-height acceleration screen failed')
    a.out.mkdir(parents=True)
    for path in a.dataset.iterdir():
        if path.name not in ('manifest.json','jog_stop.npz'):shutil.copy2(path,a.out/path.name)
    export_qpos_motion(m,modified,a.out/'jog_stop.npz',fps,source=str(file))
    manifest=json.loads((a.dataset/'manifest.json').read_text())
    for clip in manifest['clips']:
        if clip['name']=='jog_stop':clip['sha256']=digest(a.out/'jog_stop.npz');clip['ground_repair_source_sha256']=digest(file)
    manifest['scope']='Exploratory ground-height adjustment; reference dynamics and policy acceptance not validated'
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    after=np.array([foot_heights(row) for row in modified[start:end+1]])
    report={'source_sha256':digest(file),'output_sha256':digest(a.out/'jog_stop.npz'),'brake_start':start,'standing_frame':end,
            'min_before_m':h.min(axis=0).tolist(),'min_after_m':after.min(axis=0).tolist(),
            'clearance_floor_m':clearance_floor,
            'max_root_lift_m':float(correction.max()),'max_root_step_accel_m':float(np.max(abs(dd[max(0,start-9):end+1]))),
            'standing_qpos_unchanged':bool(np.array_equal(q[end:],modified[end:])),
            'scope':'reference geometry only; no contact force or policy validation'}
    (a.out/'ground-repair.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
