"""Prepare a reproducible, bounded native-control diagnostic from a reference.

No training approval is granted. Entry conditioning is recorded and applies
only before the first physics step; all later states come from simulation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--tracker',type=Path,required=True)
    p.add_argument('--controller',type=Path,required=True)
    p.add_argument('--joint-predictor',type=Path,help='Private predictor snapshot; never overwrite shared project modules')
    p.add_argument('--seconds',type=float,default=2.)
    p.add_argument('--wall-seconds',type=int,default=180)
    p.add_argument('--joint-task-weight',type=float,default=.1)
    p.add_argument('--foot-task-weight',type=float,default=.1)
    p.add_argument('--source-start-frame',type=int,help='Required only when the source artifact lacks origin metadata')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists() or min(a.seconds,a.wall_seconds,a.joint_task_weight,a.foot_task_weight)<=0:p.error('New output and positive bounds required')
    from supermimic.robots.paths import ROOT,select_robot
    select_robot('x2_ultra',check_assets=True)
    import mujoco
    import numpy as np
    from supermimic.data.physics_motion import differentiate_qpos
    m=mujoco.MjModel.from_binary_path(str(a.model));d=mujoco.MjData(m)
    if (m.nq,m.nv,m.nu)!=(38,37,31):raise ValueError('Expected native actuated X2')
    with np.load(a.reference,allow_pickle=False) as z:
        q=z['qpos'].copy();fps=float(np.asarray(z['fps']).reshape(-1)[0]);v=z['qvel'].copy() if 'qvel' in z else differentiate_qpos(m,q,fps)
        source_start=int(z['source_start_frame']) if 'source_start_frame' in z else None
    sidecar=a.reference.parent/'result.json'
    if source_start is None and sidecar.is_file():source_start=json.loads(sidecar.read_text()).get('source_start_frame')
    if a.source_start_frame is not None:
        if source_start is not None and source_start!=a.source_start_frame:raise ValueError('Source origin metadata conflicts with requested start')
        source_start=a.source_start_frame
    if source_start is None or source_start<0:raise ValueError('Record the source window origin explicitly; do not assume a complete climb')
    if q.shape[1]!=38 or v.shape!=(len(q),37) or len(q)<5 or not np.isfinite(q).all() or not np.isfinite(v).all():raise ValueError('Invalid reference')
    margin=float(np.minimum(q[0,7:]-m.jnt_range[1:,0]-.02,m.jnt_range[1:,1]-.02-q[0,7:]).min())
    if margin<=0:raise ValueError('Initial reference violates existing controller predictive interior; repair the reference, not the safety margin')
    d.qpos[:]=q[0];mujoco.mj_forward(m,d);root=int(m.jnt_bodyid[0]);feet={s:m.body('robot/'+s+'_ankle_roll_link').id for s in ['left','right']}
    obstacles=[g for g in range(m.ngeom) if int(m.body_rootid[int(m.geom_bodyid[g])])!=root and (m.geom_contype[g] or m.geom_conaffinity[g])]
    gaps={};support={}
    for side,body in feet.items():
        values=[]
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g])!=body or not(m.geom_contype[g] or m.geom_conaffinity[g]):continue
            for o in obstacles:
                w=np.zeros(6);gap=float(mujoco.mj_geomDistance(m,d,g,o,.1,w));values.append((gap,o,w.copy()))
        gap,o,w=min(values,key=lambda x:x[0]);vector=w[:3]-w[3:];length=np.linalg.norm(vector)
        if length<1e-12 or (np.sign(gap)*vector/length)[2]<.99:raise ValueError('Initial nearest foot surface is not horizontal')
        gaps[side]=gap;support[side]={'geom':m.geom(o).name,'height_m':float(w[5])}
    offset=-.0005-max(gaps.values());entry=q[0].copy();entry[2]+=offset;d.qpos[:]=entry;mujoco.mj_forward(m,d)
    contacts={side:[float(c.dist) for c in d.contact if body in [int(m.geom_bodyid[int(g)]) for g in c.geom] and c.dist<0] for side,body in feet.items()}
    if not all(contacts.values()) or min(x for values in contacts.values() for x in values)<-.005:raise ValueError('Initial actual contacts exceed existing native foot limits')
    heights=[0.]
    for g in obstacles:
        if int(m.geom_type[g])==int(mujoco.mjtGeom.mjGEOM_BOX):
            if abs(d.geom_xmat[g].reshape(3,3)[2,:2]).max()>1e-8:raise ValueError('Expected horizontal source terrain')
            heights.append(float(d.geom_xpos[g,2]+m.geom_size[g,2]))
    a.out.mkdir(parents=True);(a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    (a.out/'tracker.py').write_bytes(a.tracker.read_bytes());(a.out/'native-inverse-snapshot.py').write_bytes(a.controller.read_bytes())
    if a.joint_predictor:(a.out/'joint-prediction-snapshot.py').write_bytes(a.joint_predictor.read_bytes())
    np.savez_compressed(a.out/'reference-qpos.npz',qpos=q,qvel=v,fps=np.array([fps]))
    np.savez_compressed(a.out/'controller-posture-seed.npz',qpos=entry)
    info={'minimum_predictor_interior_margin_rad':margin,'entry_root_z_offset_m':offset,'initial_foot_gaps_m':gaps,
        'initial_contacts_m':contacts,'support_witnesses':support,'source_start_frame':source_start,'local_start_frame':0,'entry_velocity':'zero',
        'simulation_seconds':a.seconds,'wall_seconds':a.wall_seconds,'original_reference':str(a.reference),
        'original_reference_sha256':hashlib.sha256(a.reference.read_bytes()).hexdigest(),'training_approved':False}
    (a.out/'entry-contact.json').write_text(json.dumps(info,indent=2)+'\n')
    command=[sys.executable,str(a.out/'tracker.py'),'--model',str(a.model),'--reference',str(a.out/'reference-qpos.npz'),
        '--equilibrium',str(a.out/'controller-posture-seed.npz'),'--start-frame','0','--seconds',str(a.seconds),
        '--entry-velocity-mode','zero','--entry-root-z-offset',str(offset),'--joint-task-weight',str(a.joint_task_weight),
        '--foot-task-weight',str(a.foot_task_weight),
        '--recover-planned-contacts','--support-surface-heights',*[str(h) for h in sorted(set(heights))],
        '--native-controller-source',str(a.out/'native-inverse-snapshot.py'),'--out',str(a.out/'trace')]
    if a.joint_predictor:command+=['--joint-prediction-source',str(a.out/'joint-prediction-snapshot.py')]
    (a.out/'command.json').write_text(json.dumps(command,indent=2)+'\n')
    (a.out/'runner.py').write_text('''import json,subprocess,os
from pathlib import Path
run=Path(__file__).resolve().parent
(run/'pid').write_text(str(os.getpid())+'\\n')
command=json.loads((run/'command.json').read_text())
try:code=subprocess.run(command,timeout=json.loads((run/'entry-contact.json').read_text())['wall_seconds']).returncode
except subprocess.TimeoutExpired:code=124
(run/'exit-code').write_text(str(code)+'\\n')
raise SystemExit(code)
''')
    (a.out/'launch.sh').write_text('#!/bin/bash\nset -eu\nexport CUDA_VISIBLE_DEVICES=\'\' SM_ROBOT=x2_ultra OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2\nunset PYTHONPATH PYTHONHOME\ncd '+shlex.quote(str(ROOT))+'\nexec '+shlex.quote(sys.executable)+' '+shlex.quote(str(a.out/'runner.py'))+' > '+shlex.quote(str(a.out/'run.log'))+' 2>&1\n')
    (a.out/'resources.txt').write_text(f'CPU only; two threads; simulation {a.seconds}s; wall limit {a.wall_seconds}s; no GPU\n')
    print(json.dumps(info,indent=2))


if __name__=='__main__':main()
