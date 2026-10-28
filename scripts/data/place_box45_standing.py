"""Translate a checked standing endpoint onto a box without changing joints."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--box',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('output exists')
    import mujoco
    import numpy as np
    from supermimic.envs.box_scene import FixedBox
    m=mujoco.MjModel.from_binary_path(str(a.model));d=mujoco.MjData(m)
    z=np.load(a.source);q=z['qpos'].copy()
    if q.shape!=(m.nq,):raise ValueError('Standing endpoint/model dimensions differ')
    box=FixedBox(**json.loads(a.box.read_text())['box'])
    d.qpos[:]=q;mujoco.mj_kinematics(m,d);points=[]
    for g in range(m.ngeom):
        if not m.geom(g).name.rsplit('/',1)[-1].startswith('foot_'):continue
        mesh=m.geom_dataid[g]
        if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX:
            local=np.array([(x,y,z) for x in (-1,1) for y in (-1,1) for z in (-1,1)])*m.geom_size[g]
        else:
            adr,count=m.mesh_vertadr[mesh],m.mesh_vertnum[mesh];local=m.mesh_vert[adr:adr+count]
        points.extend(local@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])
    points=np.asarray(points);center=(points[:,:2].min(0)+points[:,:2].max(0))/2
    delta=np.r_[np.asarray(box.center[:2])-center,box.height]
    q[:3]+=delta
    transformed=points+delta
    if not all(box.top_contains(float(x),float(y),margin=.02) for x,y in transformed[:,:2]):raise ValueError('Standing foot hull outside box')
    a.out.mkdir(parents=True,exist_ok=False)
    np.savez(a.out/'equilibrium.npz',qpos=q,qvel=np.zeros(m.nv))
    poses=np.repeat(q[None],151,axis=0)
    np.savez_compressed(a.out/'reference-qpos.npz',qpos=poses,qvel=np.zeros((151,m.nv)),fps=np.array([50.]))
    np.savez_compressed(a.out/'aligned-diagnostic.npz',qpos=poses,fps=50.,box_center=box.center,box_half_size=box.half_size)
    report={'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),'translation_m':delta.tolist(),'joint_angles_unchanged':True,'box':json.loads(a.box.read_text())['box'],'training_approved':False}
    (a.out/'placement.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.out/'as-run.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(report,indent=2))


if __name__=='__main__':main()
