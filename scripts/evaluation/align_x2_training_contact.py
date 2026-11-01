"""Build a simulation-only level-floor alignment of a recorded X2 pose.

Aligns the lowest two training-scene foot mesh points by changing world roll
and height. Does not infer physical calibration or alter hardware state.
"""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('scene','snapshot','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('New output required')
    s=json.loads(a.snapshot.read_text());m=mujoco.MjModel.from_binary_path(str(a.scene));d=mujoco.MjData(m)
    if len(s['joint_names'])!=31 or len(s['joint_position'])!=31:raise ValueError('Joint sample')
    for name,q in zip(s['joint_names'],s['joint_position']):
        j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'robot/'+name)
        if j<0:raise ValueError('Joint missing')
        d.qpos[m.jnt_qposadr[j]]=q
    feet=[]
    for side in ('left','right'):
        b=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot/'+side+'_ankle_roll_link')
        ids=[i for i in range(m.ngeom) if m.geom_bodyid[i]==b and m.geom_contype[i] and m.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH]
        if len(ids)!=1:raise ValueError('Expected one training foot mesh')
        feet.append(ids[0])
    base=Rotation.from_quat(np.asarray(s['quat_wxyz'])[[1,2,3,0]])
    def heights(roll):
        rot=Rotation.from_rotvec([roll,0,0])*base;quat=rot.as_quat();d.qpos[:7]=[*s['position'],*quat[[3,0,1,2]]];mujoco.mj_forward(m,d)
        result=[]
        for g in feet:
            mesh=int(m.geom_dataid[g]);start=int(m.mesh_vertadr[mesh]);count=int(m.mesh_vertnum[mesh])
            vert=np.asarray(m.mesh_vert[start:start+count],dtype=float)
            world=vert@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g];result.append(float(world[:,2].min()))
        return result
    before=heights(0.);roll=brentq(lambda t:heights(t)[0]-heights(t)[1],-.08,.08)
    match=heights(roll);dz=-float(np.mean(match))-.0002;quat=(Rotation.from_rotvec([roll,0,0])*base).as_quat()
    s['quat_xyzw']=quat.tolist();s['quat_wxyz']=quat[[3,0,1,2]].tolist();s['position'][2]+=dz
    s['scope']=__doc__;s['contact_alignment']={'world_roll_rad':roll,'height_shift_m':dz,
                                                  'original_mesh_clearance_m':before}
    a.out.write_text(json.dumps(s,indent=2,allow_nan=False)+'\n');print(json.dumps(s['contact_alignment'],indent=2))


if __name__=='__main__':main()
