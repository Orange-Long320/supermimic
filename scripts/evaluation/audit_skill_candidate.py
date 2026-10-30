"""Measure collision clearance, kinematic limits and continuity by phase."""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    a=p.parse_args()
    from supermimic.data.motion_convert import load_g1_model
    model=load_g1_model();data=mujoco.MjData(model)
    manifest=json.loads((a.candidate/'manifest.json').read_text())
    with np.load(a.candidate/'candidate.npz') as d:
        q=d['joint_pos'];v=d['joint_vel'];names=list(d['body_names']);root=names.index(model.body(int(model.jnt_bodyid[0])).name)
        pos=d['body_pos_w'][:,root];quat=d['body_quat_w'][:,root];lin=d['body_lin_vel_w'][:,root]
        if len(q)!=len(manifest['phases']):raise ValueError('Frame count mismatch')
        bounds=model.jnt_range[1:];excess=np.maximum(np.maximum(bounds[:,0]-q,q-bounds[:,1]),0)
        clearance=[]
        for i in range(len(q)):
            data.qpos[:]=np.r_[pos[i],quat[i],q[i]];mujoco.mj_kinematics(model,data)
            minima={'left':float('inf'),'right':float('inf'),'other':float('inf')}
            for g in range(model.ngeom):
                if model.geom_bodyid[g]==0 or not (model.geom_contype[g] or model.geom_conaffinity[g]):continue
                name=model.geom(g).name or '';mat=data.geom_xmat[g].reshape(3,3);kind=model.geom_type[g];size=model.geom_size[g]
                if kind==mujoco.mjtGeom.mjGEOM_MESH:
                    mesh=model.geom_dataid[g];idx=model.mesh_vertadr[mesh];vertices=model.mesh_vert[idx:idx+model.mesh_vertnum[mesh]]
                    z=float((vertices@mat[2]+data.geom_xpos[g,2]).min())
                elif kind==mujoco.mjtGeom.mjGEOM_SPHERE:z=float(data.geom_xpos[g,2]-size[0])
                elif kind==mujoco.mjtGeom.mjGEOM_CAPSULE:z=float(data.geom_xpos[g,2]-abs(mat[2,2])*size[1]-size[0])
                elif kind==mujoco.mjtGeom.mjGEOM_BOX:z=float(data.geom_xpos[g,2]-abs(mat[2])@size)
                else:continue
                body=model.body(int(model.geom_bodyid[g])).name or ''
                side='left' if 'left_ankle' in body else 'right' if 'right_ankle' in body else 'other'
                minima[side]=min(minima[side],z)
            clearance.append(minima)
        phases=np.array(manifest['phases']);result={}
        for phase in dict.fromkeys(phases):
            ids=np.flatnonzero(phases==phase);arr=np.array([[clearance[i][k] for k in ('left','right','other')] for i in ids]);
            result[phase]={'frames':len(ids),'min_left_right_other_m':arr.min(0).tolist(),
                'both_feet_above_2cm_fraction':float(((arr[:,:2]>.02).all(1)).mean()),
                'max_hard_limit_excess_rad':float(excess[ids].max()),
                'max_joint_speed_rad_s':float(abs(v[ids]).max()),
                'max_root_horizontal_speed_m_s':float(np.linalg.norm(lin[ids,:2],axis=1).max())}
        report={'candidate_sha256':manifest['candidate_sha256'],'per_phase':result,
                'scope':'Kinematic geometry only; near-ground foot geometry is not measured contact force or support; no dynamic feasibility approval.'}
    out=a.candidate/'geometry-audit.json'
    with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
