"""Audit snapshot/reference support geometry in the official model; no robot I/O."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial import ConvexHull, QhullError


def hull_margin(points, query):
    try:
        hull=ConvexHull(points)
    except QhullError:
        return dict(margin_m=None,reason='Degenerate support hull',points=points.tolist())
    eq=hull.equations
    return dict(margin_m=float(np.min(-(eq[:,:2]@query+eq[:,2])/np.linalg.norm(eq[:,:2],axis=1))),
                points=points.tolist(),vertices=points[hull.vertices].tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('scene','candidate','snapshot','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('New output required')
    c=json.loads(a.candidate.read_text());s=json.loads(a.snapshot.read_text());names=c['joint_names']
    if names!=s['names']:raise ValueError('Joint order mismatch')
    m=mujoco.MjModel.from_xml_path(str(a.scene));ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]
    if min(ids)<0 or m.nv!=37:raise ValueError('Expected X2 free-base model')
    qa=m.jnt_qposadr[ids]
    feet=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n+'_ankle_roll_link') for n in ('left','right')]
    floor=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'floor')
    gs=[g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in feet and (m.geom_contype[g] or m.geom_conaffinity[g])]
    if not gs or any(m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_SPHERE for g in gs):raise ValueError('Expected official foot spheres')
    results={}
    for kind in ('snapshot','reference'):
        d=mujoco.MjData(m);q=np.array(s['q'] if kind=='snapshot' else c['startup']['position']);quat=np.array(s['imu']['orientation_wxyz'] if kind=='snapshot' else c['startup']['root_quat_wxyz']);quat/=np.linalg.norm(quat)
        d.qpos[:7]=np.r_[[0.,0.,1.],quat];d.qpos[qa]=q;mujoco.mj_forward(m,d)
        bottoms=np.array([d.geom_xpos[g,2]-m.geom_size[g,0] for g in gs]);d.qpos[2]-=min(bottoms)+.0002;mujoco.mj_forward(m,d)
        bottoms=np.array([d.geom_xpos[g,2]-m.geom_size[g,0] for g in gs]);points=np.array([d.geom_xpos[g,:2] for g in gs]);eligible=bottoms<=min(bottoms)+.0005
        com=d.subtree_com[0].copy();columns=[]
        for g in np.array(gs)[eligible]:
            point=d.geom_xpos[g].copy();point[2]-=m.geom_size[g,0];jac=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jac,None,point,int(m.geom_bodyid[g]));columns.append(jac[2,:6])
        A=np.array(columns).T;sol=lsq_linear(A,d.qfrc_bias[:6],bounds=(0.,np.inf),tol=1e-12,max_iter=200)
        contacts=[]
        for j in range(d.ncon):
            ct=d.contact[j];g1,g2=map(int,ct.geom);force=np.zeros(6);mujoco.mj_contactForce(m,d,j,force)
            if floor in (g1,g2) and int(m.geom_bodyid[g2 if g1==floor else g1]) in feet:continue
            contacts.append(dict(bodies=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])) for g in (g1,g2)],distance_m=float(ct.dist),normal_n=float(force[0])))
        results[kind]=dict(root_model_assumption=d.qpos[:7].tolist(),com_world_m=com.tolist(),all_foot_sphere_projection=hull_margin(points,com[:2]),near_floor_sphere_support=hull_margin(points[eligible],com[:2]),foot_bottoms_m=bottoms.tolist(),near_floor_spheres=int(eligible.sum()),static_vertical_equilibrium_residual=float(np.max(np.abs(A@sol.x-d.qfrc_bias[:6]))),nonfoot_contacts=contacts)
    report=dict(simulation_only=True,calibration_validated=False,hardware_ready=False,source_sha256={k:hashlib.sha256(getattr(a,k).read_bytes()).hexdigest() for k in ('scene','candidate','snapshot')},results=results)
    a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
