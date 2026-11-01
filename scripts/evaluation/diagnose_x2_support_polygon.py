"""Compare projected foot collision spheres and visual meshes for an X2 pose.

The convex hull uses every collision sphere on both feet, including spheres
that may not actually touch the floor. The mesh hull similarly includes all
vertices, including non-contact surfaces. Neither proves the actual support
polygon, mesh equivalence to hardware, or physical equilibrium.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('scene', 'snapshot', 'out'): p.add_argument('--'+key, type=Path, required=True)
    a=p.parse_args()
    if a.out.exists(): raise ValueError('new output required')
    snapshot=json.loads(a.snapshot.read_text()); names=snapshot['joint_names']; q=snapshot['joint_position']
    if len(names)!=31 or len(set(names))!=31 or len(q)!=31: raise ValueError('joint mapping')
    m=mujoco.MjModel.from_xml_path(str(a.scene)); d=mujoco.MjData(m)
    d.qpos[:7]=[*snapshot['position'], *snapshot['quat_wxyz']]
    for name, value in zip(names,q):
        jid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name)
        if jid<0 or not np.isfinite(value): raise ValueError('invalid joint')
        d.qpos[m.jnt_qposadr[jid]]=value
    mujoco.mj_forward(m,d)
    root=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'pelvis')
    if root<0: raise ValueError('missing pelvis')
    com=d.subtree_com[root].copy(); per_side={}; points=[]; mesh_points=[]
    angles=np.linspace(0,2*np.pi,128,endpoint=False); circle=np.c_[np.cos(angles),np.sin(angles)]
    for side in ('left','right'):
        body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,side+'_ankle_roll_link')
        if body<0: raise ValueError('missing foot')
        geoms=[i for i in range(m.ngeom) if m.geom_bodyid[i]==body and m.geom_contype[i] and
               m.geom_type[i]==mujoco.mjtGeom.mjGEOM_SPHERE]
        if not geoms: raise ValueError('foot collision spheres not found')
        for g in geoms:
            points.extend(d.geom_xpos[g,:2]+circle*m.geom_size[g,0])
        meshes=[i for i in range(m.ngeom) if m.geom_bodyid[i]==body and
                m.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH]
        if not meshes: raise ValueError('foot visual mesh not found')
        side_vertices=[]
        for g in meshes:
            mesh=int(m.geom_dataid[g]); start=int(m.mesh_vertadr[mesh]); count=int(m.mesh_vertnum[mesh])
            vertices=np.asarray(m.mesh_vert[start:start+count],dtype=np.float64)
            vertices=vertices@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
            mesh_points.extend(vertices[:,:2]); side_vertices.extend(vertices)
        per_side[side]={'sphere_count':len(geoms),
                        'lowest_sphere_world_z_m':float(min(d.geom_xpos[g,2]-m.geom_size[g,0] for g in geoms)),
                        'mesh_geom_count':len(meshes),'mesh_vertex_count':len(side_vertices),
                        'lowest_mesh_world_z_m':float(np.min(np.asarray(side_vertices)[:,2]))}
    hull=ConvexHull(np.asarray(points)); eq=hull.equations
    margin=float(np.min(-(eq[:,:2]@com[:2]+eq[:,2])/np.linalg.norm(eq[:,:2],axis=1)))
    mesh_hull=ConvexHull(np.asarray(mesh_points)); mesh_eq=mesh_hull.equations
    mesh_margin=float(np.min(-(mesh_eq[:,:2]@com[:2]+mesh_eq[:,2])/np.linalg.norm(mesh_eq[:,:2],axis=1)))
    out={'scope':__doc__,'com_world_m':com.tolist(),'optimistic_full_foot_hull_margin_m':margin,
         'visual_mesh_projection_hull_margin_m':mesh_margin,
         'com_inside_optimistic_hull':bool(margin>=0), 'feet':per_side,
         'assumptions':['static gravity only','all modeled foot spheres are possible contact points',
                        'no external horizontal support or torque','sampled sphere projections'],
         'hardware_ready':False}
    a.out.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))


if __name__=='__main__': main()
