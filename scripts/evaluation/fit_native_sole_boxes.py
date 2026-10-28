"""Fit conservative sole boxes inside existing foot convex hulls.

Produces a separate native model. Body inertias, actuators, friction and solver
settings are unchanged. The fit uses the supplied level reference orientation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def inscribed_rectangle(hull_equations, z_center, half_height, points):
    """Maximize planar box area with every corner inside the 3D convex hull."""
    normals=hull_equations[:,:3];offset=hull_equations[:,3]
    matrix=np.c_[normals[:,:2],abs(normals[:,:2])]
    rhs=-offset-normals[:,2]*z_center-abs(normals[:,2])*half_height
    center=points[:,:2].mean(axis=0)
    seed=np.r_[center,.005,.005]
    fit=minimize(lambda x:-np.log(x[2])-np.log(x[3]),seed,
        jac=lambda x:np.r_[0.,0.,-1/x[2],-1/x[3]],method='SLSQP',
        constraints=[LinearConstraint(matrix,-np.inf,rhs)],
        bounds=[(points[:,0].min(),points[:,0].max()),(points[:,1].min(),points[:,1].max()),(.001,.15),(.001,.15)],
        options={'maxiter':300,'ftol':1e-12})
    violation=float(np.maximum(matrix@fit.x-rhs,0).max())
    if not fit.success or violation>1e-8:raise RuntimeError(f'Inscribed sole fit failed: {fit.message}; violation={violation}')
    return fit.x,violation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--level-pose',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    m=mujoco.MjModel.from_binary_path(str(args.model));d=mujoco.MjData(m)
    d.qpos[:]=np.load(args.level_pose)['qpos'];mujoco.mj_forward(m,d)
    root=int(m.jnt_bodyid[0]);heading=Rotation.from_quat(d.qpos[[4,5,6,3]]).as_euler('xyz')[2]
    basis=Rotation.from_euler('z',heading).as_matrix()
    protected=['body_mass','body_inertia','body_ipos','body_iquat','jnt_range','dof_armature','dof_damping',
        'actuator_gainprm','actuator_biasprm','actuator_forcerange','geom_friction','geom_solref','geom_solimp',
        'geom_contype','geom_conaffinity','mesh_vert','mesh_face']
    before={n:np.asarray(getattr(m,n)).copy() for n in protected}
    rows=[]
    for side in ['left','right']:
        b=m.body('robot/'+side+'_ankle_roll_link').id
        candidates=[g for g in range(m.ngeom) if m.geom_bodyid[g]==b and (m.geom_contype[g] or m.geom_conaffinity[g])]
        if len(candidates)!=1:raise ValueError('Expected one existing collision geom per foot')
        g=candidates[0];k=int(m.geom_dataid[g]);a,n=m.mesh_vertadr[k],m.mesh_vertnum[k]
        world=m.mesh_vert[a:a+n].astype(float)@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
        points=(world-d.xpos[b])@basis
        hull=ConvexHull(points)
        hz=.004;cz=float(points[:,2].min()+.00025+hz)
        box,violation=inscribed_rectangle(hull.equations,cz,hz,points)
        cx,cy,hx,hy=box;center_world=d.xpos[b]+basis@np.array([cx,cy,cz])
        body_rotation=d.xmat[b].reshape(3,3)
        m.geom_pos[g]=body_rotation.T@(center_world-d.xpos[b])
        m.geom_quat[g]=Rotation.from_matrix(body_rotation.T@basis).as_quat()[[3,0,1,2]]
        m.geom_type[g]=int(mujoco.mjtGeom.mjGEOM_BOX);m.geom_dataid[g]=-1
        m.geom_size[g]=[hx,hy,hz];m.geom_sameframe[g]=0
        m.geom_rbound[g]=np.linalg.norm([hx,hy,hz]);m.geom_aabb[g]=[0,0,0,hx,hy,hz]
        rows.append({'side':side,'geom':m.geom(g).name,'full_dimensions_m':[2*hx,2*hy,2*hz],
            'bottom_above_original_mesh_minimum_m':.00025,'hull_constraint_violation_m':violation,
            'footprint_area_m2':float(4*hx*hy),'center_in_level_basis_m':[cx,cy,cz]})
    for name,value in before.items():np.testing.assert_array_equal(value,getattr(m,name))
    model_path=args.out/'native-sole-boxes.mjb';mujoco.mj_saveModel(m,str(model_path))
    report={'source_model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),
        'model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'level_pose_sha256':hashlib.sha256(args.level_pose.read_bytes()).hexdigest(),
        'sole_boxes':rows,'verified_unchanged_arrays':protected,
        'scope':'Independent model copy; existing training models and inputs untouched.',
        'approved_for_training':False,'dynamic_tracking_verified':False}
    (args.out/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
