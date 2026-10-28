"""Compile an isolated native X2 scene matching one Omni candidate's terrain.

Uses the existing project's robot and physics entry points. This prepares
continuous validation; it does not approve the reference or launch training.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--sole-profile',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('Output exists')
    if os.environ.get('SM_SOLE_BOX_CONFIG'):raise ValueError('Pass a sole profile explicitly; do not inherit a hidden override')
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra',check_assets=True)
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation
    from mjlab.scene import Scene,SceneCfg
    from mjlab.terrains import TerrainEntityCfg
    from mjlab.sim import MujocoCfg
    from supermimic.robots.assets import get_robot_cfg
    from supermimic.robots.dimensions import PROFILE
    from supermimic.data.physics_motion import differentiate_qpos
    candidate=json.loads((a.candidate/'result.json').read_text())
    robot=get_robot_cfg()
    if a.sole_profile:
        from supermimic.robots.sole_collision import apply_sole_box_profile
        original_spec=robot.spec_fn
        robot.spec_fn=lambda:apply_sole_box_profile(original_spec(),a.sole_profile)
    scene=Scene(SceneCfg(terrain=TerrainEntityCfg(terrain_type='plane'),entities={'robot':robot}),device='cpu')
    baseline=scene.compile()
    for i,box in enumerate(candidate['scene_boxes']):
        center,half,rotation=(np.asarray(box[k],dtype=float) for k in ['center','half_size','rotation_world'])
        if center.shape!=(3,) or half.shape!=(3,) or rotation.shape!=(3,3) or not np.isfinite(np.r_[center,half,rotation.ravel()]).all() or np.any(half<=0):raise ValueError('Invalid fixed box')
        np.testing.assert_allclose(rotation.T@rotation,np.eye(3),atol=1e-8)
        if abs(np.linalg.det(rotation)-1)>1e-8:raise ValueError('Improper box rotation')
        scene.spec.worldbody.add_geom(name=f'omni_native_box_{i}',type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=center,size=half,quat=Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]],
            contype=1,conaffinity=1,condim=3,friction=(.8,.005,.0001),rgba=(.2,.45,.7,1))
    model=scene.compile();physics=MujocoCfg(timestep=.005,iterations=10,ls_iterations=20)
    physics.apply(model);physics.apply(baseline)
    for field in ['nq','nv','nu','nbody']:
        if getattr(model,field)!=getattr(baseline,field):raise ValueError(f'Robot dimension changed: {field}')
    for field in ['body_mass','body_inertia','jnt_range','dof_armature','actuator_forcerange','actuator_ctrlrange','actuator_gainprm','actuator_biasprm']:
        np.testing.assert_array_equal(getattr(model,field),getattr(baseline,field))
    if (model.nq,model.nv,model.nu)!=(38,37,31):raise ValueError('Expected actuated X2 only; terrain must be fixed')
    if [model.joint(i).name.rsplit('/',1)[-1] for i in range(1,model.njnt)]!=PROFILE['joint_names']:raise ValueError('Reference/native joint order differs')
    retarget=mujoco.MjModel.from_xml_path(str(a.candidate/'retarget-scene.xml'))
    for i in range(1,model.nbody):
        name=model.body(i).name.rsplit('/',1)[-1]
        other=mujoco.mj_name2id(retarget,mujoco.mjtObj.mjOBJ_BODY,name)
        if other<0:
            if model.body_mass[i]!=0:raise ValueError(f'Unknown massive native body {name}')
            continue
        for field in ['body_mass','body_inertia','body_ipos','body_iquat']:
            np.testing.assert_allclose(getattr(model,field)[i],getattr(retarget,field)[other],atol=1e-12,rtol=0)
    np.testing.assert_allclose(model.jnt_range[1:],retarget.jnt_range[1:32],atol=1e-12,rtol=0)
    probes=[]
    for i,box in enumerate(candidate['scene_boxes']):
        geom=model.geom(f'omni_native_box_{i}')
        if int(model.geom_bodyid[geom.id])!=0:raise ValueError('Terrain is not fixed')
        center,half,rotation=(np.asarray(box[k]) for k in ['center','half_size','rotation_world'])
        for label,point in [('top',center+rotation[:,2]*(half[2]+.019)),('front',center-rotation[:,0]*(half[0]+.019))]:
            probe_spec=scene.spec.copy();body=probe_spec.worldbody.add_body(name='probe',pos=point);body.add_freejoint()
            body.add_geom(name='probe_geom',type=mujoco.mjtGeom.mjGEOM_SPHERE,size=(.02,0,0),mass=.01,contype=1,conaffinity=0)
            pm=probe_spec.compile();pd=mujoco.MjData(pm);mujoco.mj_forward(pm,pd)
            ids={pm.geom(f'omni_native_box_{i}').id,pm.geom('probe_geom').id}
            contacts=[float(c.dist) for c in pd.contact if {int(c.geom1),int(c.geom2)}==ids]
            if not contacts:raise ValueError(f'Missing actual native contact: box {i} {label}')
            probes.append({'box':i,'face':label,'distances_m':contacts})
    with np.load(a.candidate/'x2-reference.npz',allow_pickle=False) as z:q=z['qpos'].copy();fps=float(np.asarray(z['fps']).reshape(-1)[0])
    if q.shape[1]!=model.nq:raise ValueError('Reference differs from native robot dimensions')
    a.out.mkdir(parents=True)
    mujoco.mj_saveModel(model,str(a.out/'native.mjb'))
    np.savez_compressed(a.out/'reference-qpos.npz',qpos=q,qvel=differentiate_qpos(model,q,fps),fps=np.array([fps]))
    report={'nq':model.nq,'nv':model.nv,'nu':model.nu,'physics':asdict(physics),
        'robot_inertia_limits_actuators_unchanged':True,'scene_boxes':candidate['scene_boxes'],'contact_probes':probes,
        'source_motion':candidate['source_motion'],'reference_sha256':hashlib.sha256((a.candidate/'x2-reference.npz').read_bytes()).hexdigest(),
        'sole_profile_sha256':hashlib.sha256(a.sole_profile.read_bytes()).hexdigest() if a.sole_profile else None,
        'training_approved':False,'scope':'Native actuated scene and contact probes only; no continuous climb or learned policy execution'}
    (a.out/'scene-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
