"""Adapt G1 terrain FK landmarks to X2 using the official OmniRetarget core.

Uses the documented custom world-joint-position interface. Never copies G1
joint angles into X2. Start with a bounded few-frame CPU smoke on the server.
"""
import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import zipfile


def literal_assignment(path,name):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):return ast.literal_eval(node.value)
    raise ValueError(f'Missing upstream constant {name}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--source-audit',type=Path,required=True)
    p.add_argument('--x2-model',type=Path,required=True)
    p.add_argument('--equilibrium',type=Path,required=True)
    p.add_argument('--motion',required=True)
    p.add_argument('--frames',type=int,default=5)
    p.add_argument('--penetration-target-m',type=float,default=.0008,help='IK target tighter than the independent 1mm collision gate')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists() or a.frames<2 or not 0<a.penetration_target_m<.001:p.error('New output, at least two frames, and a positive IK penetration target below 1mm required')
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation
    from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter
    from holosoma_retargeting.src.utils import extract_foot_sticking_sequence_velocity
    a.out.mkdir(parents=True);(a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    meta=json.loads((a.x2_model/'model.json').read_text())
    source=json.loads((a.source_audit/'screening.json').read_text())
    row=next(x for x in source['motions'] if x['motion']==a.motion)
    upstream=a.source/'holosoma-sources/src/holosoma_retargeting/holosoma_retargeting/config_types'
    source_names=literal_assignment(upstream/'data_conversion.py','_ROBOT_JOINT_NAMES_DEFAULT')['g1']
    source_mapping=literal_assignment(upstream/'data_type.py','JOINTS_MAPPINGS')[('mocap','g1')]
    mapping=meta['mocap_joint_mapping'];demo_names=list(mapping);scale=1.30/1.32
    with zipfile.ZipFile(a.source/'robot-terrain.zip') as archive:
        with np.load(io.BytesIO(archive.read(a.motion)),allow_pickle=False) as z:poses=z['qpos'][:a.frames].copy();fps=float(np.asarray(z['fps']).reshape(-1)[0])
    if fps!=30:raise ValueError('The upstream core currently writes 30Hz; explicit resampling is needed for other sources')
    g1=mujoco.MjModel.from_binary_path(str(a.source_audit/'source-g1.mjb'));gd=mujoco.MjData(g1)
    source_columns=[int(g1.jnt_qposadr[g1.joint(n).id]) for n in source_names]
    ids=[g1.body(source_mapping[n]).id for n in demo_names];landmarks=[]
    for pose in poses:
        gd.qpos[:3]=pose[4:7];gd.qpos[3:7]=pose[:4];gd.qpos[source_columns]=pose[7:];mujoco.mj_kinematics(g1,gd)
        landmarks.append(gd.xpos[ids].copy()*scale)
    landmarks=np.asarray(landmarks)
    spec=mujoco.MjSpec.from_file(str(a.x2_model/'x2_omni_markers.xml'))
    spec.worldbody.add_geom(name='ground',type=mujoco.mjtGeom.mjGEOM_PLANE,size=(0,0,.05),contype=1,conaffinity=1)
    # The upstream API represents terrain as a fixed-in-time auxiliary free
    # pose suffix. It is locked to identity throughout IK, never simulated.
    terrain=spec.worldbody.add_body(name='multi_boxes');terrain.add_freejoint(name='omni_auxiliary_terrain_pose')
    anchors=[];target_boxes=[]
    for i,box in enumerate(row['boxes']):
        center=np.array(box['center'])*scale;half=np.array(box['half_size'])*scale;rotation=np.array(box['rotation_world'])
        quat=Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]]
        terrain.add_geom(name=f'multi_boxes_{i}',type=mujoco.mjtGeom.mjGEOM_BOX,pos=center,size=half,quat=quat,mass=1.,contype=1,conaffinity=1)
        if box['high'][2]>.1:
            corners=np.array([(x,y,z) for x in (-1,1) for y in (-1,1) for z in (-1,1)])*half
            anchors.extend(corners@rotation.T+center)
        target_boxes.append({'center':center.tolist(),'half_size':half.tolist(),'rotation_world':rotation.tolist()})
    for x in np.linspace(poses[:,4].min()*scale-.5,poses[:,4].max()*scale+.5,3):
        for y in np.linspace(poses[:,5].min()*scale-.5,poses[:,5].max()*scale+.5,3):anchors.append([x,y,0.])
    anchors=np.unique(np.asarray(anchors),axis=0)
    scene=a.out/'retarget-scene.xml';scene.write_text(spec.to_xml())
    model=mujoco.MjModel.from_xml_path(str(scene))
    if model.nq!=45 or model.nv!=43 or model.jnt_qposadr[model.joint('omni_auxiliary_terrain_pose').id]!=38:raise ValueError('Unexpected robot/auxiliary-pose layout')
    lower=np.r_[np.full(3,-1e6),np.full(4,-1.),np.zeros(31)];upper=np.r_[np.full(3,1e6),np.ones(4),np.zeros(31)]
    for name in meta['joint_names']:
        joint=model.joint(name).id;address=int(model.jnt_qposadr[joint]);lower[address],upper[address]=model.jnt_range[joint]
    seed=np.load(a.equilibrium,allow_pickle=False)['qpos'].copy()
    if seed.shape!=(38,):raise ValueError('X2 seed dimensions differ')
    seed[:2]=poses[0,4:6]*scale;seed[3:7]=poses[0,:4]
    for name in meta['joint_names']:
        if 'wrist' not in name:continue
        address=int(model.jnt_qposadr[model.joint(name).id]);lower[address]=max(lower[address],seed[address]-np.radians(15));upper[address]=min(upper[address],seed[address]+np.radians(15))
    constants=SimpleNamespace(ROBOT_URDF_FILE=str(a.x2_model/'x2.urdf'),OBJECT_NAME='multi_boxes',SCENE_XML_FILE=str(scene),ROBOT_DOF=31,
        FOOT_STICKING_LINKS=meta['foot_sticking_links'],DEMO_JOINTS=demo_names,JOINTS_MAPPING=mapping,
        MANUAL_LB={str(i):float(v) for i,v in enumerate(lower)},MANUAL_UB={str(i):float(v) for i,v in enumerate(upper)},MANUAL_COST={},NOMINAL_TRACKING_INDICES=np.array([],dtype=int))
    retargeter=InteractionMeshRetargeter(constants,object_urdf_path=None,visualize=False,debug=False,penetration_tolerance=a.penetration_target_m)
    # Complete qpos-address bounds avoid the upstream named-free-joint indexing
    # assumption; every actuated range is checked against the actual X2 model.
    np.testing.assert_array_equal(retargeter.q_a_lb,lower);np.testing.assert_array_equal(retargeter.q_a_ub,upper)
    objects=np.tile([0.,0.,0.,1.,0.,0.,0.],(len(poses),1))
    sticking=extract_foot_sticking_sequence_velocity(landmarks,demo_names,['LeftToeBase','RightToeBase'])
    retargeter.retarget_motion(landmarks,objects,objects,anchors,anchors,sticking,q_a_init=seed,original=True,dest_res_path=str(a.out/'upstream-result.npz'))
    with np.load(a.out/'upstream-result.npz',allow_pickle=False) as z:result=z['qpos'].copy()
    if not np.isfinite(result).all():raise ValueError('Non-finite result')
    quaternion_error=float(abs(np.linalg.norm(result[:,3:7],axis=1)-1).max())
    if quaternion_error>1e-6:raise ValueError('Non-unit output root quaternion')
    np.testing.assert_array_equal(result[:,38:],objects)
    hard=0.;penetration=0.;data=mujoco.MjData(model);witness=np.zeros(6)
    obstacles=[g for g in range(model.ngeom) if model.geom(g).name=='ground' or model.geom(g).name.startswith('multi_boxes_')]
    robot_geoms=[g for g in range(model.ngeom) if g not in obstacles and (model.geom_contype[g] or model.geom_conaffinity[g])]
    for q in result:
        data.qpos[:]=q;mujoco.mj_forward(model,data)
        for name in meta['joint_names']:
            joint=model.joint(name).id;value=q[model.jnt_qposadr[joint]];lo,hi=model.jnt_range[joint];hard=max(hard,float(max(lo-value,value-hi,0.)))
        for g in robot_geoms:
            for obstacle in obstacles:penetration=max(penetration,-float(mujoco.mj_geomDistance(model,data,g,obstacle,.02,witness)))
    np.savez_compressed(a.out/'x2-reference.npz',qpos=result[:,:38],fps=fps)
    report={'source_motion':a.motion,'frames':len(result),'fps':fps,'uniform_source_scene_scale':scale,'scene_boxes':target_boxes,
        'hard_joint_violation_rad':hard,'max_environment_penetration_m':penetration,'bounds_match_x2_model':True,
        'maximum_root_quaternion_norm_error':quaternion_error,'ik_penetration_target_m':a.penetration_target_m,
        'source_model_sha256':meta['source_sha256'],'policy_evaluation':False,'training_approved':False,
        'scope':'Bounded OmniRetarget custom-data smoke only; not full climbing, inverse-dynamics feasibility or policy execution.'}
    (a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
