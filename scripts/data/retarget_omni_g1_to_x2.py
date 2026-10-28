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
    p.add_argument('--start-frame',type=int,default=0,help='Source window start; nonzero entries are diagnostic segments, not complete climbs')
    p.add_argument('--wrist-reference-limit-deg',type=float,default=15.,help='Optional task bound around neutral wrists; 0 uses unchanged mechanical limits and a weak neutral prior')
    p.add_argument('--penetration-target-m',type=float,default=.0008,help='IK target tighter than the independent 1mm collision gate')
    p.add_argument('--source-flat-foot-lock',action='store_true',help='Pin sole marker heights only during source-confirmed flat support')
    p.add_argument('--collision-backtracking',action='store_true',help='Check actual collision distance before accepting each SQP step')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists() or a.frames<2 or a.start_frame<0 or not 0<=a.wrist_reference_limit_deg<=180 or not 0<a.penetration_target_m<.001:p.error('New output, valid source window/wrist task bound and a positive IK penetration target below 1mm required')
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation
    from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter
    from holosoma_retargeting.src.utils import extract_foot_sticking_sequence_velocity
    from holosoma_retargeting.src.utils import calculate_laplacian_coordinates
    from holosoma_retargeting.config_types.retargeter import FootLockConfig
    class X2Retargeter(InteractionMeshRetargeter):
        def _update_jacobians_and_phis_from_q(self,q):
            jacobians,distances=super()._update_jacobians_and_phis_from_q(q)
            if a.collision_backtracking:
                m=self.robot_model;d=self.robot_data
                for pair in list(distances):
                    robot=[g for g in pair if m.geom(g).name.startswith('foot_')]
                    obstacle=[g for g in pair if m.geom(g).name=='ground' or m.geom(g).name.startswith('multi_boxes_')]
                    if len(robot)!=1 or len(obstacle)!=1:continue
                    g,o=robot[0],obstacle[0];w=np.zeros(6);distance=float(mujoco.mj_geomDistance(m,d,g,o,.02,w));vector=w[:3]-w[3:];length=np.linalg.norm(vector)
                    if length<1e-10:continue
                    normal=np.sign(distance)*vector/length
                    if normal[2]<.999999 or int(m.geom_type[g])!=int(mujoco.mjtGeom.mjGEOM_MESH):continue
                    mesh=int(m.geom_dataid[g]);adr,count=m.mesh_vertadr[mesh],m.mesh_vertnum[mesh]
                    vertices=m.mesh_vert[adr:adr+count]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
                    if int(m.geom_type[o])==int(mujoco.mjtGeom.mjGEOM_BOX):
                        relative=(vertices-d.geom_xpos[o])@d.geom_xmat[o].reshape(3,3)
                        vertices=vertices[(abs(relative[:,:2])<=m.geom_size[o,:2]).all(1)]
                    if not len(vertices):continue
                    vertices=vertices[vertices[:,2]<=vertices[:,2].min()+.02]
                    directions=np.array([[np.cos(t),np.sin(t)] for t in np.arange(8)*np.pi/4])
                    indices=np.unique(np.r_[vertices[:,2].argmin(),(vertices[:,:2]@directions.T).argmax(0)])
                    body=int(m.geom_bodyid[g]);rotation=d.xmat[body].reshape(3,3).copy();position=d.xpos[body].copy()
                    for number,index in enumerate(indices):
                        point=rotation.T@(vertices[index]-position);key=('active_sole_vertex',g,o,number)
                        jacobians[key]=normal@self._calc_contact_jacobian_from_point(body,point)
                        distances[key]=float(normal@(vertices[index]-w[3:]))
            # A flat sole can have several equally closest points. A single
            # selected witness does not constrain the other active corners.
            # Cuts retain actual rigid-body surface points from rejected
            # steps; no new collision volume or support surface is invented.
            for i,(body,point,anchor,normal) in enumerate(getattr(self,'_plane_cuts',[])):
                world=self.robot_data.xpos[body]+self.robot_data.xmat[body].reshape(3,3)@point
                jacobians[('plane_witness',i)]=normal@self._calc_contact_jacobian_from_point(body,point)
                distances[('plane_witness',i)]=float(normal@(world-anchor))
            return jacobians,distances

        def _add_horizontal_plane_witnesses(self,q):
            m=self.robot_model;d=self.robot_data;d.qpos[:]=q;mujoco.mj_forward(m,d);added=0
            obstacles=[g for g in range(m.ngeom) if m.geom(g).name=='ground' or m.geom(g).name.startswith('multi_boxes_')]
            robot=[g for g in range(m.ngeom) if g not in obstacles and (m.geom_contype[g] or m.geom_conaffinity[g])]
            for g in robot:
                for o in obstacles:
                    witness=np.zeros(6);distance=float(mujoco.mj_geomDistance(m,d,g,o,.02,witness));vector=witness[:3]-witness[3:];length=np.linalg.norm(vector)
                    if distance>=-self.penetration_tolerance or length<1e-10:continue
                    normal=np.sign(distance)*vector/length
                    if normal[2]<.999999:continue
                    body=int(m.geom_bodyid[g]);point=d.xmat[body].reshape(3,3).T@(witness[:3]-d.xpos[body]);anchor=witness[3:].copy()
                    if any(b==body and np.linalg.norm(p-point)<1e-7 and abs(float(n@(a-anchor)))<1e-7 for b,p,a,n in self._plane_cuts):continue
                    self._plane_cuts.append((body,point,anchor,normal));added+=1
            return added

        def _maximum_environment_penetration(self,q):
            m=self.robot_model;d=self.robot_data;d.qpos[:]=q;mujoco.mj_forward(m,d)
            obstacles=[g for g in range(m.ngeom) if m.geom(g).name=='ground' or m.geom(g).name.startswith('multi_boxes_')]
            robot=[g for g in range(m.ngeom) if g not in obstacles and (m.geom_contype[g] or m.geom_conaffinity[g])]
            witness=np.zeros(6)
            return max(0.,max(-float(mujoco.mj_geomDistance(m,d,g,o,.02,witness)) for g in robot for o in obstacles))

        def _actual_tracking_cost(self,q,kwargs):
            d=self.robot_data;d.qpos[:]=q;mujoco.mj_kinematics(self.robot_model,d)
            points=np.array([d.xpos[self.robot_model.body(name).id].copy() for name in self.laplacian_match_links.values()])
            lap=calculate_laplacian_coordinates(np.vstack([points,kwargs['obj_pts_local']]),kwargs['adj_list'])
            tracking=self.laplacian_weights*float(np.square(lap-kwargs['target_laplacian']).sum())
            smooth=self.smooth_weight*float(np.square(q[self.q_a_indices]-kwargs['q_t_last'][self.q_a_indices]).sum())
            nominal=0.
            if kwargs.get('q_a_nominal') is not None and kwargs.get('w_nominal_tracking',0)>0:
                idx=np.asarray(self.track_nominal_indices,dtype=int)
                nominal=kwargs['w_nominal_tracking']*float(np.square(q[self.q_a_indices][idx]-kwargs['q_a_nominal'][idx]).sum())
            return tracking+smooth+nominal+float(np.sum(self.Q_diag*np.square(q[self.q_a_indices])))

        def _constraint_merit(self,q,frame):
            penetration=self._maximum_environment_penetration(q);foot_error=0.
            if self.foot_lock.enable:
                for name,body in self.foot_links.items():
                    target=self._is_foot_locked_in_window(name,frame)
                    if target is not None:foot_error=max(foot_error,abs(float(self.robot_data.xpos[self.robot_model.body(body).id,2])-target))
            violation=max(0.,penetration-self.penetration_tolerance,foot_error-self.foot_lock.tolerance)
            return violation,penetration,foot_error

        def iterate(self,**kwargs):
            if not a.collision_backtracking:return super().iterate(**kwargs)
            q=kwargs.pop('q_n').copy();budget=kwargs.pop('n_iter')*3;last_cost=np.inf;backtracks=0;self._plane_cuts=[]
            last_actual=self._actual_tracking_cost(q,kwargs);stopping_reason='iteration_budget'
            allowed=5e-5
            new_support=any(start==kwargs['frame_idx'] for ranges in self._foot_lock_windows.values() for start,end in ranges)
            for iteration in range(budget):
                before,_,_=self._constraint_merit(q,kwargs['frame_idx'])
                accepted=None
                original_step=self.step_size
                try:
                    for contraction in range(7):
                        self.step_size=original_step*2.**(-contraction)
                        solve_kwargs=dict(kwargs)
                        # The initial feasibility fallback may drop the trust
                        # region only for the first initialization step.
                        solve_kwargs['init_t']=(kwargs.get('init_t',False) and iteration==0) or (new_support and before>allowed)
                        try:proposed,cost=self.solve_single_iteration(q_a_n_last=q[self.q_a_indices],**solve_kwargs)
                        except RuntimeError as error:
                            np.savez_compressed(a.out/f'failed-frame-{kwargs["frame_idx"]}.npz',qpos=q)
                            (a.out/'solver-failure.json').write_text(json.dumps({'frame':kwargs['frame_idx'],'iteration':iteration,'constraint_merit_m':before,'trust_step':self.step_size,'source_support_transition':new_support,'initial_feasibility_fallback_enabled':solve_kwargs['init_t'],'error':str(error)},indent=2)+'\n')
                            raise
                        for reduction in range(15):
                            alpha=2.**(-reduction);trial=q+alpha*(proposed-q);mujoco.mj_normalizeQuat(self.robot_model,trial)
                            merit,penetration,foot_error=self._constraint_merit(trial,kwargs['frame_idx'])
                            if merit<=allowed or merit<before-1e-8:
                                accepted=trial;backtracks+=reduction;break
                        if accepted is not None:break
                finally:self.step_size=original_step
                if accepted is None:
                    if self._add_horizontal_plane_witnesses(proposed):continue
                    np.savez_compressed(a.out/f'failed-frame-{kwargs["frame_idx"]}.npz',qpos=q,proposed=proposed)
                    raise RuntimeError(f'No contact-merit descent at frame {kwargs["frame_idx"]}; violation={before}')
                accepted_step=float(np.linalg.norm(accepted-q));q=accepted
                actual=self._actual_tracking_cost(q,kwargs)
                if iteration>=2 and accepted_step<1e-4 and abs(actual-last_actual)<1e-5*max(1.,abs(actual)) and merit<=allowed:
                    stopping_reason='small_accepted_step_and_actual_objective_change';break
                if reduction==0 and np.isclose(cost,last_cost) and merit<=allowed:
                    stopping_reason='full_step_cost_converged';break
                last_cost=cost
                last_actual=actual
            final_merit,final,final_foot_error=self._constraint_merit(q,kwargs['frame_idx'])
            if final_merit>allowed:
                np.savez_compressed(a.out/f'failed-frame-{kwargs["frame_idx"]}.npz',qpos=q)
                raise RuntimeError(f'Contact SQP did not converge at frame {kwargs["frame_idx"]}: violation {final_merit}m after {budget} iterations')
            if not hasattr(self,'backtracking_records'):self.backtracking_records=[]
            if not hasattr(self,'_accepted_poses'):self._accepted_poses=[]
            self._accepted_poses.append(q[:38].copy())
            self.backtracking_records.append({'frame':kwargs['frame_idx'],'iterations':iteration+1,'backtracks':backtracks,'plane_witness_cuts':len(self._plane_cuts),'maximum_penetration_m':final,'maximum_locked_sole_height_error_m':final_foot_error,'actual_tracking_cost':actual,'accepted_step_norm':accepted_step,'stopping_reason':stopping_reason})
            if kwargs['frame_idx']%30==0:
                (a.out/'collision-backtracking-progress.json').write_text(json.dumps(self.backtracking_records,indent=2)+'\n')
                np.savez_compressed(a.out/f'accepted-prefix-{kwargs["frame_idx"]:05d}.npz',qpos=np.asarray(self._accepted_poses),fps=30.,source_start_frame=a.start_frame,source_motion=a.motion)
            return q,cost

        def _build_transform_qdot_to_qvel_fast(self, use_world_omega=True):
            # Retain the upstream default quaternion formula, checked below
            # against native FK differences. Avoid tuple membership between
            # NumPy scalars and pybind enums: the
            # enum's reverse equality can silently miss every hinge joint.
            transform=super()._build_transform_qdot_to_qvel_fast(use_world_omega=True)
            for joint in range(self.robot_model.njnt):
                if int(self.robot_model.jnt_type[joint]) in (int(mujoco.mjtJoint.mjJNT_HINGE),int(mujoco.mjtJoint.mjJNT_SLIDE)):
                    transform[int(self.robot_model.jnt_dofadr[joint]),int(self.robot_model.jnt_qposadr[joint])]=1.
            return transform
    a.out.mkdir(parents=True);(a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    meta=json.loads((a.x2_model/'model.json').read_text())
    source=json.loads((a.source_audit/'screening.json').read_text())
    row=next(x for x in source['motions'] if x['motion']==a.motion)
    upstream=a.source/'holosoma-sources/src/holosoma_retargeting/holosoma_retargeting/config_types'
    source_names=literal_assignment(upstream/'data_conversion.py','_ROBOT_JOINT_NAMES_DEFAULT')['g1']
    source_mapping=literal_assignment(upstream/'data_type.py','JOINTS_MAPPINGS')[('mocap','g1')]
    mapping=meta['mocap_joint_mapping'];demo_names=list(mapping);scale=1.30/1.32
    with zipfile.ZipFile(a.source/'robot-terrain.zip') as archive:
        with np.load(io.BytesIO(archive.read(a.motion)),allow_pickle=False) as z:poses=z['qpos'][a.start_frame:a.start_frame+a.frames].copy();fps=float(np.asarray(z['fps']).reshape(-1)[0])
    if len(poses)<2:raise ValueError('Source window contains fewer than two frames')
    if fps!=30:raise ValueError('The upstream core currently writes 30Hz; explicit resampling is needed for other sources')
    g1=mujoco.MjModel.from_binary_path(str(a.source_audit/'source-g1.mjb'));gd=mujoco.MjData(g1)
    source_columns=[int(g1.jnt_qposadr[g1.joint(n).id]) for n in source_names]
    ids=[g1.body(source_mapping[n]).id for n in demo_names];landmarks=[];source_soles=[]
    source_hand_centers={}
    if meta.get('hand_centroid_markers'):
        from scipy.spatial import ConvexHull
        for name in ('LeftHandMiddle3','RightHandMiddle3'):
            body=g1.body(source_mapping[name]).id
            geoms=[g for g in range(g1.ngeom) if int(g1.geom_bodyid[g])==body and g1.geom_contype[g]]
            if len(geoms)!=1 or int(g1.geom_type[geoms[0]])!=int(mujoco.mjtGeom.mjGEOM_MESH):raise ValueError('Expected pinned G1 hand collision mesh')
            g=geoms[0];mesh=int(g1.geom_dataid[g]);adr,count=g1.mesh_vertadr[mesh],g1.mesh_vertnum[mesh]
            vertices=g1.mesh_vert[adr:adr+count]@Rotation.from_quat(g1.geom_quat[g][[1,2,3,0]]).as_matrix().T+g1.geom_pos[g]
            origin=vertices.mean(0);triangles=vertices[ConvexHull(vertices).simplices];volumes=abs(np.linalg.det(triangles-origin))/6
            source_hand_centers[name]=(body,(((triangles.sum(1)+origin)/4)*volumes[:,None]).sum(0)/volumes.sum())
    sole_bodies=[[g1.body(f'{side}_ankle_roll_sphere_{k}_link').id for k in range(1,5)] for side in ('left','right')]
    sole_radii=[]
    for bodies in sole_bodies:
        radii=[]
        for body in bodies:
            geoms=[g for g in range(g1.ngeom) if int(g1.geom_bodyid[g])==body and g1.geom_contype[g]]
            if not geoms or any(int(g1.geom_type[g])!=int(mujoco.mjtGeom.mjGEOM_SPHERE) for g in geoms):raise ValueError('Expected source sole spheres')
            radii.append(float(g1.geom_size[geoms[0],0]))
        sole_radii.append(radii)
    for pose in poses:
        gd.qpos[:3]=pose[4:7];gd.qpos[3:7]=pose[:4];gd.qpos[source_columns]=pose[7:];mujoco.mj_kinematics(g1,gd)
        points=gd.xpos[ids].copy()
        for name,(body,offset) in source_hand_centers.items():points[demo_names.index(name)]+=gd.xmat[body].reshape(3,3)@offset
        # The upstream human-mocap "Foot" maps to an ankle marker, whereas
        # our X2 "Foot" is a sole heel. Reusing those semantic labels would
        # force the X2 heel upward and leave only toe-edge support. Construct
        # corresponding heel/toe sole points from the actual G1 foot spheres.
        for side_index,side in enumerate(('left','right')):
            sole_up=gd.xmat[g1.body(side+'_ankle_roll_link').id].reshape(3,3)[:,2]
            centers=gd.xpos[sole_bodies[side_index]]
            prefix=('Left','Right')[side_index]
            for label,subset in [('Foot',[0,1]),('ToeBase',[2,3])]:
                points[demo_names.index(prefix+label)]=centers[subset].mean(0)-float(np.asarray(sole_radii[side_index])[subset].mean())*sole_up
        landmarks.append(points*scale)
        bottoms=gd.xpos[np.array(sole_bodies)].copy();bottoms[:,:,2]-=np.asarray(sole_radii)
        source_soles.append(bottoms)
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
    wrist_indices=[]
    for name in meta['joint_names']:
        if 'wrist' not in name:continue
        address=int(model.jnt_qposadr[model.joint(name).id]);wrist_indices.append(address)
        if a.wrist_reference_limit_deg:
            lower[address]=max(lower[address],seed[address]-np.radians(a.wrist_reference_limit_deg));upper[address]=min(upper[address],seed[address]+np.radians(a.wrist_reference_limit_deg))
    constants=SimpleNamespace(ROBOT_URDF_FILE=str(a.x2_model/'x2.urdf'),OBJECT_NAME='multi_boxes',SCENE_XML_FILE=str(scene),ROBOT_DOF=31,
        FOOT_STICKING_LINKS=meta['foot_sticking_links'],DEMO_JOINTS=demo_names,JOINTS_MAPPING=mapping,
        MANUAL_LB={str(i):float(v) for i,v in enumerate(lower)},MANUAL_UB={str(i):float(v) for i,v in enumerate(upper)},MANUAL_COST={},NOMINAL_TRACKING_INDICES=np.array(wrist_indices if a.wrist_reference_limit_deg==0 else [],dtype=int))
    windows={'left':[],'right':[]}
    if a.source_flat_foot_lock:
        for side in range(2):
            heights=[]
            for soles in source_soles:
                points=soles[side];surfaces=[0.] if abs(points[:,2]).max()<=.003 else []
                for box in row['boxes']:
                    relative=(points-np.asarray(box['center']))@np.asarray(box['rotation_world']);half=np.asarray(box['half_size'])
                    if (abs(relative[:,:2])<=half[:2]).all() and abs(relative[:,2]-half[2]).max()<=.003:surfaces.append(float(box['high'][2]))
                heights.append(max(surfaces)*scale+.0005 if surfaces else None)
            start=0
            for end in range(1,len(heights)+1):
                if end==len(heights) or heights[end]!=heights[start]:
                    if heights[start] is not None:windows[('left','right')[side]].append((start,end-1,heights[start]))
                    start=end
    (a.out/'source-flat-support.json').write_text(json.dumps({'enabled':a.source_flat_foot_lock,'windows':windows,'source_sphere_bottom_tolerance_m':.003,'x2_marker_height_offset_m':.0005},indent=2)+'\n')
    retargeter=X2Retargeter(constants,object_urdf_path=None,visualize=False,debug=False,penetration_tolerance=a.penetration_target_m,
        foot_lock=FootLockConfig(enable=a.source_flat_foot_lock,windows=windows,tolerance=.001),w_nominal_tracking_init=.05 if a.wrist_reference_limit_deg==0 else 5.)
    # Complete qpos-address bounds avoid the upstream named-free-joint indexing
    # assumption; every actuated range is checked against the actual X2 model.
    np.testing.assert_array_equal(retargeter.q_a_lb,lower);np.testing.assert_array_equal(retargeter.q_a_ub,upper)
    objects=np.tile([0.,0.,0.,1.,0.,0.,0.],(len(poses),1))
    # Independent central differences validate the complete robot Jacobian,
    # including its free-root quaternion and every actuated qpos coordinate.
    test_pose=np.r_[seed,objects[0]];check_data=mujoco.MjData(model);epsilon=1e-6;derivative_error=0.
    retargeter.robot_data.qpos[:]=test_pose
    for body_name in mapping.values():
        body=model.body(body_name).id
        retargeter.robot_data.qpos[:]=test_pose
        analytic=retargeter._calc_contact_jacobian_from_point(body,np.zeros(3))[:,:38]
        numeric=np.zeros((3,38))
        for column in range(38):
            samples=[]
            for sign in (1,-1):
                check_data.qpos[:]=test_pose;check_data.qpos[column]+=sign*epsilon
                mujoco.mj_normalizeQuat(model,check_data.qpos);mujoco.mj_kinematics(model,check_data)
                samples.append(check_data.xpos[body].copy())
            numeric[:,column]=(samples[0]-samples[1])/(2*epsilon)
        derivative_error=max(derivative_error,float(abs(analytic-numeric).max()))
    (a.out/'jacobian-check.json').write_text(json.dumps({'max_absolute_error':derivative_error,'columns':38,'bodies':len(mapping),'epsilon':epsilon,'passed':derivative_error<1e-6},indent=2)+'\n')
    if derivative_error>=1e-6:raise ValueError(f'Independent X2 Jacobian check failed: {derivative_error}')
    sticking=extract_foot_sticking_sequence_velocity(landmarks,demo_names,['LeftToeBase','RightToeBase'])
    nominal=np.tile(np.r_[seed,objects[0]],(len(poses),1)) if a.wrist_reference_limit_deg==0 else None
    retargeter.retarget_motion(landmarks,objects,objects,anchors,anchors,sticking,q_a_init=seed,q_nominal_list=nominal,original=True,dest_res_path=str(a.out/'upstream-result.npz'))
    if a.collision_backtracking:(a.out/'collision-backtracking.json').write_text(json.dumps(retargeter.backtracking_records,indent=2)+'\n')
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
    report={'source_motion':a.motion,'source_start_frame':a.start_frame,'frames':len(result),'fps':fps,'uniform_source_scene_scale':scale,'scene_boxes':target_boxes,
        'hard_joint_violation_rad':hard,'max_environment_penetration_m':penetration,'bounds_match_x2_model':True,
        'maximum_root_quaternion_norm_error':quaternion_error,'ik_penetration_target_m':a.penetration_target_m,
        'jacobian_max_absolute_error':derivative_error,'max_actuated_joint_range_rad':float(np.ptp(result[:,7:38],axis=0).max()),
        'source_foot_landmarks':'Heel/toe sole surface points from paired G1 spheres; upstream ankle marker is not a heel',
        'collision_backtracking':a.collision_backtracking,
        'wrist_reference_limit_deg':a.wrist_reference_limit_deg,'paired_hand_collision_centroids':bool(source_hand_centers),
        'source_model_sha256':meta['source_sha256'],'policy_evaluation':False,'training_approved':False,
        'scope':'Bounded OmniRetarget custom-data smoke only; not full climbing, inverse-dynamics feasibility or policy execution.'}
    (a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
