"""Export the verified X2 model with massless sole tracking markers for OmniRetarget.

Run on the server. Markers add no collision geometry or physical mass. This is
model preparation only, not a retargeting result or a feasibility certificate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--hand-centroid-markers',action='store_true',help='Use actual hand collision-hull centres rather than wrist joint origins as hand landmarks')
    a=p.parse_args()
    if a.out.exists():p.error('Output exists')
    if os.environ.get('SM_SOLE_BOX_CONFIG'):raise ValueError('Use original collision model for marker preparation')
    from supermimic.robots.paths import ROOT,select_robot
    select_robot('x2_ultra',check_assets=True)
    import mujoco
    import numpy as np
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation
    from supermimic.robots.assets import get_spec
    from supermimic.robots.dimensions import PROFILE,robot_xml_path
    spec=get_spec();original=spec.compile();markers=[]
    profile=ROOT/'configs/robots/x2_ultra-sole-augmentation.json'
    for item in json.loads(profile.read_text())['sole_boxes']:
        side=item['side'];g=original.geom(item['geom']).id;k=int(original.geom_dataid[g]);adr,count=original.mesh_vertadr[k],original.mesh_vertnum[k]
        if hashlib.sha256(original.mesh_vert[adr:adr+count].tobytes()).hexdigest()!=item['source_mesh_vertices_sha256']:raise ValueError('Sole marker profile differs from X2 mesh')
        mesh_rotation=Rotation.from_quat(original.geom_quat[g][[1,2,3,0]]).as_matrix()
        hull=ConvexHull(original.mesh_vert[adr:adr+count]@mesh_rotation.T+original.geom_pos[g])
        size=np.array(item['half_size_m']);rotation=Rotation.from_quat(np.array(item['quaternion_in_body_wxyz'])[[1,2,3,0]])
        local=[(f'corner_{n}',np.array([x,y,-1])*size) for n,(x,y) in enumerate([(-1,-1),(-1,1),(1,-1),(1,1)])]
        local += [('heel',np.array([-1,0,-1])*size),('toe',np.array([1,0,-1])*size)]
        for suffix,point in local:
            position=rotation.apply(point)+np.array(item['position_in_body_m'])
            if (position@hull.equations[:,:3].T+hull.equations[:,3]).max()>2e-8:raise ValueError('Tracking marker leaves original sole hull')
            name=f'{side}_omni_{suffix}';spec.body(side+'_ankle_roll_link').add_body(name=name,pos=position)
            markers.append({'name':name,'parent':side+'_ankle_roll_link','position':position.tolist()})
    hand_markers=[]
    if a.hand_centroid_markers:
        for side in ('left','right'):
            body=original.body(side+'_wrist_roll_link').id
            geoms=[g for g in range(original.ngeom) if int(original.geom_bodyid[g])==body and original.geom_contype[g]]
            if len(geoms)!=1 or int(original.geom_type[geoms[0]])!=int(mujoco.mjtGeom.mjGEOM_MESH):raise ValueError('Expected one verified X2 hand collision mesh')
            g=geoms[0];mesh=int(original.geom_dataid[g]);adr,count=original.mesh_vertadr[mesh],original.mesh_vertnum[mesh]
            vertices=original.mesh_vert[adr:adr+count]@Rotation.from_quat(original.geom_quat[g][[1,2,3,0]]).as_matrix().T+original.geom_pos[g]
            hull=ConvexHull(vertices);origin=vertices.mean(0);triangles=vertices[hull.simplices]
            volumes=abs(np.linalg.det(triangles-origin))/6
            center=(((triangles.sum(1)+origin)/4)*volumes[:,None]).sum(0)/volumes.sum()
            if (center@hull.equations[:,:3].T+hull.equations[:,3]).max()>1e-9:raise ValueError('Hand centroid outside collision hull')
            name=side+'_omni_hand_center';spec.body(side+'_wrist_roll_link').add_body(name=name,pos=center)
            item={'name':name,'parent':side+'_wrist_roll_link','position':center.tolist(),'source_geom':original.geom(g).name,
                'semantics':'Collision-hull volume centroid; not a measured palm pressure centre'}
            markers.append(item);hand_markers.append(item)
    updated=spec.compile()
    for field in ['nq','nv','njnt','nu','ngeom','nmesh']:
        if getattr(updated,field)!=getattr(original,field):raise ValueError(f'Original dimensions changed: {field}')
    for body in range(1,original.nbody):
        other=updated.body(original.body(body).name).id
        for field in ['body_mass','body_inertia','body_ipos','body_iquat']:
            np.testing.assert_allclose(getattr(updated,field)[other],getattr(original,field)[body],rtol=0,atol=1e-12)
    for field in ['jnt_range','jnt_axis','jnt_pos','dof_armature','geom_type','geom_size','geom_pos','geom_quat','mesh_vert']:
        np.testing.assert_allclose(getattr(updated,field),getattr(original,field),rtol=0,atol=1e-12)
    if [updated.joint(i).name for i in range(1,updated.njnt)]!=PROFILE['joint_names']:raise ValueError('Joint order changed')
    a.out.mkdir(parents=True);root=ET.fromstring(spec.to_xml())
    source=robot_xml_path();source_xml=ET.parse(source).getroot();compiler=source_xml.find('compiler')
    meshdir=source.parent/(compiler.get('meshdir','') if compiler is not None else '')
    for mesh in root.findall('asset/mesh'):
        name=Path(mesh.get('file',''))
        if not name.is_absolute():name=(meshdir/name).resolve()
        if not name.is_file():raise FileNotFoundError(name)
        mesh.set('file',str(name))
    ET.ElementTree(root).write(a.out/'x2_omni_markers.xml')
    restored=mujoco.MjModel.from_xml_path(str(a.out/'x2_omni_markers.xml'))
    np.testing.assert_array_equal(restored.jnt_range,updated.jnt_range)
    np.testing.assert_allclose(restored.body_mass,updated.body_mass,rtol=0,atol=1e-12)
    mapping={'Spine1':'pelvis','LeftUpLeg':'left_hip_pitch_link','LeftLeg':'left_knee_link','LeftToeBase':'left_omni_toe',
        'RightUpLeg':'right_hip_pitch_link','RightLeg':'right_knee_link','RightToeBase':'right_omni_toe','LeftArm':'left_shoulder_roll_link',
        'LeftForeArm':'left_elbow_link','LeftHandMiddle3':'left_wrist_roll_link','RightArm':'right_shoulder_roll_link',
        'RightForeArm':'right_elbow_link','RightHandMiddle3':'right_wrist_roll_link','LeftFoot':'left_omni_heel','RightFoot':'right_omni_heel'}
    if a.hand_centroid_markers:
        mapping['LeftHandMiddle3']='left_omni_hand_center';mapping['RightHandMiddle3']='right_omni_hand_center'
    for body in mapping.values():restored.body(body)
    report={'source_xml':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'robot':'x2_ultra','nq':updated.nq,'nv':updated.nv,
        'joint_names':PROFILE['joint_names'],'markers':markers,'hand_centroid_markers':hand_markers,'mocap_joint_mapping':mapping,'robot_physics_unchanged':True,
        'foot_sticking_links':[x['name'] for x in markers if 'corner' in x['name']],
        'root_joint_name':restored.joint(0).name,'scope':'Prepared marker/mapping model; retargeting core and X2 contact behavior remain unverified','training_approved':False}
    (a.out/'model.json').write_text(json.dumps(report,indent=2)+'\n');(a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in report.items() if k not in ['markers','joint_names','mocap_joint_mapping','foot_sticking_links']},indent=2))


if __name__=='__main__':main()
