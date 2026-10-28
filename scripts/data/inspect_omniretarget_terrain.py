"""Server-side FK screening of pinned G1 terrain data; not an X2 feasibility test."""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--preferred-height',type=float,default=.45)
    p.add_argument('--sample-hz',type=float,default=10.)
    a=p.parse_args()
    if a.out.exists():p.error('output exists')
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation
    from scipy.spatial import ConvexHull
    raw=a.source.resolve();a.out.mkdir(parents=True)
    (a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    config=raw/'holosoma-sources/src/holosoma_retargeting/holosoma_retargeting/config_types/data_conversion.py'
    names=None
    for node in ast.parse(config.read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_ROBOT_JOINT_NAMES_DEFAULT' for t in node.targets):
            names=ast.literal_eval(node.value)['g1']
    if names is None:raise ValueError('No verified upstream G1 joint order')
    robot=ET.parse(raw/'models/g1/g1_29dof_spherehand.urdf')
    compiler=robot.getroot().find('mujoco/compiler');compiler.set('fusestatic','false');compiler.set('discardvisual','false')
    for mesh in robot.findall('.//mesh'):
        mesh.set('filename',str((raw/'models/g1'/mesh.attrib['filename']).resolve()))
    compiler.set('meshdir','/')
    ET.SubElement(robot.getroot(),'link',name='world')
    joint=ET.SubElement(robot.getroot(),'joint',name='source_free_root',type='floating')
    ET.SubElement(joint,'parent',link='world');ET.SubElement(joint,'child',link='pelvis')
    robot.write(a.out/'source-g1.urdf')
    spec=mujoco.MjSpec.from_file(str(a.out/'source-g1.urdf'));model=spec.compile();data=mujoco.MjData(model)
    if (model.nq,model.nv)!=(36,35):raise ValueError('Unexpected G1 dimensions')
    columns=[int(model.jnt_qposadr[model.joint(name).id]) for name in names]
    if sorted(columns)!=list(range(7,36)):raise ValueError('Upstream/model joint names differ')
    mujoco.mj_saveModel(model,str(a.out/'source-g1.mjb'))
    hands=[model.body(s+'_sphere_hand_link').id for s in ('left','right')]
    feet=[]
    for side in ('left','right'):
        ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,f'{side}_ankle_roll_sphere_{i}_link') for i in range(1,5)]
        if min(ids)<0:raise ValueError('Source foot markers missing')
        feet.append(ids)

    def transform(origin):
        result=np.eye(4)
        if origin is not None:
            result[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
            result[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
        return result

    def boxes(path):
        root=ET.parse(path).getroot();poses={'world':np.eye(4)};pending=list(root.findall('joint'))
        while pending:
            count=len(pending)
            for joint in pending[:]:
                parent=joint.find('parent').get('link');child=joint.find('child').get('link')
                if joint.get('type')!='fixed':raise ValueError('Expected static terrain')
                if parent in poses:poses[child]=poses[parent]@transform(joint.find('origin'));pending.remove(joint)
            if len(pending)==count:raise ValueError('Unresolved terrain frame')
        result=[]
        for link in root.findall('link'):
            for collision in link.findall('collision'):
                mesh=collision.find('geometry/mesh')
                if mesh is None:raise ValueError('Unsupported source terrain geometry')
                obj=(path.parent/mesh.get('filename')).resolve()
                if not obj.is_relative_to(raw):raise ValueError('Terrain path escapes source')
                vertices=np.array([[float(v) for v in row.split()[1:4]] for row in obj.read_text().splitlines() if row.startswith('v ')])
                vertices*=np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
                pose=poses[link.get('name')]@transform(collision.find('origin'))
                vertices=vertices@pose[:3,:3].T+pose[:3,3]
                footprint=vertices[ConvexHull(vertices[:,:2]).vertices,:2]
                if len(footprint)!=4:raise ValueError('Expected rectangular source terrain')
                axis=footprint[1]-footprint[0];axis/=np.linalg.norm(axis)
                rotation=np.eye(3);rotation[:2,0]=axis;rotation[:2,1]=[-axis[1],axis[0]]
                local=vertices@rotation;lo,hi=local.min(0),local.max(0)
                if np.minimum(abs(local-lo),abs(local-hi)).max()>1e-6:raise ValueError('Source mesh is not a horizontal rectangular prism')
                result.append({'name':link.get('name'),'low':vertices.min(0).tolist(),'high':vertices.max(0).tolist(),
                    'center':(rotation@((lo+hi)/2)).tolist(),'half_size':((hi-lo)/2).tolist(),'rotation_world':rotation.tolist()})
        return result

    rows=[];candidates=[]
    with zipfile.ZipFile(raw/'robot-terrain.zip') as archive:
        for number,entry in enumerate(sorted(x for x in archive.namelist() if x.endswith('.npz'))):
            content=archive.read(entry)
            with np.load(io.BytesIO(content),allow_pickle=False) as z:q=z['qpos'].copy();fps=float(np.asarray(z['fps']).reshape(-1)[0])
            if q.ndim!=2 or q.shape[1]!=36 or fps<=0 or not np.isfinite(q).all():raise ValueError(f'Invalid trajectory {entry}')
            quaternion_error=float(abs(np.linalg.norm(q[:,:4],axis=1)-1).max())
            if quaternion_error>1e-4:raise ValueError(f'Non-unit source quaternion: {entry}')
            stem=Path(entry).stem;scene=stem[:8];suffix=stem[8:]
            terrain=raw/'models/terrain'/scene/('multi_boxes'+suffix+'.urdf');scene_boxes=boxes(terrain)
            stride=max(1,round(fps/a.sample_hz));frames=np.arange(0,len(q),stride);positions=[];hand_positions=[]
            for frame in frames:
                data.qpos[:3]=q[frame,4:7];data.qpos[3:7]=q[frame,:4];data.qpos[columns]=q[frame,7:]
                mujoco.mj_kinematics(model,data)
                positions.append([data.xpos[ids].mean(0) for ids in feet]);hand_positions.append(data.xpos[hands].copy())
            foot=np.asarray(positions);hand=np.asarray(hand_positions)
            for box in scene_boxes:
                low,high=np.array(box['low']),np.array(box['high']);height=float(high[2])
                if height<.12:continue
                center,half,rotation=np.array(box['center']),np.array(box['half_size']),np.array(box['rotation_world'])
                local_feet=(foot-center)@rotation;local_hands=(hand-center)@rotation
                on=(abs(local_feet[:,:,:2])<=half[:2]+.02).all(2)&(abs(local_feet[:,:,2]-half[2])<.04)
                both=on.all(1);runs=[];start=None
                for k,flag in enumerate(np.r_[both,False]):
                    if flag and start is None:start=k
                    elif not flag and start is not None:runs.append((k-start,start,k));start=None
                if not runs:continue
                length,begin,end=max(runs)
                if length*stride/fps<.3 or begin==0:continue
                first=begin
                while first>0 and on[first-1].any():first-=1
                begin_window=max(0,first-round(.6*fps/stride));end_window=min(len(frames),end+1)
                distance=np.linalg.norm(np.maximum(abs(local_hands)-half,0),axis=2)
                possible_hands=bool((distance[begin_window:end_window]<.08).any())
                ground=bool((abs(foot[begin_window,:,2])<.06).all())
                candidates.append({'motion':entry,'box':box,'height_m':height,'dimensions_m':(half*2).tolist(),
                    'start_frame':int(frames[begin_window]),'end_frame':int(frames[min(end_window-1,len(frames)-1)]),
                    'both_feet_near_top_seconds':length*stride/fps,'possible_hand_contact':possible_hands,'starts_near_ground':ground,
                    'height_error_m':abs(height-a.preferred_height),'fps':fps})
            rows.append({'motion':entry,'sha256':hashlib.sha256(content).hexdigest(),'frames':len(q),'fps':fps,'quaternion_error':quaternion_error,'boxes':scene_boxes})
            if number%25==0:print(f'screened={number+1}',flush=True)
    candidates.sort(key=lambda c:(c['possible_hand_contact'],not c['starts_near_ground'],c['height_error_m']))
    report={'motions':rows,'candidate_windows':candidates,'preferred_height_m':a.preferred_height,'height_is_hard_constraint':False,
        'joint_order_source':str(config),'limitations':'G1 FK and 10Hz proximity screening only. Hand proximity is not measured support; foot markers are not force contacts. Not X2 data, not trained policy, not a physical feasibility certificate.'}
    (a.out/'screening.json').write_text(json.dumps(report,indent=2)+'\n')
    if candidates:
        best=candidates[0];filename=Path(best['motion']).name
        with zipfile.ZipFile(raw/'robot-terrain.zip') as archive:(a.out/filename).write_bytes(archive.read(best['motion']))
        (a.out/'selected.json').write_text(json.dumps(best,indent=2)+'\n')
    print(json.dumps({'motion_count':len(rows),'candidate_count':len(candidates),'shortlist':candidates[:5]},indent=2))


if __name__=='__main__':main()
