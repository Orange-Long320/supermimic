"""Compare official and training XML geometry at one recorded X2 joint pose.

This does not compile the mjlab training environment, change models, step
physics or establish which description matches hardware.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('official','training','snapshot','profile','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('New output required')
    sample=json.loads(a.snapshot.read_text());profile=json.loads(a.profile.read_text());names=profile['joint_names']
    if names!=sample['joint_names'] or len(names)!=31:raise ValueError('Joint order mismatch')
    models=[mujoco.MjModel.from_xml_path(str(path)) for path in (a.official,a.training)]; datas=[]
    for m in models:
        d=mujoco.MjData(m);d.qpos[:7]=[*sample['position'],*sample['quat_wxyz']]
        for name,q in zip(names,sample['joint_position']):
            j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name)
            if j<0 or not np.isfinite(q):raise ValueError('Invalid joint '+name)
            d.qpos[m.jnt_qposadr[j]]=q
        mujoco.mj_forward(m,d);datas.append(d)
    rows={};arms={}
    for name in names:
        joint_ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name) for m in models]
        body_ids=[int(m.jnt_bodyid[j]) for m,j in zip(models,joint_ids)]
        body_names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,b) for m,b in zip(models,body_ids)]
        if body_names[0]!=body_names[1]:raise ValueError('Joint child names differ: '+name)
        p0,p1=[d.xpos[b].copy() for d,b in zip(datas,body_ids)]
        r0,r1=[d.xmat[b].reshape(3,3).copy() for d,b in zip(datas,body_ids)]
        rows[name]={'body':body_names[0],'position_difference_m':float(np.linalg.norm(p0-p1)),
                    'orientation_difference_rad':float(Rotation.from_matrix(r0.T@r1).magnitude()),
                    'official_mass_kg':float(models[0].body_mass[body_ids[0]]),
                    'training_xml_mass_kg':float(models[1].body_mass[body_ids[1]])}
        arms[name]={'official_xml':float(models[0].dof_armature[models[0].jnt_dofadr[joint_ids[0]]]),
                    'training_profile':float(profile['armature'][name])}
    com=[];feet=[]
    for m,d in zip(models,datas):
        b=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'pelvis');com.append(d.subtree_com[b].tolist());sides={}
        for side in ('left','right'):
            b=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,side+'_ankle_roll_link')
            ids=[i for i in range(m.ngeom) if m.geom_bodyid[i]==b and m.geom_contype[i]]
            sides[side]=[{'name':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i),
                          'type':int(m.geom_type[i]),'contype':int(m.geom_contype[i]),
                          'conaffinity':int(m.geom_conaffinity[i])} for i in ids]
        feet.append(sides)
    result={'scope':__doc__,'sha256':{k:sha(getattr(a,k)) for k in ('official','training','snapshot','profile')},
            'total_mass_kg':[float(m.body_mass.sum()) for m in models],
            'pelvis_subtree_com_world_m':com,'com_difference_m':float(np.linalg.norm(np.asarray(com[0])-com[1])),
            'foot_collision_geometry':feet,'links':rows,'armature':arms,'hardware_ready':False}
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    worst=sorted(((v['position_difference_m'],k) for k,v in rows.items()),reverse=True)[:5]
    print(json.dumps({'total_mass_kg':result['total_mass_kg'],'com_difference_m':result['com_difference_m'],
                      'largest_link_position_differences_m':worst,'foot_collision_geometry':feet},indent=2))


if __name__=='__main__':main()
