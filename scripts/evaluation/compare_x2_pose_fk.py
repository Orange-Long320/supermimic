"""Compare URDF and official MuJoCo link poses at recorded joint positions.

Read-only model diagnostic; this does not validate physical encoder zeroes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('scene','urdf','snapshot','out'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists(): raise ValueError('Output must be new')
    sample=json.loads(a.snapshot.read_text()); names=sample['joint_names']; values=sample['joint_position']
    if len(names)!=31 or len(set(names))!=31 or len(values)!=31 or not np.isfinite(values).all():
        raise ValueError('Invalid named joint sample')
    q=dict(zip(names,values)); joints=ET.parse(a.urdf).getroot().findall('joint')
    parents={j.find('parent').get('link') for j in joints}; children={j.find('child').get('link') for j in joints}
    roots=parents-children
    if len(roots)!=1: raise ValueError('URDF root is ambiguous')
    poses={roots.pop():(np.zeros(3),np.eye(3))}; pending=list(joints); movable={}
    while pending:
        remain=[]
        for j in pending:
            parent=j.find('parent').get('link'); child=j.find('child').get('link')
            if parent not in poses: remain.append(j); continue
            o=j.find('origin'); xyz=np.fromstring(o.get('xyz','0 0 0'),sep=' ') if o is not None else np.zeros(3)
            rpy=np.fromstring(o.get('rpy','0 0 0'),sep=' ') if o is not None else np.zeros(3)
            pos,rot=poses[parent]; pos=pos+rot@xyz; rot=rot@Rotation.from_euler('xyz',rpy).as_matrix()
            if j.get('type') in ('revolute','continuous'):
                name=j.get('name'); axis_node=j.find('axis')
                axis=np.fromstring(axis_node.get('xyz','1 0 0'),sep=' ') if axis_node is not None else np.array([1.,0.,0.])
                if name not in q or np.linalg.norm(axis)<1e-9: raise ValueError('Missing joint or axis: '+name)
                rot=rot@Rotation.from_rotvec(axis/np.linalg.norm(axis)*q[name]).as_matrix(); movable[name]=child
            elif j.get('type')!='fixed': raise ValueError('Unsupported joint type')
            poses[child]=(pos,rot)
        if len(remain)==len(pending): raise ValueError('Disconnected URDF')
        pending=remain
    m=mujoco.MjModel.from_xml_path(str(a.scene)); d=mujoco.MjData(m); d.qpos[:7]=[0,0,0,1,0,0,0]
    for name,val in q.items():
        jid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name)
        if jid<0: raise ValueError('Missing model joint: '+name)
        d.qpos[m.jnt_qposadr[jid]]=val
    mujoco.mj_forward(m,d); base_p,base_r=poses['pelvis']; report={}
    for name in names:
        if name not in movable: raise ValueError('Missing URDF joint: '+name)
        child=movable[name]; bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,child)
        if bid<0: raise ValueError('Missing model body: '+child)
        pos,rot=poses[child]; pos=base_r.T@(pos-base_p); rot=base_r.T@rot
        sim_rot=d.xmat[bid].reshape(3,3); delta=Rotation.from_matrix(rot.T@sim_rot).magnitude()
        report[name]={'child':child,'position_error_m':float(np.linalg.norm(d.xpos[bid]-pos)),
                      'rotation_error_rad':float(delta)}
    result={'scope':__doc__,'sha256':{k:hashlib.sha256(getattr(a,k).read_bytes()).hexdigest() for k in ('scene','urdf','snapshot')},
            'links':report,'max_position_error_m':max(x['position_error_m'] for x in report.values()),
            'max_rotation_error_rad':max(x['rotation_error_rad'] for x in report.values()),
            'encoder_zero_validated':False,'hardware_ready':False}
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('max_position_error_m','max_rotation_error_rad','encoder_zero_validated')},indent=2))


if __name__=='__main__': main()
