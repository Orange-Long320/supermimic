"""Explicit, opt-in conservative X2 sole collision profile."""
import hashlib
import json
import os
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def apply_sole_box_profile(spec, path=None):
    path=path or os.environ.get('SM_SOLE_BOX_CONFIG')
    if not path:
        return spec
    from supermimic.robots.paths import robot_name
    if robot_name()!='x2_ultra':raise ValueError('The sole-box profile is specific to X2 Ultra')
    config=json.loads(Path(path).read_text())
    if config.get('robot')!='x2_ultra' or len(config.get('sole_boxes',[]))!=2:
        raise ValueError('Invalid X2 sole-box configuration')
    if len({item['geom'] for item in config['sole_boxes']})!=2 or {item['side'] for item in config['sole_boxes']}!={'left','right'}:
        raise ValueError('Expected distinct left and right foot overrides')
    mode=config.get('mode','replace')
    if mode not in ('replace','augment'):raise ValueError('Unknown sole profile mode')
    original=spec.compile()
    plans=[]
    for item in config['sole_boxes']:
        name=item['geom'];geom=spec.geom(name);g=original.geom(name).id
        if not name.startswith('foot_') or int(original.geom_type[g])!=int(mujoco.mjtGeom.mjGEOM_MESH):
            raise ValueError('Expected the original foot collision mesh')
        if original.body(int(original.geom_bodyid[g])).name!=item['side']+'_ankle_roll_link':
            raise ValueError('Sole profile targets the wrong foot body')
        k=original.geom_dataid[g];a,n=original.mesh_vertadr[k],original.mesh_vertnum[k]
        digest=hashlib.sha256(original.mesh_vert[a:a+n].tobytes()).hexdigest()
        if digest!=item['source_mesh_vertices_sha256']:raise ValueError('Sole profile does not match the original foot mesh')
        size=np.asarray(item['half_size_m'],dtype=float)
        if size.shape!=(3,) or not np.isfinite(size).all() or np.any(size<=0):raise ValueError('Invalid sole-box dimensions')
        position=np.asarray(item['position_in_body_m'],dtype=float)
        quat=np.asarray(item['quaternion_in_body_wxyz'],dtype=float)
        if position.shape!=(3,) or quat.shape!=(4,) or not np.isfinite(position).all() or not np.isfinite(quat).all() or abs(np.linalg.norm(quat)-1)>1e-6:
            raise ValueError('Invalid sole-box transform')
        source_rotation=Rotation.from_quat(original.geom_quat[g][[1,2,3,0]]).as_matrix()
        vertices=original.mesh_vert[a:a+n]@source_rotation.T+original.geom_pos[g]
        hull=ConvexHull(vertices)
        corners=np.array([[x,y,z] for x in [-1.,1.] for y in [-1.,1.] for z in [-1.,1.]])*size
        corners=Rotation.from_quat(quat[[1,2,3,0]]).apply(corners)+position
        if np.max(corners@hull.equations[:,:3].T+hull.equations[:,3])>2e-8:
            raise ValueError('Sole box extends outside the original foot convex hull')
        proxy_name=name.removesuffix('_collision')+'_sole_collision'
        if mode=='augment' and mujoco.mj_name2id(original,mujoco.mjtObj.mjOBJ_GEOM,proxy_name)>=0:
            raise ValueError('Sole contact augmentation already exists')
        plans.append((geom,size,position,quat,item['side'],proxy_name))
    for geom,size,position,quat,side,proxy_name in plans:
        if mode=='augment':
            parent=spec.body(side+'_ankle_roll_link')
            proxy=parent.add_geom(name=proxy_name,type=mujoco.mjtGeom.mjGEOM_BOX,
                size=size,pos=position,quat=quat,mass=0.)
            for field in ['contype','conaffinity','condim','friction','solref','solimp','priority','margin','gap']:
                value=getattr(geom,field)
                try:getattr(proxy,field)[:]=value
                except TypeError:setattr(proxy,field,value)
        else:
            geom.type=mujoco.mjtGeom.mjGEOM_BOX;geom.meshname=''
            geom.size[:]=size;geom.pos[:]=position;geom.quat[:]=quat
    updated=spec.compile()
    for field in ['body_mass','body_inertia','body_ipos','body_iquat','jnt_range']:
        np.testing.assert_allclose(getattr(updated,field),getattr(original,field),rtol=0,atol=1e-12,
                                   err_msg=f'Sole profile changed {field}')
    return spec
