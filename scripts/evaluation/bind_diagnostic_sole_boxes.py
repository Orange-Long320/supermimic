"""Create explicit isolated sole-box scene copies; never change training assets."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['attribution','profile','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('New output required')
    import mujoco
    import numpy as np
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation
    a.out.mkdir(parents=True);(a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    rows=json.loads(a.attribution.read_text());profile=json.loads(a.profile.read_text());changes=[]
    protected=['body_mass','body_inertia','body_ipos','body_iquat','jnt_range','dof_armature','dof_damping',
               'actuator_gainprm','actuator_biasprm','actuator_forcerange','geom_friction','geom_solref','geom_solimp','geom_contype','geom_conaffinity']
    for name,row in rows.items():
        source=Path(row['model']);m=mujoco.MjModel.from_binary_path(str(source));before={k:getattr(m,k).copy() for k in protected}
        for item in profile['sole_boxes']:
            g=m.geom('robot/'+item['geom']).id;mesh=m.geom_dataid[g];start,n=m.mesh_vertadr[mesh],m.mesh_vertnum[mesh]
            vertices=m.mesh_vert[start:start+n]
            if hashlib.sha256(vertices.tobytes()).hexdigest()!=item['source_mesh_vertices_sha256']:raise ValueError('Mesh binding mismatch')
            v=vertices@Rotation.from_quat(m.geom_quat[g][[1,2,3,0]]).as_matrix().T+m.geom_pos[g];h=ConvexHull(v)
            size=np.asarray(item['half_size_m']);corners=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])*size
            corners=Rotation.from_quat(np.array(item['quaternion_in_body_wxyz'])[[1,2,3,0]]).apply(corners)+item['position_in_body_m']
            if (corners@h.equations[:,:3].T+h.equations[:,3]).max()>2e-8:raise ValueError('Diagnostic box leaves original hull')
            m.geom_pos[g]=item['position_in_body_m'];m.geom_quat[g]=item['quaternion_in_body_wxyz'];m.geom_type[g]=int(mujoco.mjtGeom.mjGEOM_BOX);m.geom_dataid[g]=-1
            m.geom_size[g]=size;m.geom_sameframe[g]=0;m.geom_rbound[g]=np.linalg.norm(size);m.geom_aabb[g]=[0,0,0,*size]
        for key,value in before.items():np.testing.assert_array_equal(getattr(m,key),value)
        target=a.out/f'{name}.mjb';mujoco.mj_saveModel(m,str(target));row['model']=str(target)
        changes.append({'task':name,'original_model':str(source),'original_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'new_sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    (a.out/'attribution.json').write_text(json.dumps(rows,indent=2)+'\n')
    (a.out/'provenance.json').write_text(json.dumps({'physical_geometry_changed':True,'unchanged_arrays':protected,'models':changes,'approved_for_training':False},indent=2)+'\n')


if __name__=='__main__':main()
