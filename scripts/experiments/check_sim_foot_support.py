"""Compare simulator-only analytic foot support kinematics against MuJoCo.

Usage: check_sim_foot_support.py SCENE_XML SOURCE_ROOT
"""
import sys
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco
sys.path.insert(0, sys.argv[2])
from supermimic.deploy.sim_foot_support import SimFootSupport
model = mujoco.MjModel.from_xml_path(sys.argv[1]); data = mujoco.MjData(model)
fk = SimFootSupport(sys.argv[1].replace('scene.xml','x2.xml')); rng = np.random.default_rng(1243)
errors=[]
for _ in range(200):
    rot = Rotation.random(random_state=rng); omega = rng.uniform(-2,2,3)
    names=[x[0] for chain,_ in fk.chains.values() for x in chain]
    pos=dict(zip(names,rng.uniform(-.5,.5,len(names)))); vel=dict(zip(names,rng.uniform(-2,2,len(names))))
    data.qpos[:]=model.qpos0; data.qvel[:]=0; data.qpos[:7]=np.r_[[0,0,0],rot.as_quat()[[3,0,1,2]]]; data.qvel[3:6]=omega
    for name in names:
        jid=model.joint(name).id; data.qpos[model.jnt_qposadr[jid]]=pos[name]; data.qvel[model.jnt_dofadr[jid]]=vel[name]
    mujoco.mj_forward(model,data); candidates=fk.candidates(rot.as_matrix(),omega,pos,vel)
    for side in ('left','right'):
        body=model.body(side+'_ankle_roll_link').id
        gs=[i for i in range(model.ngeom) if model.geom_bodyid[i]==body and model.geom_type[i]==mujoco.mjtGeom.mjGEOM_SPHERE and model.geom_contype[i]]
        g=min(gs,key=lambda i:data.geom_xpos[i,2]-model.geom_size[i,0]); point=data.geom_xpos[g].copy();point[2]-=model.geom_size[g,0]
        jp=np.zeros((3,model.nv));jr=np.zeros_like(jp);mujoco.mj_jac(model,data,jp,jr,point,body)
        errors.append([abs(point[2]-candidates[side][0]),np.linalg.norm(-jp@data.qvel-candidates[side][1])])
print('max_height_and_velocity_errors',np.max(errors,axis=0))
assert np.max(errors)<1e-9
