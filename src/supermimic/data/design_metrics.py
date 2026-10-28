"""Batched physical-state metrics for motion design; no changes to physics."""
import numpy as np
import torch
import mujoco
from scipy.spatial import ConvexHull

from supermimic.kinematics.rotations import quat_apply, quat_apply_inverse, quat_mul


class BatchedDesignMetrics:
    """COM, momentum and exact collision-hull floor heights from sim data."""
    def __init__(self,model,device):
        self.device=torch.device(device)
        self.root=int(model.jnt_bodyid[0])
        ids=np.flatnonzero(model.body_rootid==self.root)
        self.bodies=torch.tensor(ids,device=device)
        f=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32,device=device)
        self.mass=f(model.body_mass[ids])
        self.ipos=f(model.body_ipos[ids])
        self.iquat=f(model.body_iquat[ids])
        self.inertia=f(model.body_inertia[ids])
        vertices=[];geom_ids=[];kind=[]
        for g in range(model.ngeom):
            if model.body_rootid[int(model.geom_bodyid[g])]!=self.root or not (model.geom_contype[g] or model.geom_conaffinity[g]):continue
            k=model.geom_dataid[g]
            if int(model.geom_type[g])==int(mujoco.mjtGeom.mjGEOM_BOX):
                points=np.array([[x,y,z] for x in [-1.,1.] for y in [-1.,1.] for z in [-1.,1.]])*model.geom_size[g]
            else:
                if k<0:raise ValueError('Unsupported collision geometry in design metrics')
                a,n=model.mesh_vertadr[k],model.mesh_vertnum[k]
                points=model.mesh_vert[a:a+n].astype(float)
                points=points[ConvexHull(points).vertices]
            side=0
            if "foot_" in model.geom(g).name:
                side=1 if "left" in model.body(int(model.geom_bodyid[g])).name else 2
            vertices.extend(points);geom_ids.extend([g]*len(points));kind.extend([side]*len(points))
        self.vertices=f(vertices)
        self.geom_ids=torch.tensor(geom_ids,device=device)
        kind=np.array(kind)
        self.groups=[torch.tensor(np.flatnonzero(kind==k),device=device) for k in [1,2,0]]
        self.ngeom=model.ngeom

    def __call__(self,data):
        pos=data.xpos[:,self.bodies]
        quat=data.xquat[:,self.bodies]
        xcom=pos+quat_apply(quat,self.ipos.expand_as(pos))
        com=(xcom*self.mass[None,:,None]).sum(dim=1)/self.mass.sum()
        spatial=data.cvel[:,self.bodies]
        omega=spatial[...,:3]
        velocity=spatial[...,3:]+torch.cross(omega,xcom-com[:,None,:],dim=-1)
        com_velocity=(velocity*self.mass[None,:,None]).sum(dim=1)/self.mass.sum()
        inertial_quat=quat_mul(quat,self.iquat.expand_as(quat))
        intrinsic=quat_apply(inertial_quat,quat_apply_inverse(inertial_quat,omega)*self.inertia)
        momentum=(intrinsic+torch.cross(xcom-com[:,None,:],velocity*self.mass[None,:,None],dim=-1)).sum(dim=1)
        gpos=data.geom_xpos[:]
        matrix=data.geom_xmat[:].reshape(gpos.shape[0],self.ngeom,3,3)
        height=(matrix[:,self.geom_ids,2,:]*self.vertices[None,:,:]).sum(dim=-1)+gpos[:,self.geom_ids,2]
        left,right,nonfoot=[height[:,i].amin(dim=1) for i in self.groups]
        root_quat=data.xquat[:,self.root]
        axis=torch.zeros((len(root_quat),3),device=self.device);axis[:,2]=1
        up=quat_apply(root_quat,axis)
        speed=torch.linalg.vector_norm(com_velocity,dim=-1)
        angular=torch.linalg.vector_norm(data.qvel[:,3:6],dim=-1)
        return torch.cat([com[:,2:3],com_velocity[:,2:3],momentum,speed[:,None],angular[:,None],
            left[:,None],right[:,None],nonfoot[:,None],torch.atan2(up[:,0],up[:,2])[:,None],up[:,2:3]],dim=1)
