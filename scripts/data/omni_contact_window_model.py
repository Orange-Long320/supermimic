"""Original-model contact geometry and dynamics for offline Omni refinement."""
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


class ContactWindow:
    def __init__(self,model,reference,scene,start,end,effort_fraction=.95):
        from supermimic.robots.dimensions import PROFILE
        self.m=m=mujoco.MjModel.from_binary_path(str(model));self.d=d=mujoco.MjData(m)
        if not 0<effort_fraction<=1:raise ValueError('Cannot exceed original motor ratings')
        self.effort_fraction=effort_fraction
        if (m.nq,m.nv,m.njnt)!=(38,37,32) or m.nu not in (0,31) or m.neq:raise ValueError('Original floating X2 only')
        self.names=[m.joint(i).name.rsplit('/',1)[-1] for i in range(1,m.njnt)]
        if self.names!=PROFILE['joint_names']:raise ValueError('Unexpected joint order')
        np.testing.assert_allclose(m.opt.gravity,[0,0,-9.81],rtol=0,atol=1e-12)
        np.testing.assert_allclose(m.dof_armature[6:],[PROFILE['armature'][n] for n in self.names],rtol=0,atol=1e-12)
        with np.load(reference) as z:
            q=z['qpos'];fps=float(np.asarray(z['fps']).ravel()[0]);stride=round(fps/50)
            if stride<1 or abs(fps/stride-50)>1e-9:raise ValueError('Exact50Hz subsampling required')
            self.source=q[::stride].copy()
        if not 2<=start<end<len(self.source)-2:raise ValueError('Two unchanged neighboring states required')
        self.start,self.end=start,end;self.flags=int(m.opt.disableflags)
        self.inverse_flags=self.flags|int(mujoco.mjtDisableBit.mjDSBL_CONTACT)|int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
        self.root=int(m.jnt_bodyid[0]);self.weight=float(m.body_mass.sum()*9.81)
        self.scales=np.r_[np.full(3,self.weight),np.full(3,self.weight*.7)]
        self.effort=np.array([PROFILE['effort_limit'][n] for n in self.names]);self.speed=np.array([PROFILE['velocity_limit'][n] for n in self.names])
        self.robot=[g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]]==self.root and (m.geom_contype[g] or m.geom_conaffinity[g])]
        self.terrain=[g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]]!=self.root and (m.geom_contype[g] or m.geom_conaffinity[g])]
        feet=[m.body(s+'_ankle_roll_link').id for s in ['left','right']];hands=[m.body(s+'_wrist_roll_link').id for s in ['left','right']]
        boxes=json.loads(Path(scene).read_text())['scene_boxes'];surfaces=[(np.zeros(3),np.array([np.inf,np.inf,0.]),np.eye(3))]
        surfaces += [(np.array(b['center']),np.array(b['half_size']),np.array(b['rotation_world'])) for b in boxes]
        parts=[]
        for g in self.robot:
            if m.geom_bodyid[g] not in feet+hands:continue
            if m.geom_type[g]!=int(mujoco.mjtGeom.mjGEOM_MESH):raise ValueError('Original support meshes required')
            k=m.geom_dataid[g];first,count=m.mesh_vertadr[k],m.mesh_vertnum[k]
            parts.append((g,m.mesh_vert[first:first+count].copy()))
        self.contacts=[];self.targets=[];directions=np.array([[1,1],[1,-1],[-1,1],[-1,-1]])
        for frame in range(start,end+1):
            self.geometry_state(self.source[frame]);contacts=[];active=set()
            for g,vertices in parts:
                body=int(m.geom_bodyid[g]);world=vertices@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
                for k,(c,h,r) in enumerate(surfaces):
                    local=(world-c)@r
                    faces=[(2,1)] if body in feet or k==0 else [(axis,sign) for axis in range(3) for sign in [-1,1] if (axis,sign)!=(2,-1)]
                    for axis,sign in faces:
                        tangent=[j for j in range(3) if j!=axis]
                        mask=(abs(local[:,axis]*sign-h[axis])<=.002)&(abs(local[:,tangent])<=h[tangent]).all(1);points=world[mask]
                        if not len(points):continue
                        picked=np.unique(np.argmax((points@r)[:,tangent]@directions.T,axis=0))
                        rays=np.array([sign*r[:,axis]+.8/np.sqrt(2)*(u*r[:,tangent[0]]+v*r[:,tangent[1]]) for u,v in directions]).T
                        for point in points[picked]:contacts.append((body,(point-d.xpos[body])@d.xmat[body].reshape(3,3),rays.copy()))
                        active.add(body)
            self.contacts.append(contacts)
            # Only load-bearing source bodies are fixed; an unloaded swing
            # foot remains free to adapt to X2's mass and limb proportions.
            self.targets.append([(b,d.xpos[b].copy(),Rotation.from_matrix(d.xmat[b].reshape(3,3))) for b in sorted(active)])
        self.jp=np.zeros((3,37));self.jr=np.zeros_like(self.jp)
        self.centre=m.jnt_range[1:].mean(1);self.radius=np.diff(m.jnt_range[1:],axis=1).ravel()*.45
        self.domain=np.array([.26 if any(k in n for k in ['shoulder','elbow','wrist','waist','head','neck']) else .5 for n in self.names])

    def geometry_state(self,pose):
        self.m.opt.disableflags=self.flags;self.d.qpos[:]=pose;self.d.qvel[:]=0;mujoco.mj_forward(self.m,self.d)

    def inverse(self,qs,frame):
        m,d=self.m,self.d;forward=np.zeros(37);backward=np.zeros(37)
        mujoco.mj_differentiatePos(m,forward,.02,qs[frame],qs[frame+1]);mujoco.mj_differentiatePos(m,backward,-.02,qs[frame],qs[frame-1])
        m.opt.disableflags=self.inverse_flags;d.qpos[:]=qs[frame];d.qvel[:]=(forward+backward)/2
        d.qacc[:]=(forward-backward)*50.;mujoco.mj_inverse(m,d)
        return d.qfrc_inverse.copy()

    def basis(self,points):
        m,d=self.m,self.d;columns=[]
        for body,local,rays in points:
            point=d.xpos[body]+d.xmat[body].reshape(3,3)@local;mujoco.mj_jac(m,d,self.jp,self.jr,point,body);columns.append(self.jp.T@rays)
        return np.concatenate(columns,axis=1) if columns else np.zeros((37,0))

    def correction(self,qs):
        values=np.zeros((len(qs),37))
        for i in range(self.start+1,self.end):mujoco.mj_differentiatePos(self.m,values[i],1.,self.source[i],qs[i])
        return values

    def measure(self,qs,coefficients):
        m,d=self.m,self.d;base=[];motor=[];speed=[];contact_pos=0.;contact_rot=0.;penetration=0.;self_pen=0.;witness=np.zeros(6)
        for i,frame in enumerate(range(self.start,self.end+1)):
            required=self.inverse(qs,frame);basis=self.basis(self.contacts[i]);residual=required-basis@(coefficients[i]*self.weight)
            base.append(residual[:6]/self.scales);motor.append(abs(residual[6:])/self.effort);speed.append(abs(d.qvel[6:])/self.speed)
            self.geometry_state(qs[frame])
            for g in self.robot:
                for other in self.terrain:penetration=max(penetration,-float(mujoco.mj_geomDistance(m,d,g,other,.02,witness)))
            for c in d.contact:
                if c.geom1 in self.robot and c.geom2 in self.robot:self_pen=max(self_pen,-float(c.dist))
            for body,p,r in self.targets[i]:
                contact_pos=max(contact_pos,float(np.linalg.norm(d.xpos[body]-p)))
                contact_rot=max(contact_rot,float(np.linalg.norm((r.inv()*Rotation.from_matrix(d.xmat[body].reshape(3,3))).as_rotvec())))
        base=np.asarray(base);motor=np.asarray(motor);correction=self.correction(qs)
        excess=np.maximum(motor-self.effort_fraction,0.)
        return dict(merit=float(np.square(base).sum()+10*np.square(excess).sum()),root_max_normalized=float(abs(base).max()),
            motor_max_ratio=float(motor.max()),speed_max_ratio=float(np.max(speed)),contact_position_error_m=contact_pos,
            contact_rotation_error_rad=contact_rot,environment_penetration_m=penetration,self_penetration_m=self_pen,
            root_translation_domain_max_m=float(abs(correction[:,:3]).max()),root_rotation_domain_max_rad=float(abs(correction[:,3:6]).max()),
            joint_domain_excess_rad=float(np.maximum(abs(correction[:,6:])-self.domain,0).max()),
            soft_joint_excess_rad=float(np.maximum(abs(qs[:,7:]-self.centre)-self.radius,0).max()))
