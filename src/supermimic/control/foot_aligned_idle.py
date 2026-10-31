"""Geometric foot-registered Idle planning, not contact-force certification."""
from dataclasses import dataclass
import time

import mujoco
import numpy as np
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def smooth(u):
    u = np.clip(u, 0., 1.)
    return u*u*u*(10.+u*(-15.+6.*u))


def polygon_margin(points, xy):
    if len(points) < 3 or np.linalg.matrix_rank(np.asarray(points)-np.mean(points,axis=0)) < 2:
        return float('-inf')
    hull = ConvexHull(np.asarray(points))
    return float(np.min(-(hull.equations[:,:2]@xy+hull.equations[:,2])))


def project_inside(points, xy, margin=.01):
    hull = ConvexHull(np.asarray(points))
    x = np.array(xy, dtype=float)
    for _ in range(100):
        violation = hull.equations[:,:2]@x+hull.equations[:,2]+margin
        if violation.max() <= 1e-8:
            return x
        i = int(violation.argmax())
        a = hull.equations[i,:2]
        x -= violation[i]*a/np.dot(a,a)
    raise ValueError('No interior COM point in measured foot geometry')


@dataclass
class FootPlan:
    features: np.ndarray
    root_positions: np.ndarray
    yaw: np.ndarray
    context: dict
    report: dict


class FootAlignedIdle:
    def __init__(self, scene, spec, dt=.02):
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.names = tuple(spec['joint_names'])
        self.n = len(self.names)
        self.default = np.array([spec['default_joint_pos'][k] for k in self.names])
        self.velocity_limits = np.array([spec['velocity_limit'][k] for k in self.names])
        joints = [self.model.joint(k).id for k in self.names]
        self.qadr = np.array([self.model.jnt_qposadr[i] for i in joints])
        self.vadr = np.array([self.model.jnt_dofadr[i] for i in joints])
        self.limits = self.model.jnt_range[joints].copy()
        self.leg = np.array([i for i,k in enumerate(self.names) if any(s in k for s in ('hip_', 'knee_', 'ankle_'))])
        if self.n != 31 or len(self.leg) != 12 or self.model.nq != 38:
            raise ValueError('Foot planner requires the original 31-joint X2 model')
        self.dofs = np.r_[0,1,2,3,4,5,self.vadr[self.leg]]
        self.root = self.model.body('pelvis').id
        self.feet = [self.model.body(s+'_ankle_roll_link').id for s in ('left','right')]
        self.floor = self.model.geom('floor').id
        self.geoms = [[i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] == body
                       and self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE and self.model.geom_contype[i]] for body in self.feet]
        if any(not g for g in self.geoms):
            raise ValueError('Expected original official foot collision spheres')
        self.dt = dt
        self.observation_data = mujoco.MjData(self.model)

    def _fk(self, d, root, quat, joints):
        d.qpos[:7] = np.r_[root,quat]
        d.qpos[self.qadr] = joints
        mujoco.mj_kinematics(self.model,d)
        mujoco.mj_comPos(self.model,d)

    def observe(self, position, quat, joints, joint_velocity, root_velocity, angular_velocity):
        d = self.observation_data
        self._fk(d,position,quat,joints)
        velocity = np.zeros(self.model.nv)
        velocity[:3], velocity[3:6] = root_velocity, angular_velocity
        velocity[self.vadr] = joint_velocity
        foot_velocity = []
        for body in self.feet:
            jac = np.zeros((3,self.model.nv))
            mujoco.mj_jacBody(self.model,d,jac,None,body)
            foot_velocity.append(jac@velocity)
        jac = np.zeros((3,self.model.nv))
        mujoco.mj_jacSubtreeCom(self.model,d,jac,self.root)
        gap = [float(np.min(d.geom_xpos[g,2]-self.model.geom_size[g,0])) for g in self.geoms]
        return dict(position=np.array(position), quat=np.array(quat), joints=np.array(joints),
                    joint_velocity=np.array(joint_velocity), root_velocity=np.array(root_velocity),
                    angular_velocity=np.array(angular_velocity), feet=d.xpos[self.feet].copy(),
                    foot_rotation=d.xmat[self.feet].reshape(2,3,3).copy(),
                    foot_velocity=np.array(foot_velocity), gap=np.array(gap),
                    com=d.subtree_com[self.root].copy(), com_velocity=jac@velocity)

    @staticmethod
    def ready(context):
        return bool(np.max(np.abs(context['gap'])) <= .012
                    and np.linalg.norm(context['root_velocity'][:2]) <= .05
                    and np.linalg.norm(context['angular_velocity']) <= .15
                    and np.sqrt(np.mean(context['joint_velocity']**2)) <= .15
                    and np.max(np.linalg.norm(context['foot_velocity'],axis=1)) <= .06)

    def flat_foot(self, side, xy, yaw):
        rotation = Rotation.from_euler('z',yaw).as_matrix()
        z = -float(np.min((self.model.geom_pos[self.geoms[side]]@rotation.T)[:,2]
                          -self.model.geom_size[self.geoms[side],0]))
        return np.r_[xy,z], rotation

    def points(self, positions, rotations, sides=(0,1)):
        points = []
        for side in sides:
            g = self.geoms[side]
            world = positions[side]+self.model.geom_pos[g]@rotations[side].T
            bottom = world[:,2]-self.model.geom_size[g,0]
            points.extend(world[bottom <= bottom.min()+.002,:2])
        return np.asarray(points)

    def _state(self, d, x, yaw, upper):
        joints = upper.copy()
        joints[self.leg] = x[6:]
        quat = Rotation.from_euler('xyz',[x[3],x[4],x[5]]).as_quat()[[3,0,1,2]]
        self._fk(d,x[:3],quat,joints)
        return joints

    def solve(self, d, initial, yaw, upper, foot_pos, foot_rot, com_xy, height):
        x = initial.copy()
        for _ in range(35):
            self._state(d,x,yaw,upper)
            residual, matrices = [], []
            mapping = np.zeros((self.model.nv,len(x)))
            mapping[:3,:3] = np.eye(3)
            mapping[3:6,3:6] = Rotation.from_euler('xyz',x[3:6]).as_matrix().T@np.column_stack((Rotation.from_euler('xyz',[0,x[4],x[5]]).apply([1,0,0]),Rotation.from_euler('z',x[5]).apply([0,1,0]),[0,0,1]))
            mapping[self.vadr[self.leg],np.arange(6,18)] = 1
            pe, re = [], []
            for side,body in enumerate(self.feet):
                jacp, jacr = np.zeros((3,self.model.nv)), np.zeros((3,self.model.nv))
                mujoco.mj_jacBody(self.model,d,jacp,jacr,body)
                p = foot_pos[side]-d.xpos[body]
                r = Rotation.from_matrix(foot_rot[side]@d.xmat[body].reshape(3,3).T).as_rotvec()
                pe.append(np.linalg.norm(p)); re.append(np.linalg.norm(r))
                residual.extend((30*p,2*r)); matrices.extend((30*jacp@mapping,2*jacr@mapping))
            jac = np.zeros((3,self.model.nv))
            mujoco.mj_jacSubtreeCom(self.model,d,jac,self.root)
            ce = com_xy-d.subtree_com[self.root,:2]
            residual.extend((10*ce,np.array([5*(height-x[2])]),.02*(upper[self.leg]-x[6:])))
            height_jac = np.zeros((1,len(x))); height_jac[0,2] = 5
            regularizer = np.zeros((12,len(x))); regularizer[:,6:] = .02*np.eye(12)
            matrices.extend((10*jac[:2]@mapping,height_jac,regularizer))
            if max(pe)<.00015 and max(re)<.0015 and np.linalg.norm(ce)<.0003 and abs(height-x[2])<.001:
                return x
            j, e = np.vstack(matrices), np.concatenate(residual)
            delta = np.linalg.solve(j.T@j+1e-5*np.eye(len(x)),j.T@e)
            delta[:3] = np.clip(delta[:3],-.015,.015)
            delta[3:] = np.clip(delta[3:],-.08,.08)
            x += delta
            x[2] = np.clip(x[2],.55,.72)
            x[3:5] = np.clip(x[3:5],-.35,.35)
            x[6:] = np.clip(x[6:],self.limits[self.leg,0]+1e-6,self.limits[self.leg,1]-1e-6)
        self._state(d,x,yaw,upper)
        pe = np.linalg.norm(d.xpos[self.feet]-foot_pos,axis=1)
        re = [np.linalg.norm(Rotation.from_matrix(foot_rot[s]@d.xmat[b].reshape(3,3).T).as_rotvec()) for s,b in enumerate(self.feet)]
        if pe.max()>.001 or max(re)>.01 or np.linalg.norm(d.subtree_com[self.root,:2]-com_xy)>.002:
            raise ValueError('Foot/COM IK did not converge: '+str(dict(foot_error=pe.tolist(),rotation_error=re,
                com_error=(d.subtree_com[self.root,:2]-com_xy).tolist(),height=float(x[2]),height_target=float(height),
                legs=x[6:].tolist(),target_feet=foot_pos.tolist())))
        return x

    def build(self, source, context):
        started = time.monotonic()
        failures = []
        for lowering in (0., .02, .04):
            for support in (0,1):
                try:
                    plan = self._build(source,context,support,float(context['position'][2])-lowering)
                    plan.report.update(height_lowering_m=lowering,rejected_candidates=failures,
                                       total_planning_seconds=time.monotonic()-started)
                    return plan
                except ValueError as exc:
                    failures.append(dict(support=support,lowering_m=lowering,error=str(exc)))
        raise ValueError('No reachable alignment candidate: '+str(failures))

    def register_only(self, source, context):
        """Cheap causal control; no IK or foot-reachability claim is made."""
        started = time.monotonic()
        source = np.asarray(source,dtype=float)
        gravity = source[2*self.n:2*self.n+3]
        tilt = Rotation.from_euler('xyz',[np.arctan2(-gravity[1],-gravity[2]),
                                          np.arcsin(np.clip(gravity[0],-1,1)),0.])
        data = mujoco.MjData(self.model)
        self._fk(data,[0,0,source[-1]],tilt.as_quat()[[3,0,1,2]],source[:self.n]+self.default)
        support = 0  # Match the left-foot registration of the compared rollout.
        old_yaw = Rotation.from_matrix(data.xmat[self.feet[support]].reshape(3,3)).as_euler('xyz')[2]
        new_yaw = Rotation.from_matrix(context['foot_rotation'][support]).as_euler('xyz')[2]
        yaw = float(new_yaw-old_yaw)
        xy = context['feet'][support,:2]-Rotation.from_euler('z',yaw).apply(data.xpos[self.feet[support]])[:2]
        position = np.r_[xy,source[-1]]
        report = dict(diagnostic='registration-only; original stand; no endpoint alignment claim',
                      support_side='left',measured_foot_targets=context['feet'].tolist(),
                      total_planning_seconds=time.monotonic()-started)
        return FootPlan(np.repeat(source[None],2,axis=0),np.repeat(position[None],2,axis=0),
                        np.repeat(yaw,2),context,report)

    def _build(self, source, context, support, height):
        begin = time.monotonic()
        n = self.n
        source = np.asarray(source,dtype=float)
        if np.max(np.abs(source[n:2*n]))>1e-5 or np.max(np.abs(source[2*n+3:2*n+9]))>1e-5:
            raise ValueError('Foot alignment starts only after the committed static brake')
        upper = source[:n]+self.default
        gravity = source[2*n:2*n+3]
        base = Rotation.from_euler('xyz',[np.arctan2(-gravity[1],-gravity[2]),np.arcsin(np.clip(gravity[0],-1,1)),0.])
        d = mujoco.MjData(self.model)
        self._fk(d,[0,0,source[-1]],base.as_quat()[[3,0,1,2]],upper)
        original_feet = d.xpos[self.feet].copy()
        original_rot = d.xmat[self.feet].reshape(2,3,3).copy()
        original_gap = [np.min(d.geom_xpos[g,2]-self.model.geom_size[g,0]) for g in self.geoms]
        # Both geometric support choices are audited; this is not a force label.
        swing = 1-support
        measured_yaw = [Rotation.from_matrix(r).as_euler('xyz')[2] for r in context['foot_rotation']]
        source_yaw = Rotation.from_matrix(original_rot[support]).as_euler('xyz')[2]
        yaw = float(measured_yaw[support]-source_yaw)
        rz = Rotation.from_euler('z',yaw)
        quat = (rz*base).as_quat()[[3,0,1,2]]
        root_xy = context['feet'][support,:2]-rz.apply(original_feet[support])[:2]
        x0 = np.r_[root_xy,source[-1],base.as_euler('xyz')[:2],yaw,upper[self.leg]]
        self._state(d,x0,yaw,upper)
        start_feet = d.xpos[self.feet].copy()
        start_rot = d.xmat[self.feet].reshape(2,3,3).copy()
        start_com = d.subtree_com[self.root,:2].copy()
        final_pos, final_rot, projected_pos, projected_rot = [], [], [], []
        for side in (0,1):
            p,r = self.flat_foot(side,context['feet'][side,:2],measured_yaw[side])
            final_pos.append(p);final_rot.append(r)
            sy = Rotation.from_matrix(start_rot[side]).as_euler('xyz')[2]
            p,r = self.flat_foot(side,start_feet[side,:2],sy)
            projected_pos.append(p);projected_rot.append(r)
        final_pos, final_rot = np.asarray(final_pos),np.asarray(final_rot)
        projected_pos, projected_rot = np.asarray(projected_pos),np.asarray(projected_rot)
        omega = np.sqrt(9.81/max(.3,float(context['com'][2])))
        capture = context['com'][:2]+context['com_velocity'][:2]/omega
        terminal_com = project_inside(self.points(final_pos,final_rot),capture)
        support_com = np.mean(self.points(projected_pos,projected_rot,(support,)),axis=0)
        projected = self.solve(d,x0,yaw,upper,projected_pos,projected_rot,start_com,height)
        attempts = []
        for scale in (1.,1.25,1.5,1.75):
            phase = np.cumsum(np.ceil(np.array([.2,.8,.8,.8])*scale/self.dt)*self.dt)
            duration = float(phase[-1])
            knots = np.unique(np.r_[np.arange(0,duration,.04),phase,duration])
            xs = []
            x = x0.copy()
            for t in knots:
                if t <= phase[0]:
                    x = x0+smooth(t/phase[0])*(projected-x0)
                else:
                    pos, rot = projected_pos.copy(),projected_rot.copy()
                    if t <= phase[1]:
                        com = start_com+smooth((t-phase[0])/(phase[1]-phase[0]))*(support_com-start_com)
                    elif t <= phase[2]:
                        u = (t-phase[1])/(phase[2]-phase[1]); alpha = smooth(u)
                        pos[swing] += alpha*(final_pos[swing]-projected_pos[swing])
                        pos[swing,2] += .03*16*u*u*(1-u)*(1-u)
                        r0 = Rotation.from_matrix(projected_rot[swing]); r1 = Rotation.from_matrix(final_rot[swing])
                        rot[swing] = (r0*Rotation.from_rotvec(alpha*(r0.inv()*r1).as_rotvec())).as_matrix()
                        com = support_com
                    else:
                        pos,rot = final_pos,final_rot
                        com = support_com+smooth((t-phase[2])/(phase[3]-phase[2]))*(terminal_com-support_com)
                    x = self.solve(d,x,yaw,upper,pos,rot,com,height)
                xs.append(x.copy())
            xs = np.asarray(xs)
            slopes = PchipInterpolator(knots,xs).derivative()(knots)
            slopes[0] = slopes[-1] = 0
            spline = CubicHermiteSpline(knots,xs,slopes)
            grid = np.linspace(0,duration,round(duration/self.dt)+1)
            xx,vv = spline(grid),spline(grid,1)
            q = np.repeat(upper[None],len(grid),axis=0); q[:,self.leg] = xx[:,6:]
            qv = np.zeros_like(q);qv[:,self.leg] = vv[:,6:]
            root_v = Rotation.from_euler('z',xx[:,5,None]).inv().apply(vv[:,:3])
            rotations = Rotation.from_euler('xyz',xx[:,3:6])
            gravities = rotations.inv().apply(np.tile([0,0,-1.],(len(grid),1)))
            angular = np.c_[vv[:,3]-np.sin(xx[:,4])*vv[:,5],
                              np.cos(xx[:,3])*vv[:,4]+np.sin(xx[:,3])*np.cos(xx[:,4])*vv[:,5],
                              -np.sin(xx[:,3])*vv[:,4]+np.cos(xx[:,3])*np.cos(xx[:,4])*vv[:,5]]
            features = np.c_[q-self.default,qv,gravities,angular,root_v,xx[:,2]]
            features[0] = source
            features[-1,n:2*n] = 0.; features[-1,2*n+3:2*n+9] = 0.
            report = self.audit(q,xx[:,:3],rotations.as_quat()[:,[3,0,1,2]],grid,phase,support,final_pos,final_rot,terminal_com,features)
            attempts.append(report)
            if report['passed']:
                report.update(planning_seconds=time.monotonic()-begin, support_side=('left','right')[support],
                              yaw=yaw, capture_point=capture.tolist(), terminal_com=terminal_com.tolist(),
                              measured_foot_targets=final_pos.tolist(), attempts=attempts[:-1],
                              scope='Bounded leg/root IK and geometric support proxy; no measured contact-force or dynamics acceptance')
                return FootPlan(features,xx[:,:3],xx[:,5],context,report)
        raise ValueError('No admissible foot-aligned trajectory: '+str(attempts))

    def audit(self,q,p,quat,t,phase,support,final_pos,final_rot,com_target,features):
        d = mujoco.MjData(self.model)
        gap,coms,points,terminal_error = [],[],[],None
        nonfoot_penetration = 0.
        collision_pair = None
        initial_penetration = 0.
        for i,(joint,pos) in enumerate(zip(q,p)):
            self._fk(d,pos,quat[i],joint)
            coms.append(d.subtree_com[self.root].copy())
            gap.append([np.min(d.geom_xpos[g,2]-self.model.geom_size[g,0]) for g in self.geoms])
            sides = (support,) if phase[1] < t[i] < phase[2] else (0,1)
            rotations = d.xmat[self.feet].reshape(2,3,3)
            points.append(self.points(d.xpos[self.feet],rotations,sides))
            mujoco.mj_collision(self.model,d)
            for c in d.contact[:d.ncon]:
                gs = [int(c.geom[0]),int(c.geom[1])]
                bodies = [int(self.model.geom_bodyid[g]) for g in gs]
                if self.floor not in gs or not any(b in self.feet for b in bodies):
                    if i == 0:
                        initial_penetration = max(initial_penetration,-float(c.dist))
                    if -float(c.dist) > nonfoot_penetration:
                        nonfoot_penetration = -float(c.dist)
                        collision_pair = [self.model.body(b).name for b in bodies]
        coms=np.asarray(coms);gap=np.asarray(gap)
        acceleration=np.gradient(np.gradient(coms[:,:2],self.dt,axis=0),self.dt,axis=0)
        zmp=coms[:,:2]-coms[:,2,None]/9.81*acceleration
        margins=[polygon_margin(points[i],zmp[i]) for i in range(len(t)) if t[i] > phase[0]+.08 and t[i]<phase[-1]-.08]
        excess = np.maximum(np.maximum(self.limits[:,0]-q,q-self.limits[:,1]),0)
        hard=float(excess.max())
        worst_frame,worst_joint = np.unravel_index(excess.argmax(),excess.shape)
        speed=float(max(np.max(abs(np.diff(q,axis=0)/self.dt)/self.velocity_limits),
                        np.max(abs(features[:,self.n:2*self.n])/self.velocity_limits)))
        terminal_error=float(np.max(np.linalg.norm(d.xpos[self.feet]-final_pos,axis=1)))
        terminal_rotation=max(np.linalg.norm(Rotation.from_matrix(final_rot[s]@d.xmat[b].reshape(3,3).T).as_rotvec()) for s,b in enumerate(self.feet))
        final_margin=polygon_margin(self.points(d.xpos[self.feet],d.xmat[self.feet].reshape(2,3,3)),d.subtree_com[self.root,:2])
        report=dict(duration_s=float(t[-1]),hard_excess_rad=hard,hard_excess_joint=self.names[worst_joint],hard_excess_frame=int(worst_frame),joint_speed_ratio=speed,
                    min_foot_gap_m=float(gap.min()),nonfoot_penetration_m=nonfoot_penetration,initial_nonfoot_penetration_m=initial_penetration,collision_pair=collision_pair,
                    terminal_foot_error_m=terminal_error,terminal_foot_rotation_error_rad=float(terminal_rotation),
                    terminal_com_error_m=float(np.linalg.norm(coms[-1,:2]-com_target)),
                    terminal_support_margin_m=final_margin,min_zmp_proxy_margin_m=float(min(margins)))
        inherited = float(np.maximum(np.maximum(self.limits[:,0]-q[0],q[0]-self.limits[:,1]),0).max())
        leg_hard = float(excess[:,self.leg].max())
        report.update(inherited_source_excess_rad=inherited,new_excess_rad=max(0.,hard-inherited),leg_excess_rad=leg_hard)
        report['passed']=bool(hard<=max(1e-4,inherited+1e-8) and leg_hard<=1e-4 and speed<=1 and gap.min()>=-.003 and nonfoot_penetration<=.002
                              and terminal_error<=.001 and terminal_rotation<=.01
                              and report['terminal_com_error_m']<=.002 and final_margin>=.009
                              and report['min_zmp_proxy_margin_m']>=-.002)
        return report
