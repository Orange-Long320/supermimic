"""Bounded inverse dynamics for single-support feasibility experiments."""
import mujoco
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,minimize,linprog
from scipy.spatial.transform import Rotation


class SingleSupportController:
    def __init__(self,model,qpos,effort,velocity_limits,foot_body,margin=.02):
        self.model=model;self.data=mujoco.MjData(model);self.future=mujoco.MjData(model)
        self.root=int(model.jnt_bodyid[0]);self.foot=foot_body
        self.qref=qpos.copy();self.effort=effort;self.velocity_limits=velocity_limits;self.margin=margin
        d=self.data;d.qpos[:]=qpos;mujoco.mj_forward(model,d)
        self.com_ref=d.subtree_com[self.root].copy()
        self.foot_position=d.xpos[self.foot].copy()
        self.foot_rotation=Rotation.from_matrix(d.xmat[self.foot].reshape(3,3))
        points=self.contact_points(d)
        if not len(points):raise ValueError('Single-support control requires actual initial ground contacts')
        self.rays=np.array([[x/np.sqrt(2),y/np.sqrt(2),1.] for x,y in [(1,1),(1,-1),(-1,1),(-1,-1)]]).T
        self.force_scale=float(model.body_mass.sum()*9.81/(4*len(points)))
        self.acceleration_scale=20.;self.previous=None

    def contact_points(self,data):
        points=[];m=self.model
        for contact in data.contact[:data.ncon]:
            bodies=[int(m.geom_bodyid[int(g)]) for g in contact.geom]
            if self.foot not in bodies or contact.dist>0:continue
            other=bodies[1] if bodies[0]==self.foot else bodies[0]
            if int(m.body_rootid[other])!=self.root:points.append(contact.pos.copy())
        return np.asarray(points)

    def jacobians(self,data):
        m=self.model;jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);com=np.zeros_like(jp);momentum=np.zeros_like(jp)
        mujoco.mj_jacBody(m,data,jp,jr,self.foot)
        mujoco.mj_jacSubtreeCom(m,data,com,self.root);mujoco.mj_angmomMat(m,data,momentum,self.root)
        return np.r_[jp,jr],com,momentum

    def solve(self,qpos,qvel,com_target=None,com_velocity=None,com_acceleration=None,momentum_target=None,qref=None):
        m=self.model;d=self.data;d.qpos[:]=qpos;d.qvel[:]=qvel;mujoco.mj_forward(m,d)
        foot,com,angular=self.jacobians(d)
        epsilon=1e-6;f=self.future;f.qpos[:]=qpos
        mujoco.mj_integratePos(m,f.qpos,qvel,epsilon);f.qvel[:]=qvel;mujoco.mj_forward(m,f)
        following=self.jacobians(f)
        drift=[(next_j-current)@qvel/epsilon for current,next_j in zip((foot,com,angular),following)]
        mass=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,mass)
        basis=[];jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
        points=self.contact_points(d)
        if not len(points):raise RuntimeError('Support contact lost; inverse-dynamics hold rejected')
        for point in points:
            mujoco.mj_jac(m,d,jp,jr,point,self.foot);basis.append(jp.T@self.rays)
        basis=np.concatenate(basis,axis=1);nf=basis.shape[1];n=m.nv+nf
        self.force_scale=float(m.body_mass.sum()*9.81/nf)
        acc=self.acceleration_scale;force=self.force_scale
        dynamics=np.c_[mass*acc,-basis*force]
        torque=dynamics[6:]/self.effort[:,None];bias=d.qfrc_bias[6:]/self.effort
        rotation=Rotation.from_matrix(d.xmat[self.foot].reshape(3,3))
        pose_error=np.r_[self.foot_position-d.xpos[self.foot],(self.foot_rotation*rotation.inv()).as_rotvec()]
        foot_acceleration=30.**2*pose_error-60.*(foot@qvel)-drift[0]
        base_scale=np.r_[np.full(3,m.body_mass.sum()*9.81),np.full(3,m.body_mass.sum()*9.81*.7)]
        equality=np.r_[dynamics[:6]/base_scale[:,None],np.c_[foot*acc,np.zeros((6,nf))]/40.]
        rhs=np.r_[-d.qfrc_bias[:6]/base_scale,foot_acceleration/40.]
        target=self.com_ref if com_target is None else com_target
        target_v=np.zeros(3) if com_velocity is None else com_velocity
        target_a=np.zeros(3) if com_acceleration is None else com_acceleration
        desired_com=target_a+100*(target-d.subtree_com[self.root])+20*(target_v-com@qvel)-drift[1]
        desired_momentum=10*((np.zeros(3) if momentum_target is None else momentum_target)-angular@qvel)-drift[2]
        posture=self.qref if qref is None else qref
        desired_joint=60*(posture[7:]-qpos[7:])-15*qvel[6:]
        task=np.r_[np.c_[com*acc,np.zeros((3,nf))]*2.,
            np.c_[angular*acc,np.zeros((3,nf))]/10.,
            np.c_[np.eye(m.nv)[6:]*acc,np.zeros((m.nv-6,nf))]/100.]
        target_task=np.r_[desired_com*2.,desired_momentum/10.,desired_joint/100.]
        hessian=task.T@task+np.diag(np.r_[np.full(m.nv,1e-5),np.full(nf,1e-4)])
        linear=task.T@target_task
        horizon=.08;lower,upper=m.jnt_range[1:].T
        minimum=np.maximum(2*(lower+self.margin-qpos[7:]-horizon*qvel[6:])/horizon**2,
                           (-self.velocity_limits-qvel[6:])/horizon)
        maximum=np.minimum(2*(upper-self.margin-qpos[7:]-horizon*qvel[6:])/horizon**2,
                           (self.velocity_limits-qvel[6:])/horizon)
        if np.any(minimum>maximum):raise RuntimeError('Joint predictive bounds already infeasible')
        lb=np.r_[np.full(6,-100.),minimum/acc,np.zeros(nf)]
        ub=np.r_[np.full(6,100.),maximum/acc,np.full(nf,np.inf)]
        initial=np.zeros(n) if self.previous is None or len(self.previous)!=n else self.previous.copy()
        initial=np.clip(initial,lb,ub)
        result=minimize(lambda x:.5*x@hessian@x-linear@x,initial,jac=lambda x:hessian@x-linear,
            method='SLSQP',bounds=Bounds(lb,ub),constraints=[LinearConstraint(equality,rhs,rhs),
            LinearConstraint(torque,-1-bias,1-bias)],options={'maxiter':100,'ftol':1e-8})
        eq_error=float(abs(equality@result.x-rhs).max())
        tau=dynamics[6:]@result.x+d.qfrc_bias[6:]
        motor_ratio=float((abs(tau)/self.effort).max())
        if not result.success or eq_error>1e-5 or motor_ratio>1.00001:
            feasible=linprog(np.zeros(n),A_eq=equality,b_eq=rhs,
                A_ub=np.r_[torque,-torque],b_ub=np.r_[1-bias,1+bias],
                bounds=list(zip(lb,ub)),method='highs')
            self.failure={'solver_message':str(result.message),'linear_feasible':bool(feasible.success),
                'linear_status':int(feasible.status),'linear_message':str(feasible.message)}
            self.failure_arrays={'qpos':qpos,'qvel':qvel,'equality':equality,'rhs':rhs,'torque':torque,
                'bias':bias,'lower':lb,'upper':ub,'hessian':hessian,'linear':linear,'solver_x':result.x}
            raise RuntimeError(f'Inverse dynamics rejected: {result.message}; equality={eq_error}; motor={motor_ratio}')
        self.previous=result.x
        return tau,{'equality_residual':eq_error,'motor_ratio':motor_ratio,'iterations':int(result.nit),
            'com_acceleration_error':float(np.linalg.norm(com@(result.x[:m.nv]*acc)+drift[1]-(desired_com+drift[1]))),
            'ground_force_world':(self.rays@result.x[m.nv:].reshape(-1,4).sum(0)*force).tolist()}
