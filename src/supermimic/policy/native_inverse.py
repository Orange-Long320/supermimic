"""Whole-body command design using MuJoCo's actual compliant contact law."""
import mujoco
import numpy as np
from scipy.optimize import Bounds,minimize
from scipy.spatial.transform import Rotation
from supermimic.policy.whole_body import SingleSupportController


class NativeContactController(SingleSupportController):
    """Enforce unactuated-base dynamics and motor limits through mj_inverse."""
    def solve(self,qpos,qvel,com_target=None,com_velocity=None,com_acceleration=None,momentum_target=None,qref=None,
              joint_velocity=None,joint_acceleration=None,momentum_rate=None):
        m=self.model;d=self.data;d.qpos[:]=qpos;d.qvel[:]=qvel;mujoco.mj_forward(m,d)
        foot,com,angular=self.jacobians(d)
        epsilon=1e-6;f=self.future;f.qpos[:]=qpos
        mujoco.mj_integratePos(m,f.qpos,qvel,epsilon);f.qvel[:]=qvel;mujoco.mj_forward(m,f)
        following=self.jacobians(f)
        drift=[(next_j-current)@qvel/epsilon for current,next_j in zip((foot,com,angular),following)]
        target=self.com_ref if com_target is None else com_target
        target_v=np.zeros(3) if com_velocity is None else com_velocity
        target_a=np.zeros(3) if com_acceleration is None else com_acceleration
        desired_com=target_a+getattr(self,'com_kp',100)*(target-d.subtree_com[self.root])+getattr(self,'com_kd',20)*(target_v-com@qvel)-drift[1]
        desired_momentum=(np.zeros(3) if momentum_rate is None else momentum_rate)+10*((np.zeros(3) if momentum_target is None else momentum_target)-angular@qvel)-drift[2]
        posture=self.qref if qref is None else qref
        reference_v=np.zeros(m.nv-6) if joint_velocity is None else joint_velocity
        reference_a=np.zeros(m.nv-6) if joint_acceleration is None else joint_acceleration
        desired_joint=reference_a+60*(posture[7:]-qpos[7:])+15*(reference_v-qvel[6:])
        acc=20.;task=np.r_[com*acc*2.,angular*acc/10.,np.eye(m.nv)[6:]*acc/100.]
        target_task=np.r_[desired_com*2.,desired_momentum/10.,desired_joint/100.]
        grounded=bool(len(self.contact_points(d)))
        if grounded:
            if not getattr(self,'grounded',True):
                self.foot_position=d.xpos[self.foot].copy()
                self.foot_rotation=Rotation.from_matrix(d.xmat[self.foot].reshape(3,3))
            rotation=Rotation.from_matrix(d.xmat[self.foot].reshape(3,3))
            error=np.r_[self.foot_position-d.xpos[self.foot],(self.foot_rotation*rotation.inv()).as_rotvec()]
            desired=getattr(self,'foot_acceleration_ref',np.zeros(6))+400*error+40*(getattr(self,'foot_velocity_ref',np.zeros(6))-foot@qvel)-drift[0]
            # Tangent/orientation stabilization is soft. The normal reaction is
            # determined by the simulator's compliant contact inverse, not a
            # fictitious rigid vertical acceleration constraint.
            rows=[0,1,3,4,5]
            task=np.r_[task,foot[rows]*acc/20.];target_task=np.r_[target_task,desired[rows]/20.]
        self.grounded=grounded
        hessian=task.T@task+np.eye(m.nv)*1e-6;linear=task.T@target_task
        horizon=.08;lower,upper=m.jnt_range[1:].T
        minimum=np.maximum(2*(lower+self.margin-qpos[7:]-horizon*qvel[6:])/horizon**2,
                           (-self.velocity_limits-qvel[6:])/horizon)
        maximum=np.minimum(2*(upper-self.margin-qpos[7:]-horizon*qvel[6:])/horizon**2,
                           (self.velocity_limits-qvel[6:])/horizon)
        if np.any(minimum>maximum):raise RuntimeError('Joint predictive bounds already infeasible')
        lb=np.r_[np.full(6,-100.),minimum/acc];ub=np.r_[np.full(6,100.),maximum/acc]
        scale=np.r_[np.full(3,m.body_mass.sum()*9.81),np.full(3,m.body_mass.sum()*9.81*.7),self.effort]
        cache={'x':None,'jac':None,'force':None}
        def inverse(x,jacobian=False):
            if cache['x'] is None or not np.array_equal(x,cache['x']):
                d.qacc[:]=x*acc;mujoco.mj_inverse(m,d)
                cache.update(x=x.copy(),force=d.qfrc_inverse.copy()/scale,jac=None)
            if jacobian and cache['jac'] is None:
                derivative=np.zeros((m.nv,m.nv));d.qacc[:]=x*acc
                mujoco.mjd_inverseFD(m,d,1e-5,False,None,None,derivative,None,None,None,None)
                cache['jac']=derivative.T*acc/scale[:,None]
                d.qacc[:]=x*acc;mujoco.mj_inverse(m,d)
            return cache['jac'] if jacobian else cache['force']
        constraints=[{'type':'eq','fun':lambda x:inverse(x)[:6],'jac':lambda x:inverse(x,True)[:6]},
            {'type':'ineq','fun':lambda x:np.r_[1-inverse(x)[6:],1+inverse(x)[6:]],
             'jac':lambda x:np.r_[-inverse(x,True)[6:],inverse(x,True)[6:]]}]
        initial=np.zeros(m.nv) if self.previous is None else self.previous
        initial=np.clip(initial,lb,ub)
        result=minimize(lambda x:.5*x@hessian@x-linear@x,initial,jac=lambda x:hessian@x-linear,
            method='SLSQP',bounds=Bounds(lb,ub),constraints=constraints,options={'maxiter':80,'ftol':1e-9})
        forces=inverse(result.x);eq_error=float(abs(forces[:6]).max());motor=float(abs(forces[6:]).max())
        bound_error=float(np.maximum(np.maximum(lb-result.x,result.x-ub),0).max())
        feasible=(np.isfinite(forces).all() and np.isfinite(result.x).all() and eq_error<=1e-5 and motor<=1.00001 and bound_error<=1e-7)
        if not feasible or (not result.success and int(result.status) not in (8,9)):
            self.failure={'solver_message':str(result.message),'root_force_residual_normalized':eq_error,'motor_ratio':motor}
            self.failure_arrays={'qpos':qpos,'qvel':qvel,'qacc':result.x*acc,'normalized_inverse_force':forces}
            raise RuntimeError(f'Native inverse rejected: {result.message}; root={eq_error}; motor={motor}')
        self.previous=result.x.copy()
        return forces[6:]*self.effort,{'equality_residual':eq_error,'motor_ratio':motor,'iterations':int(result.nit),
            'solver_converged':bool(result.success),'solver_status':int(result.status),'bound_residual':bound_error,
            'com_acceleration_error':float(np.linalg.norm(com@(result.x*acc)-desired_com)),
            'ground_force_world':d.qfrc_constraint[:3].tolist()}
