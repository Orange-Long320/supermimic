"""Whole-body command design using MuJoCo's actual compliant contact law."""
import mujoco
import numpy as np
from scipy.optimize import Bounds,minimize
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
from supermimic.policy.whole_body import SingleSupportController
from supermimic.policy.joint_prediction import position_acceleration_bounds,replanned_acceleration_bounds


def implicitfast_acceleration_map(model,data):
    """Map desired integrated acceleration to the instantaneous inverse input.

    For the checked joint-position servos, implicitfast solves
    (M + dt * damping) * delta_v/dt = M * qacc. Native force caps remain
    unchanged, and the optimizer keeps every motor strictly below saturation.
    Reject other force models instead of silently applying this restricted map.
    """
    m=model
    if int(m.opt.integrator)!=int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST):
        raise ValueError('Integration-aware inverse currently requires implicitfast')
    if (m.ntendon or m.opt.density or m.opt.viscosity or np.any(m.dof_dampingpoly) or
        m.nu!=m.nv-6 or np.any(m.actuator_ctrllimited) or np.any(m.actuator_dyntype) or np.any(m.actuator_gaintype) or
        np.any(m.actuator_biastype!=int(mujoco.mjtBias.mjBIAS_AFFINE)) or
        np.any(m.actuator_trntype!=int(mujoco.mjtTrn.mjTRN_JOINT)) or
        not np.array_equal(m.jnt_dofadr[m.actuator_trnid[:,0]],np.arange(6,m.nv)) or
        not np.allclose(m.actuator_gear[:,0],1.) or np.any(m.actuator_gainprm[:,1:])):
        raise ValueError('Unsupported dynamics in integration-aware inverse')
    if m.opt.disableflags & (int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)|int(mujoco.mjtDisableBit.mjDSBL_DAMPER)):
        raise ValueError('Integration-aware inverse requires enabled actuation and damping')
    mass=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,data,mass)
    damping=m.dof_damping.copy();damping[6:]-=m.actuator_biasprm[:,2]
    if np.any(damping<0):raise ValueError('Expected dissipative native motor damping')
    return np.eye(m.nv)+float(m.opt.timestep)*np.linalg.solve(mass,np.diag(damping))


def support_point_height_kinematics(data,body,local_points,foot_jacobian,foot_drift,qvel):
    """World height, acceleration Jacobian and drift for body-fixed points."""
    r=np.asarray(local_points)@data.xmat[body].reshape(3,3).T
    height=data.xpos[body,2]+r[:,2]
    jac=foot_jacobian[2]+r[:,1,None]*foot_jacobian[3]-r[:,0,None]*foot_jacobian[4]
    omega=foot_jacobian[3:]@qvel
    drift=foot_drift[2]+r[:,1]*foot_drift[3]-r[:,0]*foot_drift[4]+np.cross(omega,np.cross(omega,r))[:,2]
    return height,jac,drift


class NativeContactController(SingleSupportController):
    """Enforce unactuated-base dynamics and motor limits through mj_inverse."""
    def solve(self,qpos,qvel,com_target=None,com_velocity=None,com_acceleration=None,momentum_target=None,qref=None,
              joint_velocity=None,joint_acceleration=None,momentum_rate=None):
        m=self.model;d=self.data;d.qpos[:]=qpos;d.qvel[:]=qvel
        if hasattr(self,'servo_seed'):d.ctrl[:]=self.servo_seed
        mujoco.mj_forward(m,d)
        integration_map=implicitfast_acceleration_map(m,d) if getattr(self,'integration_aware',False) else np.eye(m.nv)
        forward_seed=np.linalg.solve(integration_map,d.qacc.copy())/20. if hasattr(self,'servo_seed') else None
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
        joint_weight=np.broadcast_to(getattr(self,'joint_task_weight',.01),(m.nv-6,))
        com_weight=np.broadcast_to(getattr(self,'com_task_weight',2.),(3,))
        momentum_weight=np.broadcast_to(getattr(self,'momentum_task_weight',.1),(3,))
        acc=20.;task=np.r_[com*acc*com_weight[:,None],angular*acc*momentum_weight[:,None],np.eye(m.nv)[6:]*acc*joint_weight[:,None]]
        target_task=np.r_[desired_com*com_weight,desired_momentum*momentum_weight,desired_joint*joint_weight]
        if hasattr(self,'root_acceleration_ref'):
            weight=float(getattr(self,'root_acceleration_weight',1.))
            task=np.r_[task,np.eye(m.nv)[3:6]*acc*weight]
            target_task=np.r_[target_task,np.asarray(self.root_acceleration_ref)*weight]
        grounded=bool(len(self.contact_points(d)))
        if grounded and getattr(self,'stabilize_support',True):
            if not getattr(self,'grounded',True):
                self.foot_position=d.xpos[self.foot].copy()
                self.foot_rotation=Rotation.from_matrix(d.xmat[self.foot].reshape(3,3))
            rotation=Rotation.from_matrix(d.xmat[self.foot].reshape(3,3))
            error=np.r_[self.foot_position-d.xpos[self.foot],(self.foot_rotation*rotation.inv()).as_rotvec()]
            desired=getattr(self,'foot_acceleration_ref',np.zeros(6))+400*error+40*(getattr(self,'foot_velocity_ref',np.zeros(6))-foot@qvel)-drift[0]
            # Tangent/orientation stabilization is soft. The normal reaction is
            # determined by the simulator's compliant contact inverse, not a
            # fictitious rigid vertical acceleration constraint.
            # A planned planted foot may need a soft normal task to recover a
            # briefly lost contact. Its reaction remains determined by native
            # compliant inverse dynamics, never by a rigid normal constraint.
            rows=list(range(6)) if getattr(self,'stabilize_support_normal',False) else [0,1,3,4,5]
            foot_weight=float(getattr(self,'foot_task_weight',.05))
            task=np.r_[task,foot[rows]*acc*foot_weight];target_task=np.r_[target_task,desired[rows]*foot_weight]
            for body,position_ref,rotation_ref in getattr(self,'additional_support_feet',[]):
                position_jac=np.zeros((3,m.nv));rotation_jac=np.zeros((3,m.nv))
                future_position_jac=np.zeros((3,m.nv));future_rotation_jac=np.zeros((3,m.nv))
                mujoco.mj_jacBody(m,d,position_jac,rotation_jac,body)
                mujoco.mj_jacBody(m,f,future_position_jac,future_rotation_jac,body)
                support_jac=np.r_[position_jac,rotation_jac]
                support_drift=(np.r_[future_position_jac,future_rotation_jac]-support_jac)@qvel/epsilon
                current_rotation=Rotation.from_matrix(d.xmat[body].reshape(3,3))
                error=np.r_[position_ref-d.xpos[body],(rotation_ref*current_rotation.inv()).as_rotvec()]
                desired=400*error-40*support_jac@qvel-support_drift
                task=np.r_[task,support_jac[rows]*acc*foot_weight]
                target_task=np.r_[target_task,desired[rows]*foot_weight]
        self.grounded=grounded
        hessian=task.T@task+np.eye(m.nv)*1e-6;linear=task.T@target_task
        objective_scale=max(1.,float(np.linalg.norm(hessian,ord=np.inf)))
        hessian=hessian/objective_scale;linear=linear/objective_scale
        lower,upper=m.jnt_range[1:].T
        minimum,maximum,horizon=replanned_acceleration_bounds(qpos[7:],qvel[6:],lower+self.margin,
            upper-self.margin,self.velocity_limits,float(m.opt.timestep))
        lb=np.r_[np.full(6,-100.),minimum/acc];ub=np.r_[np.full(6,100.),maximum/acc]
        scale=np.r_[np.full(3,m.body_mass.sum()*9.81),np.full(3,m.body_mass.sum()*9.81*.7),self.effort]
        cache={'x':None,'jac':None,'force':None}
        def inverse(x,jacobian=False):
            if cache['x'] is None or not np.array_equal(x,cache['x']):
                d.qacc[:]=integration_map@(x*acc);mujoco.mj_inverse(m,d)
                cache.update(x=x.copy(),force=d.qfrc_inverse.copy()/scale,jac=None)
            if jacobian and cache['jac'] is None:
                derivative=np.zeros((m.nv,m.nv));d.qacc[:]=integration_map@(x*acc)
                mujoco.mjd_inverseFD(m,d,1e-5,False,None,None,derivative,None,None,None,None)
                cache['jac']=(derivative.T@integration_map)*acc/scale[:,None]
                d.qacc[:]=integration_map@(x*acc);mujoco.mj_inverse(m,d)
            return cache['jac'] if jacobian else cache['force']
        # Keep numerical optimization error inside the unchanged physical cap.
        # The independent force/position acceptance tolerances remain unchanged.
        motor_budget=1.-1e-4
        constraints=[{'type':'eq','fun':lambda x:inverse(x)[:6],'jac':lambda x:inverse(x,True)[:6]},
            {'type':'ineq','fun':lambda x:np.r_[motor_budget-inverse(x)[6:],motor_budget+inverse(x)[6:]],
             'jac':lambda x:np.r_[-inverse(x,True)[6:],inverse(x,True)[6:]]}]
        if getattr(self,'root_acceleration_hard',False):
            root_target=np.asarray(self.root_acceleration_ref)/acc
            root_matrix=np.eye(m.nv)[3:6]
            constraints.append({'type':'eq','fun':lambda x:root_matrix@x-root_target,
                                'jac':lambda x:root_matrix})
        geometry_matrix=None;geometry_lower=None
        if getattr(self,'foot_geometry_floor',None) is not None:
            if not hasattr(self,'foot_vertices'):
                points=[]
                for g in range(m.ngeom):
                    if int(m.geom_bodyid[g])!=self.foot or not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
                    k=int(m.geom_dataid[g])
                    if k<0:raise ValueError('Foot prediction currently requires collision meshes')
                    a,n=m.mesh_vertadr[k],m.mesh_vertnum[k]
                    mesh=m.mesh_vert[a:a+n].astype(float)
                    mesh=mesh[ConvexHull(mesh).vertices]
                    world=mesh@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
                    points.extend((world-d.xpos[self.foot])@d.xmat[self.foot].reshape(3,3))
                if not points:raise ValueError('No support collision vertices')
                self.foot_vertices=np.asarray(points)
            height,point_jac,point_drift=support_point_height_kinematics(d,self.foot,self.foot_vertices,foot,drift[0],qvel)
            bound,_=position_acceleration_bounds(height,point_jac@qvel,self.foot_geometry_floor,np.inf,
                                                getattr(self,'foot_prediction_horizon',.02))
            geometry_matrix=point_jac*acc;geometry_lower=bound-point_drift
            constraints.append({'type':'ineq','fun':lambda x:geometry_matrix@x-geometry_lower,
                                'jac':lambda x:geometry_matrix})
        velocity_rows=[];velocity_lower=[]
        if grounded:
            dt=float(m.opt.timestep)
            if getattr(self,'minimum_foot_pitch_rate',None) is not None:
                velocity_rows.append(foot[4]*acc*dt)
                velocity_lower.append(self.minimum_foot_pitch_rate-foot[4]@qvel-drift[0][4]*dt)
            for dof,value in getattr(self,'minimum_joint_velocities',{}).items():
                if not 6<=dof<m.nv:raise ValueError('Expected an actuated joint velocity constraint')
                velocity_rows.append(np.eye(m.nv)[dof]*acc*dt);velocity_lower.append(value-qvel[dof])
        velocity_matrix=np.asarray(velocity_rows);velocity_rhs=np.asarray(velocity_lower)
        if len(velocity_rows):
            constraints.append({'type':'ineq','fun':lambda x:velocity_matrix@x-velocity_rhs,
                                'jac':lambda x:velocity_matrix})
        initial=np.zeros(m.nv) if self.previous is None else self.previous
        initial=np.clip(initial,lb,ub)
        result=minimize(lambda x:.5*x@hessian@x-linear@x,initial,jac=lambda x:hessian@x-linear,
            method='SLSQP',bounds=Bounds(lb,ub),constraints=constraints,options={'maxiter':80,'ftol':1e-9})
        def assess(candidate):
            result=candidate
            forces=inverse(result.x);eq_error=float(abs(forces[:6]).max());motor=float(abs(forces[6:]).max())
            bound_error=float(np.maximum(np.maximum(lb-result.x,result.x-ub),0).max())
            root_acc_error=float(abs(result.x[3:6]*acc-self.root_acceleration_ref).max()) if getattr(self,'root_acceleration_hard',False) else 0.
            geometry_error=float(np.maximum(geometry_lower-geometry_matrix@result.x,0).max()) if geometry_matrix is not None else 0.
            velocity_error=float(np.maximum(velocity_rhs-velocity_matrix@result.x,0).max()) if len(velocity_rows) else 0.
            feasible=(velocity_error<=1e-5 and root_acc_error<=1e-5 and geometry_error<=1e-5 and np.isfinite(forces).all() and np.isfinite(result.x).all() and eq_error<=1e-5 and motor<=1.00001 and bound_error<=1e-7)
            return forces,eq_error,motor,bound_error,root_acc_error,geometry_error,feasible
        first_result=result;forward_retry=False
        forces,eq_error,motor,bound_error,root_acc_error,geometry_error,feasible=assess(result)
        if (not feasible or (not result.success and int(result.status) not in (8,9))) and forward_seed is not None:
            retry=minimize(lambda x:.5*x@hessian@x-linear@x,np.clip(forward_seed,lb,ub),jac=lambda x:hessian@x-linear,
                method='SLSQP',bounds=Bounds(lb,ub),constraints=constraints,options={'maxiter':120,'ftol':1e-9})
            checked=assess(retry)
            if checked[-1] and (retry.success or int(retry.status) in (8,9)):
                result=retry;forces,eq_error,motor,bound_error,root_acc_error,geometry_error,feasible=checked;forward_retry=True
            else:
                forces,eq_error,motor,bound_error,root_acc_error,geometry_error,feasible=assess(first_result)
        if not feasible or (not result.success and int(result.status) not in (8,9)):
            self.failure={'solver_message':str(result.message),'root_force_residual_normalized':eq_error,'motor_ratio':motor,'root_acceleration_residual':root_acc_error,'geometry_acceleration_residual':geometry_error}
            self.failure_arrays={'qpos':qpos,'qvel':qvel,'qacc':integration_map@(result.x*acc),'optimization_acceleration':result.x*acc,'normalized_inverse_force':forces}
            raise RuntimeError(f'Native inverse rejected: {result.message}; root={eq_error}; motor={motor}')
        self.previous=result.x.copy()
        self.last_explicit_acceleration=integration_map@(result.x*acc)
        return forces[6:]*self.effort,{'equality_residual':eq_error,'motor_ratio':motor,'iterations':int(result.nit),
            'forward_seed_retry':forward_retry,'first_solver_status':int(first_result.status),'solver_converged':bool(result.success),'solver_status':int(result.status),'bound_residual':bound_error,
            'objective_scale':objective_scale,'prediction_horizon_s':horizon,'integration_aware':bool(getattr(self,'integration_aware',False)),'root_acceleration_residual':root_acc_error,
            'geometry_acceleration_residual':geometry_error,
            'support_velocity_residual':float(np.maximum(velocity_rhs-velocity_matrix@result.x,0).max()) if len(velocity_rows) else 0.,
            'com_acceleration_error':float(np.linalg.norm(com@(result.x*acc)-desired_com)),
            'desired_com_acceleration':(desired_com+drift[1]).tolist(),
            'achieved_com_acceleration':(com@(result.x*acc)+drift[1]).tolist(),
            'desired_momentum_rate':(desired_momentum+drift[2]).tolist(),
            'achieved_momentum_rate':(angular@(result.x*acc)+drift[2]).tolist(),
            'ground_force_world':d.qfrc_constraint[:3].tolist()}
