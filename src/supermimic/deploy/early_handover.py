"""Early handover controller core. No ROS, motor output, or power operation.

The caller must validate timestamps and each complete sensor packet. All bounds,
gains and safety thresholds must be provided by a separately verified contract.
"""
from collections import deque
from enum import Enum
import math

import numpy as np
from supermimic.deploy.entry import QuinticPose


class Phase(Enum):
    PREPARE='prepare'
    HANDOVER='handover'
    ACTIVE='active'
    FAULT='fault'


class EarlyHandover:
    def __init__(self, start, target, default, scale, prep_kp, prep_kd, policy_kp, policy_kd,
                 measured_lower, measured_upper, command_lower, command_upper, max_step,
                 *, max_velocity, max_acceleration, minimum_duration, blend_seconds,
                 max_tick_gap, max_tilt, max_angular_speed, policy, tracker_factory,
                 history_source='executed'):
        vectors=[np.asarray(x,dtype=float) for x in (start,target,default,scale,prep_kp,prep_kd,policy_kp,policy_kd,
                 measured_lower,measured_upper,command_lower,command_upper,max_step,max_velocity,max_acceleration)]
        n=len(vectors[0]);
        if n<1 or any(x.shape!=(n,) or not np.isfinite(x).all() for x in vectors):raise ValueError('Invalid contract vectors')
        (start,target,self.default,self.scale,self.prep_kp,self.prep_kd,self.policy_kp,self.policy_kd,
         self.measured_lower,self.measured_upper,self.command_lower,self.command_upper,self.max_step,vmax,amax)=vectors
        if (np.any(self.scale<=0) or np.any(self.measured_lower>=self.measured_upper)
            or np.any(self.command_lower>=self.command_upper) or np.any(self.max_step<=0)
            or any(np.any(x<0) for x in (self.prep_kp,self.prep_kd,self.policy_kp,self.policy_kd))
            or np.any(start<self.measured_lower) or np.any(start>self.measured_upper)
            or np.any(start<self.command_lower) or np.any(start>self.command_upper)
            or np.any(target<self.command_lower) or np.any(target>self.command_upper)
            or not all(math.isfinite(v) and v>0 for v in (blend_seconds,max_tick_gap,max_tilt,max_angular_speed))
            or max_tilt>=math.pi or history_source not in ('executed','policy')):raise ValueError('Invalid explicit control contract')
        self.path=QuinticPose(start,target,vmax,amax,minimum_duration=minimum_duration)
        self.blend_seconds=blend_seconds;self.max_gap=max_tick_gap;self.max_tilt=max_tilt;self.max_omega=max_angular_speed
        self.policy=policy;self.tracker_factory=tracker_factory;self.tracker=None
        self.phase=Phase.PREPARE;self.reason=None;self.fault_context=None;self.history=deque(maxlen=10)
        self.history_source=history_source
        # No policy action has yet been generated. The policy-history mode
        # matches the training reset convention of zero previous action.
        self.previous=(np.zeros_like(start) if history_source=='policy' else (start-self.default)/self.scale)
        self.last_target=start.copy();self.last_tick=None;self.started=None
        self.requested=False;self.blend_started=None;self.blend_start=None

    def fault(self,reason,context=None):
        self.phase=Phase.FAULT;self.reason=str(reason);self.fault_context=context;raise ValueError(self.reason)

    def request_handover(self):
        if self.phase is not Phase.PREPARE or len(self.history)!=10:raise ValueError('Ten fresh observations required')
        self.requested=True

    def step(self, now, position, quat_wxyz, q, qd, omega, gravity):
        if self.phase is Phase.FAULT:raise ValueError('Controller fault latched: '+self.reason)
        position,quat,q,qd,omega,gravity=(np.asarray(x,dtype=float) for x in (position,quat_wxyz,q,qd,omega,gravity))
        if (position.shape!=(3,) or quat.shape!=(4,) or q.shape!=self.default.shape or qd.shape!=q.shape
            or omega.shape!=(3,) or gravity.shape!=(3,) or not np.isfinite(np.r_[now,position,quat,q,qd,omega,gravity]).all()
            or abs(np.linalg.norm(quat)-1.)>.01 or np.linalg.norm(gravity)<1e-8):self.fault('Invalid state')
        if self.last_tick is not None and not 0<now-self.last_tick<=self.max_gap:
            self.fault(f'State timing gap: {now-self.last_tick:.6f}s at {now:.6f}')
        if np.any(q<self.measured_lower) or np.any(q>self.measured_upper):self.fault('Measured joint outside contract')
        tilt=math.acos(float(np.clip(-gravity[2]/np.linalg.norm(gravity),-1.,1.)))
        if tilt>self.max_tilt or np.linalg.norm(omega)>self.max_omega:self.fault('Tilt or angular speed outside contract')
        self.last_tick=now
        if self.started is None:self.started=now
        self.history.append((q-self.default,qd,omega,gravity,self.previous.copy()))
        if self.requested and self.phase is Phase.PREPARE:
            self.phase=Phase.HANDOVER;self.blend_started=now;self.blend_start=self.last_target.copy()
            try:self.tracker=self.tracker_factory(position.copy(),quat.copy())
            except Exception as exc:self.fault('Tracker initialization failed: '+str(exc))
        action=None
        if self.phase is Phase.PREPARE:
            target,_,_=self.path.sample(now-self.started);velocity=np.zeros_like(q);kp=self.prep_kp;kd=self.prep_kd
        else:
            obs=np.concatenate([h[term] for term in range(5) for h in self.history])
            try:
                goal=self.tracker.goal(position,quat,q,'stand');action=np.asarray(self.policy.infer(obs,goal),dtype=float)
                self.tracker.advance()
            except Exception as exc:self.fault('Policy or goal failed: '+str(exc))
            if action.shape!=self.default.shape or not np.isfinite(action).all():self.fault('Invalid policy output')
            desired=self.default+self.scale*action
            s=float(np.clip((now-self.blend_started)/self.blend_seconds,0.,1.));blend=10*s**3-15*s**4+6*s**5
            target=(1-blend)*self.blend_start+blend*desired;velocity=np.zeros_like(q)
            kp=(1-blend)*self.prep_kp+blend*self.policy_kp;kd=(1-blend)*self.prep_kd+blend*self.policy_kd
            if s>=1.:self.phase=Phase.ACTIVE
        bad=np.flatnonzero((target<self.command_lower)|(target>self.command_upper))
        if len(bad):self.fault('Target outside command contract',dict(indices=bad.tolist(),
            target=target[bad].tolist(),lower=self.command_lower[bad].tolist(),upper=self.command_upper[bad].tolist(),
            raw_policy_desired=None if action is None else (self.default+self.scale*action)[bad].tolist()))
        delta=target-self.last_target;bad=np.flatnonzero(np.abs(delta)>self.max_step)
        if len(bad):self.fault('Target step exceeds command contract',dict(indices=bad.tolist(),
            delta=delta[bad].tolist(),allowed=self.max_step[bad].tolist()))
        self.last_target=target.copy()
        if self.history_source=='policy':
            if action is not None:self.previous=action.copy()
        else:self.previous=(target-self.default)/self.scale
        return {'position':target.copy(),'velocity':velocity.copy(),'effort':np.zeros_like(q),
                'stiffness':kp.copy(),'damping':kd.copy(),'phase':self.phase.value}
