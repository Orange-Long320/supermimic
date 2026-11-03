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
                 history_source='executed', max_tilt_delta=None, handover_max_tilt=None,
                 zero_stand_anchor_xy=False):
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
        if max_tilt_delta is not None and not (math.isfinite(max_tilt_delta) and 0<max_tilt_delta<math.pi):
            raise ValueError('Invalid explicit control contract')
        self.blend_seconds=blend_seconds;self.max_gap=max_tick_gap;self.max_tilt=max_tilt;self.max_omega=max_angular_speed
        # Optional support-referenced attitude guard: the first sample defines the
        # reference attitude, so a statically tilted mount (for example a hanging
        # harness) does not fault every tick. max_tilt stays a hard ceiling.
        self.max_tilt_delta=max_tilt_delta;self.tilt_reference=None
        # Handover gate: the policy is trained from an upright reference stance.
        # Entering it from a body that is measurably tilted makes the policy
        # correct a lean the legs cannot hold, which saturates the target instead
        # of standing up, so refuse the request while the tilt is too large.
        if handover_max_tilt is not None and not (math.isfinite(handover_max_tilt) and 0<handover_max_tilt<math.pi):
            raise ValueError('Invalid explicit control contract')
        if not isinstance(zero_stand_anchor_xy, bool):raise ValueError('Invalid stand anchor option')
        self.zero_stand_anchor_xy=zero_stand_anchor_xy
        self.handover_max_tilt=handover_max_tilt;self.last_tilt=None
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
        if self.handover_max_tilt is not None:
            tilt=self.last_tilt
            if tilt is None or tilt>self.handover_max_tilt:
                raise ValueError('Handover refused: measured body tilt %.3f rad exceeds the %.3f rad gate'
                                 %(float('nan') if tilt is None else tilt,self.handover_max_tilt))
        self.requested=True

    def step(self, now, position, quat_wxyz, q, qd, omega, gravity):
        if self.phase is Phase.FAULT:raise ValueError('Controller fault latched: '+self.reason)
        position,quat,q,qd,omega,gravity=(np.asarray(x,dtype=float) for x in (position,quat_wxyz,q,qd,omega,gravity))
        if (position.shape!=(3,) or quat.shape!=(4,) or q.shape!=self.default.shape or qd.shape!=q.shape
            or omega.shape!=(3,) or gravity.shape!=(3,) or not np.isfinite(np.r_[now,position,quat,q,qd,omega,gravity]).all()
            or abs(np.linalg.norm(quat)-1.)>.01 or np.linalg.norm(gravity)<1e-8):self.fault('Invalid state')
        if self.last_tick is not None and not 0<now-self.last_tick<=self.max_gap:
            self.fault(f'State timing gap: {now-self.last_tick:.6f}s at {now:.6f}')
        bad=np.flatnonzero((q<self.measured_lower)|(q>self.measured_upper))
        if len(bad):self.fault('Measured joint outside contract',dict(indices=bad.tolist(),
            measured=q[bad].tolist(),lower=self.measured_lower[bad].tolist(),upper=self.measured_upper[bad].tolist()))
        tilt=math.acos(float(np.clip(-gravity[2]/np.linalg.norm(gravity),-1.,1.)))
        if self.max_tilt_delta is not None and self.tilt_reference is None:self.tilt_reference=tilt
        deviation=0. if self.tilt_reference is None else abs(tilt-self.tilt_reference)
        self.last_tilt=tilt
        if (tilt>self.max_tilt or deviation>(self.max_tilt_delta if self.max_tilt_delta is not None else 0.)
                or np.linalg.norm(omega)>self.max_omega):self.fault('Tilt or angular speed outside contract',
            dict(tilt_rad=tilt,max_tilt_rad=self.max_tilt,tilt_deviation_rad=deviation,
                 max_tilt_delta_rad=self.max_tilt_delta,angular_speed_rad_s=float(np.linalg.norm(omega)),
                 max_angular_speed_rad_s=self.max_omega))
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
            # Term-major history: every q over the window, then every qd, then every
            # omega, gravity and previous action. This is the training layout
            # (world_model/heads.py: OBS_TERMS, "term-major history"), and the two
            # conventions are both 990 wide, so a silent swap is invisible to shape
            # checks.
            obs=np.concatenate([h[term] for term in range(5) for h in self.history])
            try:
                goal=self.tracker.goal(position,quat,q,'stand')
                if self.zero_stand_anchor_xy:
                    goal=np.asarray(goal).copy()
                    if goal.shape!=(729,):raise ValueError('Zero-XY experiment requires the 729-element anchor goal')
                    goal[-9:-7]=0.
                action=np.asarray(self.policy.infer(obs,goal),dtype=float)
                self.tracker.advance()
            except Exception as exc:self.fault('Policy or goal failed: '+str(exc))
            if action.shape!=self.default.shape or not np.isfinite(action).all():self.fault('Invalid policy output')
            desired=self.default+self.scale*action
            # Diagnostic only: the first policy ticks are what decide whether the
            # handover is gentle, and a saturated target alone does not say which
            # joint moved or how far the policy asked to go.
            self.diagnostics=dict(action=action.copy(),desired=desired.copy(),measured=q.copy())
            if not hasattr(self,'obs_dump'):
                self.obs_dump=dict(obs=obs.tolist(),q=q.tolist(),qd=list(map(float,qd)),
                    omega=list(map(float,omega)),gravity=list(map(float,gravity)),
                    previous=list(map(float,self.previous)),goal=list(map(float,goal)),
                    action=action.tolist(),default=self.default.tolist(),scale=self.scale.tolist())
            s=float(np.clip((now-self.blend_started)/self.blend_seconds,0.,1.));blend=10*s**3-15*s**4+6*s**5
            target=(1-blend)*self.blend_start+blend*desired;velocity=np.zeros_like(q)
            kp=(1-blend)*self.prep_kp+blend*self.policy_kp;kd=(1-blend)*self.prep_kd+blend*self.policy_kd
            if s>=1.:self.phase=Phase.ACTIVE
        # The reference deployment (official_sim_policy.py) sends
        # `default + scale*action` with no range check at all; the actuator saturates
        # at its limit. Faulting here instead made every handover latch on whatever
        # joint the policy saturates, which is a normal actuator condition rather
        # than a contract violation. Clamp, and record how far and how often.
        over=np.flatnonzero((target<self.command_lower)|(target>self.command_upper))
        if len(over):
            self.saturated=int(getattr(self,'saturated',0))+1
            excess=float(np.max(np.r_[self.command_lower[over]-target[over],target[over]-self.command_upper[over]]))
            self.saturation_max=float(max(getattr(self,'saturation_max',0.),excess))
            self.saturation_joints=sorted({int(i) for i in over})
            target=np.clip(target,self.command_lower,self.command_upper)
        # Same reasoning as the range clamp: the reference deployment sends the
        # policy target without a per-tick step limit. A time-based handover ramp
        # jumps whenever the host stalls for a few hundred milliseconds, and that
        # jump is a rate condition, not a contract violation. Rate-limit the command
        # instead (0.15 rad per tick is already ~7 rad/s at 48 Hz) and record it.
        delta=target-self.last_target
        over_step=np.flatnonzero(np.abs(delta)>self.max_step)
        if len(over_step):
            self.rate_limited=int(getattr(self,'rate_limited',0))+1
            self.rate_limit_max=float(max(getattr(self,'rate_limit_max',0.),
                float(np.max(np.abs(delta[over_step])-self.max_step[over_step]))))
            target=self.last_target+np.clip(delta,-self.max_step,self.max_step)
        self.last_target=target.copy()
        if self.history_source=='policy':
            if action is not None:self.previous=action.copy()
        else:self.previous=(target-self.default)/self.scale
        return {'position':target.copy(),'velocity':velocity.copy(),'effort':np.zeros_like(q),
                'stiffness':kp.copy(),'damping':kd.copy(),'phase':self.phase.value}
