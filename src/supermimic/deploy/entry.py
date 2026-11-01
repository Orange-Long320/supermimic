"""Preparation/handover primitives. No transport, default hardware gains or I/O.

All thresholds and gains are explicit caller inputs; simulation values do not
authorize hardware. The returned command is a proposal until independently
validated and accepted by an actuator supervisor.
"""
from dataclasses import dataclass
from enum import Enum
import math

import numpy as np


class EntryPhase(Enum):
    PREPARE = 'prepare'
    SETTLE = 'settle'
    HISTORY = 'history'
    HANDOVER = 'handover'
    ACTIVE = 'active'
    FAULT = 'fault'


class QuinticPose:
    def __init__(self, start, target, max_velocity, max_acceleration, *, minimum_duration):
        self.start, self.target = np.asarray(start, dtype=float), np.asarray(target, dtype=float)
        speed, acceleration = np.asarray(max_velocity, dtype=float), np.asarray(max_acceleration, dtype=float)
        if (self.start.ndim != 1 or any(v.shape != self.start.shape for v in (self.target, speed, acceleration))
                or not np.isfinite(np.r_[self.start, self.target, speed, acceleration, minimum_duration]).all()
                or min(minimum_duration, float(speed.min()), float(acceleration.min())) <= 0):
            raise ValueError('Invalid preparation trajectory contract')
        delta = np.abs(self.target-self.start)
        # Exact maxima of the normalized quintic's first/second derivatives.
        self.duration = max(minimum_duration, float(np.max(1.875*delta/speed)),
                            float(np.max(np.sqrt((10/math.sqrt(3))*delta/acceleration))))

    def sample(self, elapsed):
        if not math.isfinite(elapsed): raise ValueError('Invalid preparation time')
        s = float(np.clip(elapsed/self.duration, 0., 1.))
        blend = 10*s**3-15*s**4+6*s**5
        velocity = (30*s**2-60*s**3+30*s**4)/self.duration
        acceleration = (60*s-180*s**2+120*s**3)/self.duration**2
        delta = self.target-self.start
        return self.start+blend*delta, velocity*delta, acceleration*delta


@dataclass(frozen=True)
class EntryThresholds:
    position_error: float
    joint_speed: float
    angular_speed: float
    gravity_angle: float
    settle_seconds: float
    settle_timeout: float
    max_tick_gap: float
    history_frames: int
    handover_seconds: float

    def __post_init__(self):
        values = [v for k, v in vars(self).items() if k != 'history_frames']
        if (not np.isfinite(values).all() or min(values) <= 0 or self.history_frames < 1
                or self.settle_timeout <= self.settle_seconds or self.gravity_angle >= math.pi):
            raise ValueError('Invalid entry thresholds')


class EntryController:
    def __init__(self, trajectory, thresholds, prep_kp, prep_kd, policy_kp, policy_kd):
        self.path, self.thresholds = trajectory, thresholds
        self.gains = tuple(np.asarray(v, dtype=float) for v in (prep_kp, prep_kd, policy_kp, policy_kd))
        if any(v.shape != trajectory.start.shape or not np.isfinite(v).all() or np.any(v < 0) for v in self.gains):
            raise ValueError('Invalid entry gains')
        self.phase = EntryPhase.PREPARE; self.reason = None
        self.started = self.last_tick = self.settle_started = self.quiet_since = self.handover_started = None
        self.history_samples = 0; self.last_command = None
        self.preparation_bias = np.zeros_like(trajectory.start)

    def fault(self, reason):
        self.phase = EntryPhase.FAULT; self.reason = str(reason)
        raise ValueError(self.reason)

    def step(self, now, q, velocity, omega, gravity, *, support_ready, policy_target=None, preparation_bias=None,
             handover_requested=True):
        if self.phase == EntryPhase.FAULT: raise ValueError('Entry fault is latched: '+self.reason)
        if type(support_ready) is not bool: self.fault('Support state must be explicitly known')
        if type(handover_requested) is not bool: self.fault('Handover request must be explicit boolean')
        q, velocity, omega, gravity = (np.asarray(x, dtype=float) for x in (q, velocity, omega, gravity))
        if (q.shape != self.path.start.shape or velocity.shape != q.shape or omega.shape != (3,)
                or gravity.shape != (3,) or not np.isfinite(np.r_[now, q, velocity, omega, gravity]).all()):
            self.fault('Invalid entry sample')
        if self.last_tick is not None and not 0 < now-self.last_tick <= self.thresholds.max_tick_gap:
            self.fault('Entry sample clock or gap')
        self.last_tick = now
        if self.started is None: self.started = now
        if preparation_bias is not None:
            if self.phase not in (EntryPhase.PREPARE, EntryPhase.SETTLE, EntryPhase.HISTORY):
                self.fault('Preparation compensation cannot change after handover starts')
            bias = np.asarray(preparation_bias, dtype=float)
            if bias.shape != q.shape or not np.isfinite(bias).all(): self.fault('Invalid preparation compensation')
            self.preparation_bias = bias.copy()
        prep_kp, prep_kd, policy_kp, policy_kd = self.gains
        if self.phase == EntryPhase.PREPARE:
            target, target_velocity, _ = self.path.sample(now-self.started)
            if now-self.started >= self.path.duration:
                self.phase = EntryPhase.SETTLE; self.settle_started = now
            command = (target+self.preparation_bias, target_velocity, prep_kp.copy(), prep_kd.copy())
        else:
            norm = np.linalg.norm(gravity)
            quiet = (support_ready is True and norm > 1e-8 and np.max(np.abs(q-self.path.target)) <= self.thresholds.position_error
                     and np.max(np.abs(velocity)) <= self.thresholds.joint_speed
                     and np.linalg.norm(omega) <= self.thresholds.angular_speed
                     and math.acos(float(np.clip(-gravity[2]/norm, -1., 1.))) <= self.thresholds.gravity_angle)
            if self.phase in (EntryPhase.SETTLE, EntryPhase.HISTORY):
                if now-self.settle_started > self.thresholds.settle_timeout:
                    self.fault('Prepared pose did not settle before timeout')
                if self.phase == EntryPhase.SETTLE:
                    self.quiet_since = (now if self.quiet_since is None else self.quiet_since) if quiet else None
                    if self.quiet_since is not None and now-self.quiet_since >= self.thresholds.settle_seconds:
                        self.phase = EntryPhase.HISTORY; self.history_samples = 0
                else:
                    if quiet:
                        self.history_samples += 1
                        if self.history_samples >= self.thresholds.history_frames and handover_requested:
                            # First handover frame is exactly the preparation command.
                            self.phase = EntryPhase.HANDOVER; self.handover_started = now
                    else:
                        self.phase = EntryPhase.SETTLE; self.quiet_since = None; self.history_samples = 0
                command = (self.path.target+self.preparation_bias, np.zeros_like(q), prep_kp.copy(), prep_kd.copy())
            else:
                if not support_ready: self.fault('Required support lost during handover or active control')
                if policy_target is None: self.fault('Missing policy target during handover/active')
                policy_target = np.asarray(policy_target, dtype=float)
                if policy_target.shape != q.shape or not np.isfinite(policy_target).all(): self.fault('Invalid policy target')
                s = float(np.clip((now-self.handover_started)/self.thresholds.handover_seconds, 0., 1.))
                blend = 10*s**3-15*s**4+6*s**5
                command = ((1-blend)*(self.path.target+self.preparation_bias)+blend*policy_target, np.zeros_like(q),
                           (1-blend)*prep_kp+blend*policy_kp, (1-blend)*prep_kd+blend*policy_kd)
                if s >= 1.: self.phase = EntryPhase.ACTIVE
        self.last_command = tuple(v.copy() for v in command)
        return self.last_command


class PreparationLoadCompensator:
    """Slow command-space bias for static PD load error, never sensor zeroing.

    Caller supplies the validated effort budget; bias is slew limited and stops
    integrating farther into predicted actuator saturation. Freeze at handover;
    EntryController fades its final command contribution continuously to zero.
    """
    def __init__(self, kp, kd, effort, *, integral_gain, max_bias_speed, effort_fraction):
        self.kp, self.kd, self.effort = (np.asarray(v, dtype=float) for v in (kp, kd, effort))
        if (self.kp.ndim != 1 or any(v.shape != self.kp.shape for v in (self.kd, self.effort))
                or not np.isfinite(np.r_[self.kp, self.kd, self.effort, integral_gain, max_bias_speed, effort_fraction]).all()
                or np.any(self.kp <= 0) or np.any(self.kd < 0) or np.any(self.effort <= 0)
                or integral_gain <= 0 or max_bias_speed <= 0 or not 0 < effort_fraction < 1):
            raise ValueError('Invalid preparation compensation contract')
        self.ki, self.speed = integral_gain, max_bias_speed
        self.limit = effort_fraction*self.effort/self.kp
        self.bias = np.zeros_like(self.kp)

    def update(self, desired, measured, velocity, dt):
        desired, measured, velocity = (np.asarray(x, dtype=float) for x in (desired, measured, velocity))
        if (any(x.shape != self.bias.shape for x in (desired, measured, velocity))
                or not np.isfinite(np.r_[desired, measured, velocity, dt]).all() or not 0 < dt <= .031):
            raise ValueError('Invalid preparation compensation sample')
        change = np.clip(self.ki*(desired-measured)*dt, -self.speed*dt, self.speed*dt)
        proposed = np.clip(self.bias+change, -self.limit, self.limit)
        effort = self.kp*(desired+proposed-measured)-self.kd*velocity
        drives_saturation = (np.abs(effort) >= self.effort) & ((proposed-self.bias)*effort > 0)
        self.bias = np.where(drives_saturation, self.bias, proposed)
        return self.bias.copy()
