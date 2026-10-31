"""Offline control boundary. Timestamps use one local monotonic clock, in seconds."""
from dataclasses import dataclass
from collections import deque
from enum import Enum
from math import acos, isfinite, sqrt
from typing import Protocol


class DeploymentError(ValueError):
    pass


def finite(values):
    return all(isfinite(x) for x in values)


@dataclass(frozen=True)
class JointSample:
    name: str
    position: float
    velocity: float
    error_code: int


@dataclass(frozen=True)
class State:
    joints: tuple[JointSample, ...]
    joint_times: tuple[float, ...]  # per-joint local receive time, not ROS header time
    imu_time: float
    orientation_wxyz: tuple[float, float, float, float]
    angular_velocity: tuple[float, float, float]


@dataclass(frozen=True)
class Contract:
    names: tuple[str, ...]
    default: tuple[float, ...]
    scale: tuple[float, ...]
    lower: tuple[float, ...]  # measured joint bounds; distinct from PD setpoints
    upper: tuple[float, ...]
    max_target_step: tuple[float, ...]
    max_state_age: float
    max_skew: float
    history: int
    goal_dim: int
    # Supply only from a verified command interface. None preserves the original
    # conservative bounds; never infer permissive setpoint bounds from a policy.
    target_lower: tuple[float, ...] | None = None
    target_upper: tuple[float, ...] | None = None

    def __post_init__(self):
        n = len(self.names)
        vectors = (self.default, self.scale, self.lower, self.upper, self.max_target_step)
        if not n or len(set(self.names)) != n or any(not x for x in self.names):
            raise DeploymentError('joint names must be nonempty and unique')
        if any(len(v) != n or not finite(v) for v in vectors):
            raise DeploymentError('invalid joint vector')
        if any(lo >= hi or step <= 0 or not lo <= q <= hi for lo, hi, step, q in zip(self.lower, self.upper, self.max_target_step, self.default)):
            raise DeploymentError('invalid limits, step or default')
        if any(s <= 0 for s in self.scale) or not finite((self.max_state_age, self.max_skew)) or min(self.max_state_age, self.max_skew) <= 0:
            raise DeploymentError('invalid scale or timing')
        if self.history < 1 or self.goal_dim < 1:
            raise DeploymentError('invalid policy dimensions')
        if (self.target_lower is None) != (self.target_upper is None):
            raise DeploymentError('both target bounds are required')
        if self.target_lower is not None:
            if any(len(v) != n or not finite(v) for v in (self.target_lower, self.target_upper)):
                raise DeploymentError('invalid target bounds')
            if any(not lo <= q <= hi or lo >= hi
                   for lo, q, hi in zip(self.target_lower, self.default, self.target_upper)):
                raise DeploymentError('invalid target range or default')

    @property
    def target_bounds(self):
        return (self.lower, self.upper) if self.target_lower is None else (self.target_lower, self.target_upper)


@dataclass(frozen=True)
class StartupPose:
    """Explicit input-matching gate, not a motion generator or calibration.

    Use the selected reference pose, not the action offset in Contract.default.
    Tolerances must be supplied by the caller and independently validated.
    """
    names: tuple[str, ...]
    position: tuple[float, ...]
    position_tolerance: tuple[float, ...]
    max_speed: tuple[float, ...]
    gravity: tuple[float, float, float]
    max_gravity_angle: float

    def validate(self, spec):
        n = len(spec.names)
        if self.names != spec.names or any(len(v) != n or not finite(v)
                for v in (self.position, self.position_tolerance, self.max_speed)):
            raise DeploymentError('startup joint contract mismatch')
        if any(t <= 0 or speed <= 0 or not lo <= q <= hi
               for q, t, speed, lo, hi in zip(self.position, self.position_tolerance,
                                            self.max_speed, spec.lower, spec.upper)):
            raise DeploymentError('invalid startup pose or tolerances')
        if (len(self.gravity) != 3 or not finite((*self.gravity, self.max_gravity_angle))
                or abs(sum(v*v for v in self.gravity)-1.) > 1e-6
                or not 0 < self.max_gravity_angle < 3.141592653589793):
            raise DeploymentError('invalid startup gravity')

    def matches(self, joints, gravity):
        if any(abs(j.position-q) > tolerance or abs(j.velocity) > speed
               for j, q, tolerance, speed in zip(joints, self.position, self.position_tolerance, self.max_speed)):
            return False
        norm = sqrt(sum(v*v for v in gravity))
        if norm < 1e-8:
            return False
        cosine = sum(a*b for a, b in zip(gravity, self.gravity))/norm
        return acos(max(-1., min(1., cosine))) <= self.max_gravity_angle


def order_and_validate(state: State, spec: Contract, now: float):
    if len(state.joints) != len(state.joint_times) or not finite((now, state.imu_time, *state.joint_times)):
        raise DeploymentError('invalid timestamps')
    if len(state.orientation_wxyz) != 4 or len(state.angular_velocity) != 3 or not finite((*state.orientation_wxyz, *state.angular_velocity)):
        raise DeploymentError('invalid IMU')
    norm = sqrt(sum(v*v for v in state.orientation_wxyz))
    if abs(norm - 1.0) > .01:
        raise DeploymentError('IMU quaternion not normalized')
    by_name = {}
    for joint, stamp in zip(state.joints, state.joint_times):
        if joint.name in by_name or not finite((joint.position, joint.velocity)) or joint.error_code:
            raise DeploymentError('duplicate, invalid or faulted joint')
        by_name[joint.name] = (joint, stamp)
    if set(by_name) != set(spec.names):
        raise DeploymentError('joint set does not match configured model')
    stamps = [by_name[name][1] for name in spec.names] + [state.imu_time]
    if any(t > now or now - t > spec.max_state_age for t in stamps) or max(stamps) - min(stamps) > spec.max_skew:
        raise DeploymentError('stale, future or asynchronous state')
    ordered = tuple(by_name[name][0] for name in spec.names)
    if any(not lo <= j.position <= hi for j, lo, hi in zip(ordered, spec.lower, spec.upper)):
        raise DeploymentError('measured joint outside limits')
    return ordered


class Policy(Protocol):
    def infer(self, observation: tuple[float, ...], goal: tuple[float, ...]) -> tuple[float, ...]: ...


class GoalProvider(Protocol):
    def goal(self, state: State, command: str) -> tuple[float, ...]: ...


class Mode(Enum):
    DISARMED = 'disarmed'
    SHADOW = 'shadow'
    FAULT = 'fault'


class ShadowController:
    """Computes position targets for logging only. No ROS publishing or actuation."""
    def __init__(self, spec: Contract, policy: Policy, goals: GoalProvider, *, startup: StartupPose | None = None):
        self.spec, self.policy, self.goals = spec, policy, goals
        self.startup = startup
        if startup is not None:
            startup.validate(spec)
        self.startup_complete = False
        self.mode, self.reason = Mode.DISARMED, None
        self.previous_action = (0.,) * len(spec.names)
        self.previous_target = None
        self.history = deque(maxlen=spec.history)

    def start(self):
        if self.mode != Mode.DISARMED:
            raise DeploymentError('reset required before starting')
        self.mode = Mode.SHADOW

    def stop(self):
        self.mode = Mode.DISARMED
        self.previous_action = (0.,) * len(self.spec.names)
        self.previous_target = None
        self.history.clear()
        self.reason = None
        self.startup_complete = False

    def step(self, state: State, command: str, now: float):
        if self.mode != Mode.SHADOW:
            raise DeploymentError('controller is not in shadow mode')
        try:
            joints = order_and_validate(state, self.spec, now)
            # IMU orientation is wxyz, body-to-world; project world gravity into body.
            w, x, y, z = state.orientation_wxyz
            gravity = (2*(w*y-x*z), -2*(w*x+y*z), -1+2*(x*x+y*y))
            if self.startup is not None and not self.startup_complete:
                if command != 'stand':
                    raise DeploymentError('startup requires stand command')
                if not self.startup.matches(joints, gravity):
                    self.history.clear()
                    return None
            terms = (tuple(j.position-q for j, q in zip(joints, self.spec.default)),
                     tuple(j.velocity for j in joints), state.angular_velocity, gravity, self.previous_action)
            self.history.append(terms)
            # Never invent an initial observation history.
            if len(self.history) < self.spec.history:
                return None
            obs = tuple(v for term in range(5) for frame in self.history for v in frame[term])
            goal = tuple(self.goals.goal(state, command))
            if len(goal) != self.spec.goal_dim or not finite(goal):
                raise DeploymentError('invalid goal')
            action = tuple(self.policy.infer(obs, goal))
            if len(action) != len(joints) or not finite(action):
                raise DeploymentError('invalid policy action')
            target = tuple(q + s*a for q, s, a in zip(self.spec.default, self.spec.scale, action))
            previous = self.previous_target or tuple(j.position for j in joints)
            lower, upper = self.spec.target_bounds
            if any(not lo <= q <= hi or abs(q-p) > step for q, p, lo, hi, step in zip(target, previous, lower, upper, self.spec.max_target_step)):
                raise DeploymentError('target outside command or step limits')
            self.previous_action, self.previous_target = action, target
            self.startup_complete = True
            return target
        except Exception as exc:
            self.mode, self.reason = Mode.FAULT, str(exc)
            raise
