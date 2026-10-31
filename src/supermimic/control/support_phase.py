"""Exploratory, observation-only stop/stand stage monitor.

This does not select references or publish controls. It requires actual foot
normal loads and cannot treat foot geometry as measured support.
"""
from dataclasses import dataclass
from enum import Enum
from math import isfinite


class SupportStage(Enum):
    MOVING = 'moving'
    BRAKING = 'braking'
    LANDING = 'landing'
    HOLD = 'hold'
    UNVERIFIED = 'unverified'


@dataclass(frozen=True)
class SupportMeasurement:
    timestamp: float
    planar_speed: float
    angular_speed: float
    foot_normal_loads: tuple[float, float]
    foot_slip_speeds: tuple[float, float]


@dataclass(frozen=True)
class SupportStageConfig:
    weight_n: float
    max_age_s: float = .1
    entry_speed_m_s: float = .08
    exit_speed_m_s: float = .13
    entry_load_fraction: float = .1
    exit_load_fraction: float = .06
    entry_slip_m_s: float = .03
    exit_slip_m_s: float = .06
    entry_angular_rad_s: float = .2
    exit_angular_rad_s: float = .4
    stable_seconds: float = .5

    def __post_init__(self):
        v=tuple(getattr(self,k) for k in self.__dataclass_fields__)
        if not all(isfinite(x) and x>0 for x in v):raise ValueError('Invalid support stage thresholds')
        if (self.entry_speed_m_s>=self.exit_speed_m_s or self.entry_load_fraction<=self.exit_load_fraction
                or self.entry_slip_m_s>=self.exit_slip_m_s or self.entry_angular_rad_s>=self.exit_angular_rad_s):
            raise ValueError('Entry and exit thresholds must provide hysteresis')


class SupportStageMonitor:
    """Label stages from measured signals; output is not an actuation command."""
    def __init__(self, config: SupportStageConfig):
        self.config=config
        self.stage=SupportStage.UNVERIFIED
        self.stable_since=None
        self.last_now=None

    def update(self, now: float, movement_requested: bool, sample: SupportMeasurement | None):
        if not isfinite(now) or (self.last_now is not None and now < self.last_now):
            raise ValueError('Nonmonotonic stage-monitor time')
        self.last_now=now
        if movement_requested:
            self.stage=SupportStage.MOVING;self.stable_since=None
            return self.stage
        if self.stage==SupportStage.MOVING:
            self.stage=SupportStage.BRAKING
        valid=(sample is not None and len(sample.foot_normal_loads)==2 and len(sample.foot_slip_speeds)==2
               and all(isfinite(x) for x in (sample.timestamp,sample.planar_speed,sample.angular_speed,
                                               *sample.foot_normal_loads,*sample.foot_slip_speeds))
               and 0 <= now-sample.timestamp <= self.config.max_age_s
               and sample.planar_speed>=0 and sample.angular_speed>=0
               and min(*sample.foot_normal_loads,*sample.foot_slip_speeds)>=0)
        if not valid:
            self.stage=SupportStage.UNVERIFIED;self.stable_since=None
            return self.stage
        c=self.config
        grounded=min(sample.foot_normal_loads)>=c.entry_load_fraction*c.weight_n
        if self.stage in (SupportStage.UNVERIFIED,SupportStage.BRAKING):
            self.stage=SupportStage.LANDING if grounded and sample.planar_speed<=c.entry_speed_m_s else SupportStage.BRAKING
        if self.stage==SupportStage.HOLD:
            unstable=(sample.planar_speed>c.exit_speed_m_s or sample.angular_speed>c.exit_angular_rad_s
                      or min(sample.foot_normal_loads)<c.exit_load_fraction*c.weight_n
                      or max(sample.foot_slip_speeds)>c.exit_slip_m_s)
            if unstable:self.stage=SupportStage.LANDING;self.stable_since=None
        if self.stage==SupportStage.LANDING:
            settled=(grounded and sample.planar_speed<=c.entry_speed_m_s
                     and sample.angular_speed<=c.entry_angular_rad_s
                     and max(sample.foot_slip_speeds)<=c.entry_slip_m_s)
            if not settled:self.stable_since=None
            elif self.stable_since is None:self.stable_since=now
            elif now-self.stable_since>=c.stable_seconds:self.stage=SupportStage.HOLD
        return self.stage
