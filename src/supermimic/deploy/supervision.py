"""Actuator-free consent and target-lease supervision for deployment adapters.

No ROS publisher or motor-power API lives here. A deployment approval never
substitutes for a fresh, explicit motor-power confirmation bound to a plan.
"""
from dataclasses import dataclass
import math
import re


@dataclass(frozen=True)
class MotorPowerConsent:
    user_confirmed: bool
    plan_sha256: str
    confirmed_wall: float

    def require(self, plan_sha256, now_wall, *, maximum_age=300.):
        if (self.user_confirmed is not True or not re.fullmatch('[0-9a-f]{64}', self.plan_sha256)
                or self.plan_sha256 != plan_sha256 or not math.isfinite(now_wall)
                or not math.isfinite(self.confirmed_wall) or not 0 <= now_wall-self.confirmed_wall <= maximum_age):
            raise ValueError('Fresh user motor-power confirmation for this exact plan is required')


class TargetLeaseSupervisor:
    """Separate-process adapter may tick this independently of its policy worker.

    The fallback is explicit zero-stiffness damping for a supported robot, not a
    freestanding balance controller. Its gains/support conditions need separate
    hardware validation. Faults latch even if fresh policy targets later arrive.
    """
    def __init__(self, names, fallback_damping, *, max_state_age, max_target_age,
                 max_active_seconds=None, outage_grace=None):
        self.names = tuple(names); self.damping = tuple(fallback_damping)
        if (not self.names or len(set(self.names)) != len(self.names) or len(self.damping) != len(self.names)
                or not all(math.isfinite(v) and v>=0 for v in self.damping)
                or not all(math.isfinite(v) and v>0 for v in (max_state_age,max_target_age))):
            raise ValueError('Invalid lease supervisor contract')
        if max_active_seconds is not None and (type(max_active_seconds) not in (int,float)
                or not math.isfinite(max_active_seconds) or not 0<max_active_seconds<=30):
            raise ValueError('Invalid bounded active duration')
        self.max_state_age, self.max_target_age = max_state_age, max_target_age
        # Outage judgement shared with the controller's tick-gap limit. Never used to
        # accept stale data: only to decide when an absence of fresh input becomes
        # fallback instead of a latch on one late packet.
        self.lease_grace = float(outage_grace) if outage_grace else 7 * max(max_state_age, max_target_age)
        self.max_active_seconds=max_active_seconds;self.first_output_at=None
        self.last_sequence = -1; self.latest = None; self.fault_reason = None

    def accept(self, packet):
        if self.fault_reason is not None: return False
        try:
            if tuple(packet['names']) != self.names: raise ValueError('Target joint order')
            for key in ('position','velocity','effort','stiffness','damping'):
                values=packet[key]
                if len(values)!=len(self.names) or not all(math.isfinite(x) for x in values):raise ValueError('Invalid '+key)
                if key in ('stiffness','damping') and any(x<0 for x in values):raise ValueError('Negative '+key)
            if type(packet['sequence']) is not int or packet['sequence']<=self.last_sequence:raise ValueError('Target sequence rollback/replay')
            if not all(math.isfinite(packet[k]) for k in ('generated','state_sampled')):raise ValueError('Invalid target clocks')
            self.latest = {key:(list(value) if isinstance(value,(tuple,list)) else value) for key,value in packet.items()}
            self.last_sequence = packet['sequence']
            return True
        except (KeyError, TypeError, ValueError) as exc:
            self.fault_reason=str(exc)
            return False

    def tick(self, now):
        if not math.isfinite(now): self.fault_reason='Invalid supervisor clock'
        if self.latest is not None and self.fault_reason is None:
            if not 0<=now-self.latest['generated']<=self.lease_grace:
                # Same grace as the state lease: one late target is not an outage.
                self.fault_reason='Policy target lease expired'
            elif not 0<=now-self.latest['state_sampled']<=self.lease_grace:
                # A host-level stall of ~100 ms was observed on this robot, so a lease
                # tuned to the 50 ms packet bound latches on jitter and then rejects
                # every later target as a step violation. The grace equals the
                # controller's own tick-gap limit: both stages call the stream stopped
                # at the same instant, and neither is stricter than the contract.
                self.fault_reason='Source state lease expired'
            elif self.first_output_at is None:self.first_output_at=now
            if (self.fault_reason is None and self.max_active_seconds is not None
                    and now-self.first_output_at>=self.max_active_seconds):
                self.fault_reason='Bounded manually supervised trial ended'
        if self.latest is None and self.fault_reason is None:
            return None  # Never emits an invented startup pose before any target.
        if self.fault_reason is not None:
            n=len(self.names)
            return {'names':self.names,'position':[0.]*n,'velocity':[0.]*n,'effort':[0.]*n,
                    'stiffness':[0.]*n,'damping':self.damping,'fallback':True,'reason':self.fault_reason}
        return dict(self.latest, fallback=False)
