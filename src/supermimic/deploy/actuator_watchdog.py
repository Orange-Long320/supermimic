"""Actuator-side deadline protocol model, with no motor or ROS transport.

This logic must execute at an independent actuator boundary to cover loss of
the command host. Running it on PC2 does NOT create hardware protection. The
current X2 HAL has no command expiry; this model cannot add that capability.

Only validated, atomically applied command frames may call accept(). Challenges
bound transport age using the receiver's monotonic clock, never a remote clock.
Every refusal latches inhibit; fresh packets cannot automatically recover it.
"""
from dataclasses import dataclass
import math
import re
import secrets


@dataclass(frozen=True)
class Challenge:
    session: str
    nonce: str
    issued_at: float


class ActuatorDeadlineWatchdog:
    def __init__(self, *, command_timeout, max_state_age, max_round_trip):
        if (not all(type(v) in (int,float) and math.isfinite(v) and v>0
                    for v in (command_timeout,max_state_age,max_round_trip))
                or max_round_trip>min(command_timeout,max_state_age)):
            raise ValueError('Invalid actuator deadline contract')
        self.timeout=command_timeout; self.max_state_age=max_state_age; self.max_round_trip=max_round_trip
        self.session=None; self.pending=None; self.last_clock=None; self.deadline=None
        self.last_command=-1; self.last_state=-1; self.last_accepted_at=None
        self.inhibit_required=True; self.reason=None

    def _trip(self, reason):
        self.reason=self.reason or reason
        self.inhibit_required=True
        self.pending=None
        return False

    def _clock(self, now):
        if type(now) not in (int,float) or not math.isfinite(now):
            return self._trip('Invalid receiver clock')
        if self.last_clock is not None and now<self.last_clock:
            return self._trip('Receiver clock moved backward')
        self.last_clock=now
        if self.reason is not None:
            return False
        if self.deadline is not None and now>=self.deadline:
            return self._trip('Actuator command deadline expired')
        return True

    def begin(self, session, now, *, inhibit_observed):
        # An already tripped object has no reset/rearm path. The physical
        # implementation must retain its latch across host restarts as well.
        if self.session is not None or self.reason is not None:
            raise ValueError('Watchdog session already used or fault latched')
        if (inhibit_observed is not True or not isinstance(session,str)
                or re.fullmatch('[0-9a-f]{32}',session) is None):
            raise ValueError('Explicit inhibited state and a unique session required')
        if not self._clock(now):
            raise ValueError(self.reason)
        self.session=session
        self.deadline=now+self.timeout
        # Receiving a first valid frame is required before inhibit may clear.

    def challenge(self, now):
        if self.session is None or not self._clock(now):
            return None
        if self.pending is not None:
            if now-self.pending.issued_at>self.max_round_trip:
                self._trip('Actuator challenge expired')
                return None
            return self.pending  # Reissuing never refreshes the deadline.
        self.pending=Challenge(self.session,secrets.token_hex(16),now)
        return self.pending

    def accept(self, *, session, nonce, command_sequence, state_sequence, state_age, now):
        if self.session is None or not self._clock(now):
            return False
        if (session!=self.session or self.pending is None or nonce!=self.pending.nonce):
            return self._trip('Wrong session or replayed challenge')
        if (type(command_sequence) is not int or command_sequence<=self.last_command
                or type(state_sequence) is not int or state_sequence<=self.last_state):
            return self._trip('Command or source state replay')
        if type(state_age) not in (int,float) or not math.isfinite(state_age) or state_age<0:
            return self._trip('Invalid source state age')
        round_trip=now-self.pending.issued_at
        # Conservatively include the entire challenge/response round trip.
        age_bound=state_age+round_trip
        if round_trip>self.max_round_trip or age_bound>=self.max_state_age:
            return self._trip('Source or transport deadline exceeded')
        self.last_command=command_sequence; self.last_state=state_sequence
        self.last_accepted_at=now
        self.deadline=now+min(self.timeout,self.max_state_age-age_bound)
        self.pending=None
        self.inhibit_required=False
        return True

    def tick(self, now):
        self._clock(now)
        return self.inhibit_required
