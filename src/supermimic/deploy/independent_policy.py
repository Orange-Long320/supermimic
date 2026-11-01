"""Actuator-free 11500 preparation and requested policy handover coordinator.

This computes proposals only. It has no motor transport or power interface.
Measured-state validation and independent actuator supervision remain required.
"""
from collections import deque

import numpy as np

from supermimic.deploy.entry import EntryController, EntryPhase


class IndependentPolicyCoordinator:
    def __init__(self, entry, default, scale, policy, tracker_factory):
        if not isinstance(entry, EntryController): raise ValueError('Expected entry controller')
        self.entry=entry; self.default=np.asarray(default,dtype=float); self.scale=np.asarray(scale,dtype=float)
        if (self.default.shape!=entry.path.start.shape or self.scale.shape!=self.default.shape
                or not np.isfinite(np.r_[self.default,self.scale]).all() or np.any(self.scale==0)):
            raise ValueError('Invalid policy action contract')
        self.policy=policy; self.tracker_factory=tracker_factory; self.tracker=None
        self.history=deque(maxlen=entry.thresholds.history_frames)
        self.previous=(entry.path.start-self.default)/self.scale
        self.requested=False

    def request_handover(self):
        if self.entry.phase is not EntryPhase.HISTORY or len(self.history)!=self.history.maxlen:
            raise ValueError('Preparation and observation history not ready')
        self.requested=True

    def step(self, now, position, quat_wxyz, q, qd, omega, gravity, *, support_ready):
        q=np.asarray(q,dtype=float); qd=np.asarray(qd,dtype=float)
        omega=np.asarray(omega,dtype=float); gravity=np.asarray(gravity,dtype=float)
        position=np.asarray(position,dtype=float); quat=np.asarray(quat_wxyz,dtype=float)
        if (position.shape!=(3,) or quat.shape!=(4,) or not np.isfinite(np.r_[position,quat]).all()
                or abs(np.linalg.norm(quat)-1.)>.01):
            self.entry.fault('Invalid pelvis pose')
        if (q.shape!=self.default.shape or qd.shape!=q.shape or omega.shape!=(3,) or gravity.shape!=(3,)
                or not np.isfinite(np.r_[q,qd,omega,gravity]).all()):
            self.entry.fault('Invalid observation')
        if self.entry.phase is EntryPhase.FAULT: self.entry.fault('Entry fault is latched')
        before=self.entry.phase; proposed=None
        if before in (EntryPhase.HISTORY,EntryPhase.HANDOVER,EntryPhase.ACTIVE):
            self.history.append((q-self.default,qd,omega,gravity,self.previous.copy()))
            if self.tracker is None:
                try: self.tracker=self.tracker_factory(position.copy(),quat.copy())
                except Exception as exc: self.entry.fault('Tracker initialization failed: '+str(exc))
        if before in (EntryPhase.HANDOVER,EntryPhase.ACTIVE):
            if len(self.history)!=self.history.maxlen: self.entry.fault('Incomplete real observation history')
            obs=np.concatenate([h[term] for term in range(5) for h in self.history])
            try:
                goal=self.tracker.goal(position,quat,q,'stand')
                action=np.asarray(self.policy.infer(obs,goal),dtype=float)
                self.tracker.advance()
            except Exception as exc:
                self.entry.fault('Policy or goal failed: '+str(exc))
            if action.shape!=self.default.shape or not np.isfinite(action).all(): self.entry.fault('Invalid policy action')
            proposed=self.default+self.scale*action
        command=self.entry.step(now,q,qd,omega,gravity,support_ready=support_ready,
                                policy_target=proposed,handover_requested=self.requested)
        if self.entry.phase is EntryPhase.SETTLE and before is not EntryPhase.SETTLE:
            self.history.clear(); self.tracker=None; self.requested=False
        self.previous=(command[0]-self.default)/self.scale
        return command
