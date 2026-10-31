"""Experimental Walk/Idle retrieval planner; geometry is not measured support.

Keep the committed future window, match candidate entries to both that window
and measured tracking error, and join a recorded trajectory with a C1 correction.
No learned motion model, contact-force prediction, or dynamics guarantee is used.
"""
from collections import deque

import numpy as np

from supermimic.control.skill_phase import SkillPhase


class ContextIdlePlanner(SkillPhase):
    def __init__(self, *args, joint_limits, velocity_limits, foot_features,
                 baseline_motion=False, baseline_idle=False, idle_join_frames=None,
                 idle_entry_offset=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.joint_limits = np.asarray(joint_limits, dtype=float)
        self.velocity_limits = np.asarray(velocity_limits, dtype=float)
        if (self.joint_limits.shape != (self.nj, 2)
                or self.velocity_limits.shape != (self.nj,)
                or not np.isfinite(self.joint_limits).all()
                or not np.isfinite(self.velocity_limits).all()
                or np.any(self.joint_limits[:, 0] >= self.joint_limits[:, 1])
                or np.any(self.velocity_limits <= 0)):
            raise ValueError('Invalid planner mechanical limits')
        self.foot_features = foot_features
        self.baseline_motion, self.baseline_idle = baseline_motion, baseline_idle
        if idle_join_frames not in (None, 10, 20, 30) or idle_entry_offset not in (None, 0, 5, 10):
            raise ValueError('Invalid diagnostic Idle entry or join')
        self.idle_join_frames, self.idle_entry_offset = idle_join_frames, idle_entry_offset
        self.context = deque(maxlen=4)
        self.loops = dict(self.loops)
        for name, (start, end) in tuple(self.loops.items()):
            if baseline_motion:
                # Read-only causal control: replay the original motion path,
                # including its original unscreened seam. Never a fix/adoption.
                continue
            clip = self.gaits[name]
            seams = []
            # Existing cycle ranking ignores hard limits of its Hermite join.
            # Check nearby recorded end frames before publishing any control.
            for delta in (0, -1, 1, -2, 2):
                candidate_end = end+delta
                if not start+20 <= candidate_end < len(clip):
                    continue
                for join in (10, 15, 20):
                    _, info = self._candidate(clip[candidate_end], clip[start:start+join+1], join)
                    if info['valid']:
                        seams.append((abs(delta)+.001*info['acceleration_rms_rad_s2'], candidate_end, join, info))
            if not seams:
                raise ValueError('No mechanically admissible sustained cycle: '+name)
            _, selected_end, join, info = min(seams, key=lambda item: item[0])
            self.loops[name] = (start, selected_end)
            self.events.append(dict(kind='cycle_configuration', skill=name, original_bounds=[start, end],
                                    selected_bounds=[start, selected_end], join_frames=join, **info))
        # Cache geometry of template entries before starting the control loop.
        self.entries = {}
        for mode in ('walk', 'stand'):
            clip = self.gaits['walk'] if mode == 'walk' else self.stops['walk']
            first = 0 if mode == 'walk' else self.brake_start['walk']
            offsets = (idle_entry_offset,) if mode == 'stand' and idle_entry_offset is not None else (0, 5, 10)
            self.entries[mode] = [(first+i, self._feet(clip[first+i]))
                                  for i in offsets if first+i+21 < len(clip)]
        if any(not v for v in self.entries.values()):
            raise ValueError('References lack planner entry candidates')

    def _feet(self, row):
        feet = np.asarray(self.foot_features(row), dtype=float)
        if feet.shape != (2, 3) or not np.isfinite(feet).all():
            raise ValueError('Expected two finite foot positions in heading coordinates')
        return feet

    def set_context(self, joint_position_delta, joint_velocity):
        q, v = np.asarray(joint_position_delta), np.asarray(joint_velocity)
        if q.shape != (self.nj,) or v.shape != (self.nj,) or not np.isfinite([q, v]).all():
            raise ValueError('Invalid measured planner context')
        self.context.append((q.copy(), v.copy()))

    def _next(self):
        if self.baseline_motion:
            return super()._next()
        if self.plan or self.source == 'stand':
            return super()._next()
        clip = self.gaits[self.source]
        end = self.loops[self.source][1] if self.source in self.loops else len(clip)-1
        if self.index <= end:
            return super()._next()
        if self.source not in self.loops:
            raise ValueError('Context planner requires explicit sustained cycle bounds')
        start = self.loops[self.source][0]
        source = self.queue[-1]
        candidates = []
        for join in (10, 15, 20):
            if start+join > end:
                continue
            rows, info = self._candidate(source, clip[start:start+join+1], join)
            if info['valid']:
                candidates.append((info['acceleration_rms_rad_s2'], join, rows, info))
        if not candidates:
            raise ValueError('No mechanically admissible cycle seam')
        _, join, rows, info = min(candidates, key=lambda item: item[0])
        self.events.append(dict(kind='loop_seam', scheduler_step=self.steps, skill=self.source,
                                from_frame=end, to_frame=start, bridge_end_frame=start+join,
                                **info))
        self.plan.extend(rows)
        self.index = start+join+1
        return self.plan.popleft()

    def _candidate(self, source, template, join):
        rows = np.concatenate((self._blend(source, template[:join+1]), template[join+1:]))
        q = np.concatenate((source[None, :self.nj], rows[:, :self.nj]))
        declared_v = rows[:, self.nj:2*self.nj]
        excess = float(np.maximum(np.maximum(self.joint_limits[:, 0]-q,
                                             q-self.joint_limits[:, 1]), 0).max())
        finite_v = np.diff(q, axis=0)/self.dt
        speed_ratio = float(max(np.max(np.abs(declared_v)/self.velocity_limits),
                                np.max(np.abs(finite_v)/self.velocity_limits)))
        acceleration = np.diff(np.concatenate((source[None, self.nj:2*self.nj], declared_v)), axis=0)/self.dt
        acceleration_rms = float(np.sqrt(np.mean(acceleration**2)))
        return rows, dict(hard_excess_rad=excess, joint_speed_ratio=speed_ratio,
                          acceleration_rms_rad_s2=acceleration_rms,
                          valid=bool(np.isfinite(rows).all() and excess <= 1e-4 and speed_ratio <= 1.))

    def select(self, mode):
        mode = 'stand' if mode == 'idle' else mode
        if mode not in ('walk', 'stand'):
            raise ValueError('Context planner prototype supports Walk and Idle only')
        if mode == self.requested:
            return
        if (mode == 'stand' and self.baseline_idle) or (mode == 'walk' and self.baseline_motion):
            return super().select(mode)
        if not self.context:
            raise ValueError('Measured joint context required before replanning')
        source = self.queue[-1].copy()
        current = self.queue[0]
        q, v = self.context[-1]
        # Constant tracking-error extrapolation to the committed horizon.
        # This is only a ranking heuristic, not a predicted physical state.
        estimated = source.copy()
        estimated[:self.nj] += q-current[:self.nj]
        estimated[self.nj:2*self.nj] += v-current[self.nj:2*self.nj]
        feet = self._feet(estimated)
        clip = self.gaits['walk'] if mode == 'walk' else self.stops['walk']
        candidates, choices = [], []
        for entry, entry_feet in self.entries[mode]:
            joins = (self.idle_join_frames,) if mode == 'stand' and self.idle_join_frames is not None else (10, 20)
            for join in joins:
                # Idle retains the complete recorded settling tail. Walking
                # hands back to the gait at the end of its entry bridge.
                template = clip[entry:entry+join+1] if mode == 'walk' else clip[entry:]
                rows, diagnostic = self._candidate(source, template, join)
                score = (np.mean(((estimated[:self.nj]-clip[entry, :self.nj])/.25)**2)
                         + .1*np.mean(((estimated[self.nj:2*self.nj]-clip[entry, self.nj:2*self.nj]))**2)
                         + .25*np.mean(((feet-entry_feet)/.05)**2)
                         + .001*diagnostic['acceleration_rms_rad_s2']**2
                         + .1*join*self.dt)
                if mode == 'stand':
                    # Prefer a shorter integrated reference stop, subject to
                    # entry matching. This is not measured stopping distance.
                    score += .5*np.linalg.norm(rows[:, 2*self.nj+6:2*self.nj+8], axis=1).sum()*self.dt/.3
                info = dict(entry_frame=entry, join_frames=join, score=float(score), **diagnostic)
                candidates.append(info)
                if diagnostic['valid']:
                    choices.append((float(score), rows, info))
        if not choices:
            self.events.append(dict(kind='planner_rejected', scheduler_step=self.steps,
                                    target_mode=mode, candidates=candidates))
            raise ValueError('No mechanically admissible context transition')
        _, rows, selected = min(choices, key=lambda x: x[0])
        previous = self.requested
        old_source, old_index = self.source, self.index
        self.plan.clear()
        self.plan.extend(rows)
        self.source = mode
        self.index = selected['entry_frame']+selected['join_frames']+1 if mode == 'walk' else 0
        self.requested = mode
        self.events.append(dict(kind='context_transition', scheduler_step=self.steps,
                                **{'from': previous, 'to': mode}, target_mode='Idle' if mode == 'stand' else 'Walk',
                                source_at_selection=old_source, next_frame_at_selection=old_index,
                                committed_frames=self.horizon, planned_frames=len(rows),
                                measured_history_frames=len(self.context),
                                tracking_position_rms_rad=float(np.sqrt(np.mean((q-current[:self.nj])**2))),
                                selected=selected, candidates=candidates,
                                scope='retrieval and C1 correction; no contact or dynamics proof'))
