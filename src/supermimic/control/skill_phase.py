"""Continuous future-goal scheduler for exploratory keyboard motion tracking.

The fixed-size published window is never rewritten. This scheduler does not
establish physical feasibility or contact preservation of interpolated goals.
"""
from collections import deque

import numpy as np


class SkillPhase:
    def __init__(self, gaits, stops, brake_start, joint_count, horizon=10, dt=.02, loops=None):
        self.gaits = {k: np.asarray(v, dtype=np.float32) for k, v in gaits.items()}
        self.stops = {k: np.asarray(v, dtype=np.float32) for k, v in stops.items()}
        self.brake_start = brake_start
        self.nj, self.horizon, self.dt = joint_count, horizon, dt
        self.loops = loops or {}
        for name, (start, end) in self.loops.items():
            if name not in self.gaits or not (0 <= start < end < len(self.gaits[name]) and end-start >= 20):
                raise ValueError('Invalid gait cycle bounds')
        if not self.gaits or set(self.gaits) != set(self.stops) or horizon < 1:
            raise ValueError('Missing gait or stop references')
        self.stand = self.stops['walk'][-1].copy()
        self.stand[self.nj:2*self.nj] = 0.
        self.queue = deque([self.stand.copy() for _ in range(horizon)])
        self.plan = deque()
        self.requested = 'stand'
        self.source = 'stand'
        self.index = 0
        self.events = []
        self.steps = 0

    def _blend(self, source, template):
        """Correct a target trajectory to match the published tail and tangent."""
        template = np.asarray(template, dtype=np.float32)
        n = len(template)-1
        if n < 2:
            raise ValueError('Transition too short')
        t = np.arange(1, n+1, dtype=np.float64) / n
        duration = n*self.dt
        h00 = 2*t**3-3*t**2+1
        h10 = t**3-2*t**2+t
        dh00 = (6*t**2-6*t)/duration
        dh10 = 3*t**2-4*t+1
        delta_q = source[:self.nj]-template[0, :self.nj]
        delta_v = source[self.nj:2*self.nj]-template[0, self.nj:2*self.nj]
        rows = template[1:].copy()
        rows[:, :self.nj] += h00[:, None]*delta_q + (duration*h10)[:, None]*delta_v
        rows[:, self.nj:2*self.nj] += dh00[:, None]*delta_q + dh10[:, None]*delta_v
        rows[:, 2*self.nj:] += h00[:, None]*(source[2*self.nj:]-template[0, 2*self.nj:])
        grav = rows[:, 2*self.nj:2*self.nj+3]
        grav /= np.maximum(np.linalg.norm(grav, axis=1, keepdims=True), 1e-8)
        return rows

    def select(self, mode):
        if mode != 'stand' and mode not in self.gaits:
            raise ValueError('Unknown skill: '+mode)
        if mode == self.requested:
            return
        tail = self.queue[-1].copy()
        self.plan.clear()
        previous = self.requested
        if previous != 'stand':
            stop = self.stops[previous][self.brake_start[previous]:]
            # The trained stop has a 30-frame brake followed by a static hold.
            brake = stop[:31]
            for row in self._blend(tail, brake):
                self.plan.append(row)
            tail = self.plan[-1].copy()
        if mode != 'stand':
            # Use the recorded first 0.3 s to leave the common standing pose.
            start = self.gaits[mode][:16]
            for row in self._blend(tail, start):
                self.plan.append(row)
            self.source, self.index = mode, 16
        else:
            self.source, self.index = 'stand', 0
        self.requested = mode
        self.events.append({'from': previous, 'to': mode, 'committed_frames': self.horizon, 'planned_frames': len(self.plan)})

    def _next(self):
        if self.plan:
            return self.plan.popleft()
        if self.source == 'stand':
            return self.stand.copy()
        clip = self.gaits[self.source]
        cycle_end = self.loops[self.source][1] if self.source in self.loops else len(clip)-1
        if self.index > cycle_end:
            # Select a matching interior phase, then explicitly bridge the seam.
            lo, hi = min(30, len(clip)//4), max(min(31, len(clip)//4+1), len(clip)-25)
            candidates = clip[lo:hi]
            tail = self.queue[-1]
            score = np.mean((candidates[:, :self.nj]-tail[:self.nj])**2, axis=1)
            score += .0025*np.mean((candidates[:, self.nj:2*self.nj]-tail[self.nj:2*self.nj])**2, axis=1)
            start = self.loops[self.source][0] if self.source in self.loops else lo+int(np.argmin(score))
            end = min(start+15, cycle_end)
            self.events.append({'kind':'loop_seam', 'skill':self.source, 'scheduler_step':self.steps,
                'from_frame':cycle_end, 'to_frame':start, 'bridge_end_frame':end,
                'joint_position_rms_rad':float(np.sqrt(np.mean((tail[:self.nj]-clip[start,:self.nj])**2))),
                'joint_velocity_rms_rad_s':float(np.sqrt(np.mean((tail[self.nj:2*self.nj]-clip[start,self.nj:2*self.nj])**2)))})
            self.plan.extend(self._blend(tail, clip[start:end+1]))
            self.index = end+1
            return self.plan.popleft()
        row = clip[self.index].copy()
        self.index += 1
        return row

    def window(self):
        return np.stack(self.queue)

    def advance(self):
        self.queue.popleft()
        self.queue.append(self._next())
        self.steps += 1
