"""Measured root translation for an experimental Idle damping input."""
from collections import deque

import numpy as np


class RootVelocityEstimate:
    """Least-squares velocity over six received odometry samples (~0.1 s)."""
    def __init__(self):
        self.samples = deque(maxlen=6)

    def update(self, timestamp, position):
        p = np.asarray(position, dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all() or not np.isfinite(timestamp):
            raise ValueError('Invalid odometry for Idle velocity estimate')
        if self.samples and timestamp <= self.samples[-1][0]:
            if timestamp == self.samples[-1][0] and np.array_equal(p, self.samples[-1][1]):
                return self.velocity()
            raise ValueError('Nonmonotonic odometry for Idle velocity estimate')
        self.samples.append((float(timestamp), p.copy()))
        return self.velocity()

    def velocity(self):
        if len(self.samples) < 2:
            return np.zeros(3)
        t = np.array([x[0] for x in self.samples])-self.samples[-1][0]
        t -= t.mean()
        positions = np.array([x[1] for x in self.samples])
        v = np.sum(t[:,None]*(positions-positions.mean(axis=0)), axis=0)/np.sum(t*t)
        v[2] = 0.
        return v
