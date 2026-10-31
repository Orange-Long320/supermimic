"""Bounded diagnostic recording; JSON and disk writes occur after collection.

This stores float64 copies, not references to mutable policy buffers. It is not
an actuator log or a way to hide dropped observations: overflow stops the probe.
"""
import json
from pathlib import Path
import time

import numpy as np


class DeferredPolicyLog:
    def __init__(self, path, names, *, capacity, observation_size, goal_size):
        self.path = Path(path)
        self.names = tuple(names)
        if self.path.exists() or capacity < 1 or min(observation_size, goal_size) < 1:
            raise ValueError('New output and positive bounded log dimensions required')
        sizes = dict(joint_position=len(names), position=3, observation=observation_size,
                     goal=goal_size, action=len(names), raw_pd_target=len(names))
        self.columns = {}; offset = 2
        for key, size in sizes.items():
            self.columns[key] = slice(offset, offset+size); offset += size
        self.rows = np.empty((capacity, offset), dtype=np.float64)
        self.outside = np.empty((capacity, len(names)), dtype=bool)
        self.indices = {name: i for i, name in enumerate(names)}
        self.count = 0; self.session = None
        self.capture_finished = None; self.export_seconds = None

    def append(self, *, elapsed, state_stamp, session_id, measured_outside_file_bounds, **values):
        if self.count == len(self.rows):
            raise RuntimeError('Diagnostic recording capacity exhausted; no records discarded')
        if self.session is not None and session_id != self.session:
            raise RuntimeError('Diagnostic recording session changed')
        if set(values) != set(self.columns):
            raise ValueError('Diagnostic recording fields differ')
        row = self.rows[self.count]; row[:2] = elapsed, state_stamp
        for key, columns in self.columns.items():
            if np.shape(values[key]) != (columns.stop-columns.start,):
                raise ValueError('Diagnostic recording shape: '+key)
            row[columns] = values[key]
        self.outside[self.count].fill(False)
        for name in measured_outside_file_bounds:
            self.outside[self.count, self.indices[name]] = True
        self.session = session_id; self.count += 1

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.capture_finished = time.monotonic()
        with self.path.open('x') as output:
            for i, row in enumerate(self.rows[:self.count]):
                record = dict(elapsed=float(row[0]), state_stamp=float(row[1]), session_id=self.session,
                    measured_outside_file_bounds=[n for n, bad in zip(self.names, self.outside[i]) if bad],
                    eligible_for_control=False, previous_action_scope='unexecuted shadow output')
                record.update({key: row[columns].tolist() for key, columns in self.columns.items()})
                output.write(json.dumps(record, allow_nan=False)+'\n')
        self.export_seconds = time.monotonic()-self.capture_finished
