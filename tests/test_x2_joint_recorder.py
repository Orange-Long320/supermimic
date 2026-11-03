"""Offline checks for interrupted, read-only joint recording."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np


PATH = Path(__file__).resolve().parents[1] / 'scripts/deploy/record_x2_joint_states.py'
spec = importlib.util.spec_from_file_location('x2_joint_recorder', PATH)
recorder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder)


class RecorderTests(unittest.TestCase):
    def test_rates_use_actual_sample_intervals_and_single_sample_is_valid(self):
        rows = [{'t': 1., 'q': [0.]}, {'t': 1.1, 'q': [.2]}]
        result = recorder.summarize_rows(rows, ['a'], np.array([0.]), 'deadline')
        self.assertAlmostEqual(result['max_abs_rate_rad_s'], 2.)
        self.assertIsNone(recorder.summarize_rows(rows[:1], ['a'], np.array([0.]), 'stop')['max_abs_rate_rad_s'])

    def test_spin_failure_persists_samples_and_propagates_failure(self):
        names = ['joint%d' % i for i in range(31)]
        class FakeNode:
            def __init__(self, *args, **kwargs): self.callbacks = []
            def create_subscription(self, msg, topic, callback, qos): self.callbacks.append(callback)
            def destroy_node(self): pass
        class FakeExecutor:
            def add_node(self, node): self.node = node; self.count = 0
            def spin_once(self, timeout_sec):
                self.count += 1
                if self.count == 1:
                    msg = types.SimpleNamespace(joints=[types.SimpleNamespace(name=n, position=.1) for n in names])
                    self.node.callbacks[0](msg)
                    return
                raise RuntimeError('context invalid')
        modules = {name: types.ModuleType(name) for name in
                   ('rclpy', 'rclpy.node', 'rclpy.executors', 'rclpy.qos', 'aimdk_msgs', 'aimdk_msgs.msg')}
        modules['rclpy'].init = lambda: None
        modules['rclpy'].try_shutdown = lambda: None
        modules['rclpy.node'].Node = FakeNode
        modules['rclpy.executors'].SingleThreadedExecutor = FakeExecutor
        qos = modules['rclpy.qos']
        qos.QoSProfile = lambda **kw: None
        qos.ReliabilityPolicy = types.SimpleNamespace(BEST_EFFORT=1)
        qos.DurabilityPolicy = types.SimpleNamespace(VOLATILE=1)
        modules['aimdk_msgs.msg'].JointStateArray = object
        with tempfile.TemporaryDirectory() as tmp:
            startup, out = Path(tmp)/'startup.json', Path(tmp)/'out.json'
            startup.write_text(json.dumps({'joint_names': names, 'position': [0.]*31}))
            argv = ['record', '--seconds', '10', '--startup', str(startup), '--out', str(out)]
            with patch.dict(sys.modules, modules), patch.object(sys, 'argv', argv):
                with self.assertRaisesRegex(RuntimeError, 'context invalid'):
                    recorder.main()
            result = json.loads(out.read_text())
            self.assertEqual(result['summary']['samples'], 1)
            self.assertIn('context invalid', result['summary']['termination'])
            self.assertAlmostEqual(result['rows'][0]['q'][0], .1)


if __name__ == '__main__':
    unittest.main()
