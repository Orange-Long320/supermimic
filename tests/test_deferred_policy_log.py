import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from supermimic.deploy.deferred_policy_log import DeferredPolicyLog


class DeferredLogTests(unittest.TestCase):
    def test_no_disk_io_during_capture_and_mutation_does_not_change_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'frames.jsonl'
            log = DeferredPolicyLog(path, ['a', 'b'], capacity=2, observation_size=3, goal_size=2)
            q = np.array([.1, .2])
            packet = dict(elapsed=.02, state_stamp=123., session_id='one', joint_position=q,
                          position=[0., 0., .65], observation=[1., 2., 3.], goal=[4., 5.],
                          action=[6., 7.], raw_pd_target=[8., 9.], measured_outside_file_bounds=['b'])
            with log:
                log.append(**packet)
                q[:] = 42
                log.append(**packet)
                self.assertFalse(path.exists())
                with self.assertRaisesRegex(RuntimeError, 'capacity'): log.append(**packet)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(rows[0]['joint_position'], [.1, .2])
            self.assertEqual(rows[1]['joint_position'], [42., 42.])
            self.assertEqual(rows[0]['measured_outside_file_bounds'], ['b'])
            self.assertFalse(rows[0]['eligible_for_control'])
            self.assertEqual(log.count, 2)


if __name__ == '__main__': unittest.main()
