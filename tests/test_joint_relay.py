import copy
import json
import unittest

from supermimic.deploy.joint_relay import GROUP_NAMES, JointRelayDecoder


class RelayTest(unittest.TestCase):
    def setUp(self):
        self.session = 'a' * 32
        self.decoder = JointRelayDecoder(self.session, list(set().union(*GROUP_NAMES.values())))
        self.packet = {'schema': 1, 'session': self.session, 'sequence': 1, 'published_ns': 1000,
                       'samples': [{'group': 'head', 'header_ns': 900, 'received_ns': 950,
                                    'source_count': 4, 'domain': 0,
                                    'joints': [[n, 0.0, 0.0, 0] for n in sorted(GROUP_NAMES['head'])]}]}

    def decode(self, packet):
        return self.decoder.decode(json.dumps(packet), 1010)

    def test_original_times_survive(self):
        row = self.decode(self.packet)[0][1]
        self.assertEqual((row['header_ns'], row['received_ns']), (900, 950))

    def test_duplicate_group_joint_and_future_rejected(self):
        for change in ('duplicate', 'wrong_group', 'future', 'nonfinite'):
            p = copy.deepcopy(self.packet)
            if change == 'duplicate': p['samples'][0]['joints'][1][0] = p['samples'][0]['joints'][0][0]
            if change == 'wrong_group': p['samples'][0]['joints'][0][0] = next(iter(GROUP_NAMES['waist']))
            if change == 'future': p['samples'][0]['received_ns'] = 1001
            if change == 'nonfinite': p['samples'][0]['joints'][0][1] = float('nan')
            with self.subTest(change=change), self.assertRaises(ValueError): self.decode(p)

    def test_sequence_gap_fails_then_recovers(self):
        self.decode(self.packet)
        p = copy.deepcopy(self.packet); p['sequence'] = 3
        with self.assertRaises(ValueError): self.decode(p)
        p['sequence'] = 4; p['samples'][0]['header_ns'] = 901; p['samples'][0]['source_count'] = 5
        self.assertEqual(len(self.decode(p)), 1)

    def test_stale_or_wrong_session_rejected(self):
        p = copy.deepcopy(self.packet); p['session'] = 'b'*32
        with self.assertRaises(ValueError): self.decode(p)
        with self.assertRaises(ValueError): self.decoder.decode(json.dumps(self.packet), 50_001_001)


if __name__ == '__main__': unittest.main()
