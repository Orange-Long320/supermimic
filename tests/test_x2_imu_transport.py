"""Evidence attribution tests; synthetic metadata only, no ROS or actuation."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

script = Path(__file__).resolve().parents[1]/'scripts/evaluation/audit_x2_imu_transport.py'
spec = importlib.util.spec_from_file_location('imu_transport_audit', script)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AttributionTests(unittest.TestCase):
    def report(self, lost=False, unsupported=False):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'input.csv'
            with path.open('w') as stream:
                writer = csv.writer(stream)
                writer.writerow(('reader','header_ns','callback_ns','rmw_source_ns','rmw_received_ns',
                                 'publication_sequence','reception_sequence','gid'))
                for reader in ('native_best_effort','native_reliable','relay_reliable'):
                    times = [0, 5, 10, 15, 20, 25, 30, 35, 40] if lost else [0, 5, 40, 45]
                    for i,t in enumerate(times):
                        if lost and reader != 'native_reliable' and 5 < t < 40:
                            continue
                        seq = audit.UNSUPPORTED if unsupported else i+1
                        ns = 1_000_000_000+t*1_000_000
                        writer.writerow((reader,ns,ns+100,ns,ns+50,seq,i+1,'one-native-publisher'))
            return audit.analyze(path)

    def test_reader_loss_is_proven_by_independent_received_samples(self):
        r = self.report(lost=True)
        gap = r['readers']['native_best_effort']['gaps_over_30ms'][0]
        self.assertIn('reader missed', gap['attribution'])
        self.assertEqual(gap['publication_sequence_delta'], 7)
        self.assertEqual(len(gap['intermediate_headers_by_reader']['native_reliable']), 6)

    def test_consecutive_writer_ids_identify_published_header_gap(self):
        r = self.report()
        gap = r['readers']['native_reliable']['gaps_over_30ms'][0]
        self.assertEqual(gap['publication_sequence_delta'], 1)
        self.assertIn('consecutive publisher', gap['attribution'])

    def test_unsupported_writer_ids_leave_source_attribution_unresolved(self):
        r = self.report(unsupported=True)
        gap = r['readers']['native_reliable']['gaps_over_30ms'][0]
        self.assertIsNone(gap['publication_sequence_delta'])
        self.assertIn('unresolved', gap['attribution'])


if __name__ == '__main__':
    unittest.main()
