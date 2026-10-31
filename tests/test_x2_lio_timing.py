"""Protocol dimensions and CDR framing checks for recorded X2 packets."""
import importlib.util
from pathlib import Path
import struct
import unittest

spec = importlib.util.spec_from_file_location(
    'timing', Path(__file__).resolve().parents[1] / 'scripts/evaluation/audit_x2_lio_timing.py')
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)


class TimingProtocolTest(unittest.TestCase):
    def test_microsecond_offsets_become_nanoseconds(self):
        packets = []
        for sequence, micros, ring, offset in [(0, 200, 0, 332), (1, 1200, 95, 333)]:
            packet = bytearray(1200)
            packet[:4] = bytes.fromhex('55aa5aa5')
            packet[4:6] = sequence.to_bytes(2, 'big')
            packet[10:16] = (1).to_bytes(6, 'big')
            packet[16:20] = micros.to_bytes(4, 'big')
            struct.pack_into('>HH', packet, 32 + 12*ring, offset, 200)
            packets.append(packet)
        base, points, seqs = timing.reconstruct_points(b''.join(packets))
        self.assertEqual(base, 1_000_200_000)
        self.assertEqual(points, [(332_000, 0), (1_333_000, 95)])
        self.assertEqual(seqs, [0, 1])

    def test_invalid_packet_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            timing.reconstruct_points(bytes(1199))
        with self.assertRaises(ValueError):
            timing.reconstruct_points(bytes(1200))

    def test_native_message_header_layout(self):
        data = bytearray(b'\0\1\0\0')
        def integer(n):
            while len(data) % 4:
                data.append(0)
            data.extend(struct.pack('<I', n))
        for value in [1, 10, 4]:
            integer(value)
        data.extend(b'abc\0')
        for value in [3, 1, 20, 4]:
            integer(value)
        data.extend(b'abcd')
        integer(3)
        data.extend(b'xyz')
        parsed = timing.parse_raw_cdr(data)
        self.assertEqual(parsed['stamp'], 1_000_000_010)
        self.assertEqual(parsed['meas_stamp'], 1_000_000_020)
        self.assertEqual(parsed['frame'], 'abc')
        self.assertEqual(parsed['msop'], b'abcd')
        self.assertEqual(parsed['difop'], b'xyz')
        with self.assertRaises(ValueError):
            timing.parse_raw_cdr(data[:-1])


if __name__ == '__main__':
    unittest.main()
