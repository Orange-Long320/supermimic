"""Sensor-format checks; run on the server with ROS Humble sourced."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

try:
    from sensor_msgs.msg import PointCloud2, PointField
except ImportError:
    PointCloud2 = None

spec = importlib.util.spec_from_file_location(
    'x2_fastlio_bag', Path(__file__).resolve().parents[1] / 'scripts/evaluation/prepare_x2_fastlio_bag.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class ExtrinsicDirectionTest(unittest.TestCase):
    def test_rotated_translation_is_inverted_into_imu_frame(self):
        t, r = adapter.lidar_pose_in_imu([1, 0, 0], [0, 0, np.sqrt(.5), np.sqrt(.5)])
        np.testing.assert_allclose(t, [0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(np.array(r).reshape(3, 3) @ [1, 0, 0],
                                   [0, -1, 0], atol=1e-12)

    def test_invalid_quaternion_is_rejected(self):
        with self.assertRaises(ValueError):
            adapter.lidar_pose_in_imu([0, 0, 0], [0, 0, 0, 0])


@unittest.skipIf(PointCloud2 is None, 'ROS sensor_msgs unavailable')
class PointLayoutTest(unittest.TestCase):
    def cloud(self):
        # Deliberately use big-endian, non-contiguous fields and unsorted times.
        dtype = np.dtype({'names': ['x', 'y', 'z', 'intensity', 'ring', 'timestamp'],
            'formats': ['>f4', '>f4', '>f4', '>f4', '>u2', '>f8'],
            'offsets': [0, 4, 8, 16, 20, 24], 'itemsize': 32})
        a = np.zeros(2, dtype=dtype)
        a['x'] = [9, 3]
        a['ring'] = [95, 0]
        a['timestamp'] = [95_000_000, 332_000]
        m = PointCloud2()
        m.height, m.width, m.point_step, m.row_step = 1, 2, 32, 64
        m.is_bigendian = True
        m.fields = [PointField(name=n, offset=dtype.fields[n][1],
            datatype=PointField.UINT16 if n == 'ring' else
            PointField.FLOAT64 if n == 'timestamp' else PointField.FLOAT32,
            count=1) for n in dtype.names]
        m.data = a.tobytes()
        return m

    def test_point_association_and_scan_end_survive_conversion(self):
        m, _ = adapter.convert_cloud(self.cloud(), 1e-9)
        x = np.ndarray(2, dtype='<f4', buffer=m.data, strides=(24,))
        t = np.ndarray(2, dtype='<f4', buffer=m.data, offset=16, strides=(24,))
        ring = np.ndarray(2, dtype='<u2', buffer=m.data, offset=20, strides=(24,))
        np.testing.assert_array_equal(x, [3, 9])
        np.testing.assert_array_equal(ring, [0, 95])
        np.testing.assert_allclose(t, [.000332, .095], rtol=1e-6)

    def test_wrong_time_unit_is_rejected(self):
        with self.assertRaises(ValueError):
            adapter.convert_cloud(self.cloud(), 1.)


if __name__ == '__main__':
    unittest.main()
