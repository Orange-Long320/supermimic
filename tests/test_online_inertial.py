import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from supermimic.deploy.online_inertial import parse_filter_state, propagate_imu, fit_level_floor


class OnlineInertialTest(unittest.TestCase):
    def state(self, rotation=None):
        rotation = rotation or Rotation.identity()
        return parse_filter_state([1., 0., 0., 0., *rotation.as_quat(), 0., 0., 0.,
                                   .01, -.02, .03, .02, -.01, .04, 0., 0., -9.81, 1.01, 40, 2])

    def samples(self, state, acceleration=None, omega=None):
        acceleration = np.zeros(3) if acceleration is None else np.asarray(acceleration)
        omega = np.zeros(3) if omega is None else np.asarray(omega)
        acc = (state.rotation.inv().apply(acceleration-state.gravity)+state.accel_bias)/state.accel_scale
        return [[t, *(omega+state.gyro_bias), *acc] for t in np.linspace(1., 1.1, 21)]

    def test_tilted_static_bias_and_scale(self):
        s = self.state(Rotation.from_euler('xyz', [.3, -.5, .8]))
        out, omega = propagate_imu(s, self.samples(s), 1.12)
        np.testing.assert_allclose(out.position, 0., atol=1e-12)
        np.testing.assert_allclose(out.velocity, 0., atol=1e-12)
        np.testing.assert_allclose(omega, 0., atol=1e-12)
        self.assertAlmostEqual(out.stamp, 1.1)

    def test_constant_acceleration(self):
        s = self.state(); out, _ = propagate_imu(s, self.samples(s, [1, 0, 0]), 1.1)
        np.testing.assert_allclose(out.position, [.005, 0, 0], atol=1e-12)
        np.testing.assert_allclose(out.velocity, [.1, 0, 0], atol=1e-12)

    def test_constant_rotation(self):
        s = self.state(); out, _ = propagate_imu(s, self.samples(s, omega=[0, 0, 1]), 1.1)
        np.testing.assert_allclose(out.rotation.as_rotvec(), [0, 0, .1], atol=1e-12)

    def test_rejects_future_gap_and_missing_history(self):
        s = self.state(); rows = self.samples(s)
        for value, now in ((rows, 1.09), (rows[1:], 1.1), ([rows[0], rows[-1]], 1.1), (rows[::-1], 1.1)):
            with self.assertRaises(ValueError): propagate_imu(s, value, now)
        with self.assertRaises(ValueError): parse_filter_state([0.]*23)

    def test_floor_fit_and_wall_rejection(self):
        rng = np.random.default_rng(8)
        xy = rng.uniform(-2, 2, (800, 2)); z = -.9+rng.normal(0, .002, (800, 1))
        floor, quality = fit_level_floor(np.column_stack([xy, z]), [0, 0, -.23])
        self.assertAlmostEqual(floor, -.9, places=3)
        self.assertLess(quality['rms_m'], .003)
        wall = np.column_stack([np.ones(800), rng.uniform(-2, 2, 800), rng.uniform(-2, 1, 800)])
        with self.assertRaises(ValueError): fit_level_floor(wall, [0, 0, -.23])


if __name__ == '__main__': unittest.main()
