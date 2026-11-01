import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from supermimic.deploy.online_inertial import parse_filter_state, propagate_imu, fit_level_floor, CachedImuPropagation, grid_tick_pending
from dataclasses import replace


class OnlineInertialTest(unittest.TestCase):
    def test_grid_tick_retries_until_success_without_repeating(self):
        self.assertTrue(grid_tick_pending(25_000_000, None))
        self.assertTrue(grid_tick_pending(25_000_000, 0))
        self.assertFalse(grid_tick_pending(39_999_999, 20_000_000))
        self.assertTrue(grid_tick_pending(40_000_000, 20_000_000))

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

    def test_cached_integration_matches_full_and_new_bias_correction(self):
        state=self.state();rows=self.samples(state, [1., .2, -.1], [.1, -.2, .3]);cache=CachedImuPropagation()
        for count in (5, 9, 13, 21):
            actual,gyro=cache.advance(state,rows[:count],rows[count-1][0])
            expected,expected_gyro=propagate_imu(state,rows[:count],rows[count-1][0])
            np.testing.assert_allclose(actual.position,expected.position,atol=1e-14)
            np.testing.assert_allclose(actual.velocity,expected.velocity,atol=1e-14)
            np.testing.assert_allclose(actual.rotation.as_matrix(),expected.rotation.as_matrix(),atol=1e-14)
            np.testing.assert_allclose(gyro,expected_gyro,atol=1e-14)
        correction=replace(state,stamp=1.05,position=np.ones(3),gyro_bias=np.zeros(3))
        actual,_=cache.advance(correction,rows,1.1);expected,_=propagate_imu(correction,rows,1.1)
        np.testing.assert_allclose(actual.position,expected.position,atol=1e-14)
        np.testing.assert_allclose(actual.rotation.as_matrix(),expected.rotation.as_matrix(),atol=1e-14)
        with self.assertRaisesRegex(ValueError,'stale'):cache.advance(correction,rows,1.26)

    def test_grid_time_integrates_partial_interval_without_extrapolation(self):
        state=self.state(); rows=self.samples(state, [1., 0., 0.], [0., 0., 1.])
        # Separate acceleration and rotation cases retain exact analytic answers.
        linear=self.samples(state,[1.,0.,0.])
        out,_=propagate_imu(state,linear,1.12,until=1.073)
        np.testing.assert_allclose(out.position,[.5*.073**2,0.,0.],atol=1e-12)
        np.testing.assert_allclose(out.velocity,[.073,0.,0.],atol=1e-12)
        out,_=propagate_imu(state,rows,1.12,until=1.073)
        np.testing.assert_allclose(out.rotation.as_rotvec(),[0.,0.,.073],atol=1e-12)
        self.assertEqual(out.stamp,1.073)
        for until in (1.101,.99):
            with self.assertRaisesRegex(ValueError,'bracket'):
                propagate_imu(state,rows,1.12,until=until)
        with self.assertRaisesRegex(ValueError,'gap'):
            propagate_imu(state,[rows[0],rows[-1]],1.12,until=1.073)

    def test_partial_gyro_interpolation_and_grid_cache(self):
        state=self.state(); rows=np.asarray(self.samples(state))
        rows[:,3]+=2*(rows[:,0]-1.)
        out,gyro=propagate_imu(state,rows,1.12,until=1.073)
        np.testing.assert_allclose(gyro,[0.,0.,.146],atol=1e-12)
        np.testing.assert_allclose(out.rotation.as_rotvec(),[0.,0.,.073**2],atol=1e-12)
        cache=CachedImuPropagation()
        for until in (1.013,1.033,1.053,1.073,1.093):
            actual,_=cache.advance(state,rows,1.12,until=until)
            expected,_=propagate_imu(state,rows,1.12,until=until)
            np.testing.assert_allclose(actual.position,expected.position,atol=1e-12)
            np.testing.assert_allclose(actual.rotation.as_matrix(),expected.rotation.as_matrix(),atol=1e-12)
        with self.assertRaisesRegex(ValueError,'stale'):
            cache.advance(state,rows,1.21,until=1.093)


if __name__ == '__main__': unittest.main()
