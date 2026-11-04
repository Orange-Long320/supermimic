import numpy as np
import pytest

from supermimic.deploy.imu_dead_reckoning import ImuDeadReckoning


def test_yaw_is_integrated_from_gyro_without_an_orientation_measurement():
    estimator = ImuDeadReckoning(zupt=False, gravity_leveling=False)
    for i in range(1001):
        estimator.update(i*.002, [0., 0., .2], [0., 0., 9.81])
    np.testing.assert_allclose(estimator.rotation.as_rotvec(), [0., 0., .4], atol=1e-12)
    np.testing.assert_allclose(estimator.x, 0., atol=1e-12)


def test_acceleration_integrates_to_velocity_and_position_without_oracle_initialization():
    estimator = ImuDeadReckoning(zupt=False, gravity_leveling=False)
    for i in range(501):
        estimator.update(i*.002, [0., 0., 0.], [1., 0., 9.81])
    np.testing.assert_allclose(estimator.x, [.5, 0., 1., 0.], atol=1e-12)


def test_still_zupt_reduces_unobserved_velocity_under_its_stillness_hypothesis():
    estimator = ImuDeadReckoning()
    estimator.x[2:] = [1., -.3]
    for i in range(501):
        estimator.update(i*.002, [0., 0., 0.], [0., 0., 9.81])
    assert estimator.zupt_updates > 0
    assert np.linalg.norm(estimator.x[2:]) < 1e-3


@pytest.mark.parametrize('stamp', [0., -.002, .031, float('nan')])
def test_bad_timestamp_cannot_silently_advance_state(stamp):
    estimator = ImuDeadReckoning()
    estimator.update(0., [0., 0., 0.], [0., 0., 9.81])
    with pytest.raises(ValueError):
        estimator.update(stamp, [0., 0., 0.], [0., 0., 9.81])
    np.testing.assert_array_equal(estimator.x, np.zeros(4))
