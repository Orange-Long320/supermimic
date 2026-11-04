import numpy as np
import pytest

from supermimic.deploy.imu_dead_reckoning import ImuDeadReckoning
from supermimic.deploy.imu_dead_reckoning import fit_static_initialization
from supermimic.deploy.imu_dead_reckoning import EncoderQuietWindow
from scipy.spatial.transform import Rotation


def test_yaw_is_integrated_from_gyro_without_an_orientation_measurement():
    estimator = ImuDeadReckoning(zupt=False, gravity_leveling=False, require_initialization=False)
    for i in range(1001):
        estimator.update(i*.002, [0., 0., .2], [0., 0., 9.81])
    np.testing.assert_allclose(estimator.rotation.as_rotvec(), [0., 0., .4], atol=1e-12)
    np.testing.assert_allclose(estimator.x, 0., atol=1e-12)


def test_acceleration_integrates_to_velocity_and_position_without_oracle_initialization():
    estimator = ImuDeadReckoning(zupt=False, gravity_leveling=False, require_initialization=False)
    for i in range(501):
        estimator.update(i*.002, [0., 0., 0.], [1., 0., 9.81])
    np.testing.assert_allclose(estimator.x, [.5, 0., 1., 0.], atol=1e-12)


def test_still_zupt_reduces_unobserved_velocity_under_its_stillness_hypothesis():
    estimator = ImuDeadReckoning(zupt=True, require_initialization=False)
    estimator.x[2:] = [1., -.3]
    for i in range(501):
        estimator.update(i*.002, [0., 0., 0.], [0., 0., 9.81])
    assert estimator.zupt_updates > 0
    assert np.linalg.norm(estimator.x[2:]) < 1e-3


@pytest.mark.parametrize('stamp', [0., -.002, .031, float('nan')])
def test_bad_timestamp_cannot_silently_advance_state(stamp):
    estimator = ImuDeadReckoning(require_initialization=False)
    estimator.update(0., [0., 0., 0.], [0., 0., 9.81])
    with pytest.raises(ValueError):
        estimator.update(stamp, [0., 0., 0.], [0., 0., 9.81])
    np.testing.assert_array_equal(estimator.x, np.zeros(4))


def static_samples():
    rotation = Rotation.from_euler('xyz', [.1, -.08, 0.])
    acc = rotation.inv().apply([0., 0., 9.81]).tolist()
    return [{'time_s': i*.002, 'gyro': [.01, -.02, .005], 'acceleration': acc,
             'joint_pos': [0., .2], 'joint_vel': [0., 0.]} for i in range(1501)]


def test_static_tilt_and_nonzero_gyro_bias_are_estimated_only_from_sensors():
    samples = static_samples()
    calibration = fit_static_initialization(samples, declared_stationary=True)
    np.testing.assert_allclose(calibration['gyro_bias'], [.01, -.02, .005], atol=1e-12)
    estimator = ImuDeadReckoning()
    estimator.initialize(calibration)
    for sample in samples:
        estimator.update(sample['time_s'], sample['gyro'], sample['acceleration'])
    np.testing.assert_allclose(estimator.rotation.as_euler('xyz'), [.1, -.08, 0.], atol=1e-12)
    np.testing.assert_allclose(estimator.x, 0., atol=1e-12)
    np.testing.assert_allclose(estimator.gyro, 0., atol=1e-12)
    assert not calibration['yaw_observable']
    with pytest.raises(ValueError, match='precede'):
        estimator.initialize(calibration)


def test_no_propagation_before_static_initialization():
    estimator = ImuDeadReckoning()
    assert not estimator.zupt and not estimator.gravity_leveling
    with pytest.raises(ValueError, match='not passed'):
        estimator.update(0., [0., 0., 0.], [0., 0., 9.81])


def test_static_thresholds_do_not_replace_an_independent_rest_declaration():
    with pytest.raises(ValueError, match='independently declared'):
        fit_static_initialization(static_samples())


@pytest.mark.parametrize('change', ['encoder_motion', 'short', 'gap', 'freefall'])
def test_reject_bad_static_window_instead_of_assuming_zero_velocity(change):
    samples = static_samples()
    if change == 'encoder_motion': samples[100]['joint_vel'] = [.1, 0.]
    elif change == 'short': samples = samples[:-1]
    elif change == 'gap': samples[100]['time_s'] = samples[99]['time_s']
    elif change == 'freefall': samples[100]['acceleration'] = [0., 0., 0.]
    with pytest.raises(ValueError):
        fit_static_initialization(samples, declared_stationary=True)


def test_moving_encoders_veto_imu_only_gravity_and_zero_velocity_updates():
    quiet = EncoderQuietWindow()
    estimator = ImuDeadReckoning(zupt=True, gravity_leveling=True, require_initialization=False)
    estimator.x[2] = .5
    for i in range(501):
        t = i*.002
        evidence = quiet.update(t, [t*.1, 0.], [.1, 0.])
        assert not evidence
        estimator.update(t, [0., 0., 0.], [0., 0., 9.81],
                         gravity_static_evidence=evidence, zupt_static_evidence=evidence)
    assert estimator.zupt_updates == 0
    np.testing.assert_allclose(estimator.x[2], .5)


def test_encoder_gate_requires_a_complete_quiet_window():
    quiet = EncoderQuietWindow()
    for i in range(100):
        assert not quiet.update(i*.002, [0., 0.], [0., 0.])
    assert quiet.update(.2, [0., 0.], [0., 0.])
    assert not quiet.update(.202, [.02, 0.], [0., 0.])


def test_interval_mean_gyro_is_not_averaged_with_the_previous_packet_again():
    estimator = ImuDeadReckoning(zupt=False, gravity_leveling=False,
                                  require_initialization=False, interval_average=True)
    estimator.update(0., [0., 0., 0.], [0., 0., 9.81])
    estimator.update(.002, [0., 0., 1.], [0., 0., 9.81])
    np.testing.assert_allclose(estimator.rotation.as_rotvec(), [0., 0., .002], atol=1e-12)


def test_interval_specific_force_uses_midpoint_attitude_without_gravity_leak():
    estimator = ImuDeadReckoning(zupt=False, gravity_leveling=False,
                                  require_initialization=False, interval_average=True)
    estimator.update(0., [1., 0., 0.], [0., 0., 9.81])
    for i in range(1, 501):
        midpoint = Rotation.from_rotvec([(i-.5)*.002, 0., 0.])
        estimator.update(i*.002, [1., 0., 0.], midpoint.inv().apply([0., 0., 9.81]))
    np.testing.assert_allclose(estimator.rotation.as_rotvec(), [1., 0., 0.], atol=1e-12)
    np.testing.assert_allclose(estimator.x, 0., atol=1e-12)
