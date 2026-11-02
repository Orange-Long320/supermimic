import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from supermimic.deploy.imu_calibration import body_bias_at_declared_rest, check_static_window


def test_three_axis_body_bias_rotates_with_sensor():
    rotations = Rotation.from_euler('xyz', [[0., 0., 0.], [.4, -.3, .8]]).as_matrix()
    bias = np.array([.12, -.08, .05])
    expected = np.einsum('nji,j->ni', rotations, [0., 0., 9.81])
    acc = expected + bias
    found = body_bias_at_declared_rest(acc, rotations)
    np.testing.assert_allclose(found, bias, atol=1e-12)
    corrected = np.einsum('nij,nj->ni', rotations, acc-found)
    np.testing.assert_allclose(corrected, np.tile([0., 0., 9.81], (2, 1)), atol=1e-12)


def test_motion_rejected_without_claiming_quiet_means_rest():
    stamps = np.arange(1001, dtype=np.int64)*2_000_000
    acc = np.tile([0., 0., 9.81], (len(stamps), 1))
    gyro = np.zeros_like(acc)
    check_static_window(stamps, acc, gyro, 2., .5, .08)
    gyro[400, 0] = .1
    with pytest.raises(ValueError, match='moving'):
        check_static_window(stamps, acc, gyro, 2., .5, .08)


def test_unobserved_initial_velocity_fails_closed():
    script = Path(__file__).resolve().parents[1] / 'scripts/deploy/publish_pelvis_packet.py'
    env = dict(os.environ, SM_MOTOR_OUTPUT='0')
    result = subprocess.run([sys.executable, str(script), '--profile', 'unused', '--out', 'unused'],
                            text=True, capture_output=True, env=env)
    assert result.returncode != 0
    assert 'Initial velocity is unobserved' in result.stderr
    result = subprocess.run([sys.executable, str(script), '--profile', 'unused', '--out', 'unused',
                             '--init-bias-seconds', '3'], text=True, capture_output=True, env=env)
    assert result.returncode != 0
    assert 'requires --assume-initial-rest' in result.stderr
