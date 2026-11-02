"""Conditional static IMU calibration helpers; thresholds cannot prove rest."""
import numpy as np


def check_static_window(stamps, acc, gyro, seconds, accel_tol, gyro_tol):
    stamps = np.asarray(stamps, dtype=np.int64)
    acc, gyro = np.asarray(acc), np.asarray(gyro)
    if (len(stamps) < 100 or stamps[-1] - stamps[0] < (seconds - .1) * 1e9
            or np.any(np.diff(stamps) <= 0) or np.any(np.diff(stamps) > 30_000_000)
            or not np.isfinite(acc).all() or not np.isfinite(gyro).all()
            or np.any(abs(np.linalg.norm(acc, axis=1)-9.81) > accel_tol)
            or np.any(np.linalg.norm(gyro, axis=1) > gyro_tol)):
        raise ValueError('Calibration window is moving, invalid, or has an IMU timestamp gap')


def body_bias_at_declared_rest(acc, sensor_to_world):
    """Assumes independently declared rest, known gravity and correct attitude."""
    expected = np.einsum('nji,j->ni', np.asarray(sensor_to_world), [0., 0., 9.81])
    return np.mean(np.asarray(acc) - expected, axis=0)
