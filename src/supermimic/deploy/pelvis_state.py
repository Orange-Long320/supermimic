"""Pose algebra for offline, logging-only X2 state conversion; no ROS or actuation."""
import numpy as np


def gravity_level_rotation(up):
    """Minimal rotation taking static specific-force direction to world +Z."""
    v=np.asarray(up,dtype=float)
    if v.shape!=(3,) or not np.isfinite(v).all() or np.linalg.norm(v)<1e-8:
        raise ValueError('Invalid gravity direction')
    v=v/np.linalg.norm(v); target=np.array([0.,0.,1.])
    c=float(v@target)
    if c < -1+1e-10:return np.diag([1.,-1.,-1.])
    cross=np.cross(v,target)
    k=np.array([[0.,-cross[2],cross[1]],[cross[2],0.,-cross[0]],[-cross[1],cross[0],0.]])
    return np.eye(3)+k+k@k/(1+c)


def pelvis_from_imu(p_world_imu, r_world_imu, p_pelvis_imu, r_pelvis_imu):
    """T_WP = T_WI inv(T_PI), using measured-joint FK for the moving mount."""
    r = np.asarray(r_world_imu) @ np.asarray(r_pelvis_imu).T
    p = np.asarray(p_world_imu) - r @ np.asarray(p_pelvis_imu)
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        raise ValueError('Nonfinite pelvis transform')
    return p, r


def pelvis_angular_velocity(omega_imu, r_pelvis_imu, omega_mount_in_pelvis):
    """Remove waist-chain motion from an IMU attached above the moving waist."""
    return np.asarray(r_pelvis_imu) @ np.asarray(omega_imu) - np.asarray(omega_mount_in_pelvis)


def latest_causal(rows, now_ns, *, max_age_s, stamp_key='header_ns',
                  at_ns=None, received_times=None):
    """Select only received, nonfuture data; reject old acquisition/header times."""
    received = (np.asarray([r['received_ns'] for r in rows], dtype=np.int64)
                if received_times is None else received_times)
    at_ns = now_ns if at_ns is None else min(at_ns, now_ns)
    i = int(np.searchsorted(received, now_ns, side='right'))-1
    while i >= 0 and rows[i][stamp_key] > at_ns:
        i -= 1
    if i < 0 or (at_ns-rows[i][stamp_key])*1e-9 > max_age_s:
        raise ValueError('Missing, future or stale sample')
    return rows[i]
