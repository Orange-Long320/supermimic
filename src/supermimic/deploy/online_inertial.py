"""Bounded, causal IMU forward propagation from exported FAST-LIO corrections.

This is a diagnostic state bridge, not an independently validated estimator.
Acquisition timestamps are preserved; propagation stops at received IMU data.
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class InertialState:
    stamp: float
    position: np.ndarray
    rotation: Rotation
    velocity: np.ndarray
    gyro_bias: np.ndarray
    accel_bias: np.ndarray
    gravity: np.ndarray
    accel_scale: float


def parse_filter_state(values):
    """Schema: t,p3,q_xyzw4,v3,bg3,ba3,g3,acc_scale,imu_count,cloud_count."""
    values = np.asarray(values, dtype=float)
    if values.shape != (23,) or not np.isfinite(values).all():
        raise ValueError('Invalid filter-state schema or values')
    if abs(np.linalg.norm(values[4:8])-1.) > 1e-3 or not 9.3 < np.linalg.norm(values[17:20]) < 10.3:
        raise ValueError('Invalid filter rotation or gravity')
    if not .8 < values[20] < 1.2:
        raise ValueError('Unexpected accelerometer units/scale')
    return InertialState(float(values[0]), values[1:4].copy(), Rotation.from_quat(values[4:8]),
                         values[8:11].copy(), values[11:14].copy(), values[14:17].copy(),
                         values[17:20].copy(), float(values[20]))


def propagate_imu(state, samples, now, *, max_horizon=.2, max_gap=.03):
    """Samples are received [stamp, gyro_xyz, accel_xyz] in acquisition order.

    Midpoint gyro/acceleration integration matches FAST-LIO's acceleration
    normalization and bias convention. No future sample or clock retimestamping.
    """
    rows = np.asarray(samples, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != 7 or len(rows) < 2 or not np.isfinite(rows).all():
        raise ValueError('Missing or invalid IMU history')
    times = rows[:, 0]
    if not np.isfinite(now) or np.any(np.diff(times) <= 0) or times[-1] > now or state.stamp > now:
        raise ValueError('Nonmonotonic or future input')
    start = int(np.searchsorted(times, state.stamp, side='right'))-1
    if start < 0 or state.stamp > times[-1] or times[-1]-state.stamp > max_horizon:
        raise ValueError('IMU history does not cover bounded propagation')
    p, r, v, stamp = state.position.copy(), state.rotation, state.velocity.copy(), state.stamp
    for i in range(start+1, len(rows)):
        before, after = rows[i-1], rows[i]
        gap = after[0]-before[0]
        if gap > max_gap:
            raise ValueError('IMU acquisition gap')
        dt = after[0]-stamp
        if dt <= 0:
            continue
        fraction = (stamp-before[0])/gap
        at_start = before[1:]+fraction*(after[1:]-before[1:])
        average = .5*(at_start+after[1:])
        omega = average[:3]-state.gyro_bias
        specific_force = average[3:]*state.accel_scale-state.accel_bias
        mid_rotation = r*Rotation.from_rotvec(omega*dt*.5)
        acceleration = mid_rotation.apply(specific_force)+state.gravity
        p += v*dt+.5*acceleration*dt*dt
        v += acceleration*dt
        r = r*Rotation.from_rotvec(omega*dt)
        stamp = float(after[0])
    return replace(state, stamp=stamp, position=p, rotation=r, velocity=v), rows[-1, 1:4]-state.gyro_bias


def fit_level_floor(points, pelvis, *, seed=81):
    """Static initialization plane diagnostic in an already gravity-level frame."""
    points = np.asarray(points, dtype=float)
    pelvis = np.asarray(pelvis, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or pelvis.shape != (3,):
        raise ValueError('Invalid floor input')
    radial = np.linalg.norm(points[:, :2]-pelvis[:2], axis=1)
    cloud = points[np.isfinite(points).all(axis=1) & (radial>.4) & (radial<3.) & (points[:, 2]<pelvis[2]-.25)]
    if len(cloud) < 100:
        raise ValueError('Insufficient floor candidates')
    rng = np.random.default_rng(seed); best = np.zeros(len(cloud), dtype=bool)
    for _ in range(150):
        tri = cloud[rng.choice(len(cloud), 3, replace=False)]
        normal = np.cross(tri[1]-tri[0], tri[2]-tri[0]); norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal /= norm
        if abs(normal[2]) < np.cos(np.deg2rad(5)):
            continue
        mask = abs((cloud-tri[0])@normal) < .015
        if mask.sum() > best.sum(): best = mask
    if best.sum() < 80 or best.mean() < .1:
        raise ValueError('No supported near-level floor plane')
    chosen = cloud[best]; center = chosen.mean(axis=0)
    _, _, vh = np.linalg.svd(chosen-center, full_matrices=False)
    normal = vh[-1]*(1 if vh[-1, 2]>0 else -1)
    rms = float(np.sqrt(np.mean(((chosen-center)@normal)**2)))
    tilt = float(np.degrees(np.arccos(np.clip(normal[2], -1, 1))))
    z = float(center[2]-(normal[:2]@(pelvis[:2]-center[:2]))/normal[2])
    if rms > .012 or tilt > 5. or not .3 < pelvis[2]-z < 1.2:
        raise ValueError('Floor plane quality/height gate failed')
    return z, {'inliers': int(best.sum()), 'candidates': len(cloud), 'rms_m': rms, 'tilt_deg': tilt}
