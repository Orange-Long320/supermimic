"""Run the recorded support-foot fusion on a native-rate IMU recording.

Same estimator as ``eval_recorded_imu_fusion.py`` (contact-Jacobian foot
velocity, height/vertical-speed support gate, innovation-gated slip guard), but
driven at whatever rate the IMU actually arrived instead of a fixed 50 Hz grid.

Three constants that were implicitly tied to a 20 ms step are re-expressed in
seconds so the comparison isolates sample rate rather than retuning: the
foot-velocity low-pass keeps its time constant, the minimum support dwell keeps
its seconds, and the process noise scales with dt. Everything else is copied.

Two variants are produced from the SAME recording: ``native`` uses every IMU
sample, ``snapshot_50hz`` keeps only the latest sample at each 20 ms tick, which
is what the deployed state chain forwards.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

G = 9.81
TAU_FOOT_VEL = .02 / np.log(2.)   # original: 0.5/0.5 EMA at 20 ms
Q_BASE = np.diag([1e-7, 1e-7, 4e-5, 4e-5, 1e-7, 1e-7])
R_BASE = np.eye(2) * (.10 ** 2)
DT_REF = .02


def stamp_seconds(record):
    return record['stamp'][0] + record['stamp'][1] * 1e-9


def dedupe(ts, *arrays):
    order = np.argsort(ts, kind='stable')
    ts = ts[order]
    arrays = [a[order] for a in arrays]
    keep = np.r_[True, np.diff(ts) > 0.]
    return (ts[keep], *[a[keep] for a in arrays])


def load_stream(path, fields):
    rows = [json.loads(s) for s in path.open()]
    ts = np.array([stamp_seconds(r) for r in rows])
    values = {f: np.array([r[f] for r in rows]) for f in fields}
    return ts, values


def load_joint_timeline(path, names, recv_to_time):
    """Assemble the 31-joint vector at each leg-group snapshot, keyed by sensor time.

    One row per leg message: the leg group drives foot kinematics, and the other
    groups are at most ~1.3 ms apart from it in the same publishing cycle.
    """
    index = {n: i for i, n in enumerate(names)}
    latest, times, rows = {}, [], []
    with path.open() as source:
        for line in source:
            r = json.loads(line)
            latest[r['group']] = r
            if r['group'] != 'leg' or len(latest) < 4:
                continue
            position = np.zeros(len(names))
            velocity = np.zeros(len(names))
            seen = np.zeros(len(names), dtype=bool)
            for record in latest.values():
                for name, q, qd in record['joints']:
                    column = index.get(name)
                    if column is None:
                        continue
                    position[column] = q
                    velocity[column] = qd
                    seen[column] = True
            if not seen.all():
                continue
            times.append(recv_to_time(r['recv']))
            rows.append(np.concatenate([position, velocity]))
    times = np.array(times)
    rows = np.array(rows)
    order = np.argsort(times, kind='stable')
    return times[order], rows[order]


def support_latest(times, values, query):
    """Latest sample at or before each query time; the recorded high rate makes this cheap."""
    idx = np.searchsorted(times, query, side='right') - 1
    if idx.min() < 0:
        raise ValueError('Query precedes the first sample')
    return values[idx]


def foot_geometry(model, data, qpos_addr, dof_addr, feet, quats, joints, joint_vel, gyro):
    """Foot contact-point positions and their velocities from the contact Jacobian."""
    spheres = [[j for j in range(model.ngeom) if model.geom_bodyid[j] == body
                and model.geom_type[j] == mujoco.mjtGeom.mjGEOM_SPHERE and model.geom_contype[j]]
               for body in feet]
    if not all(spheres):
        raise ValueError('Missing foot collision spheres')
    rel = np.zeros((len(quats), 2, 3))
    foot_v = np.zeros_like(rel)
    for i in range(len(quats)):
        data.qpos[:7] = np.r_[[0., 0., 0.], quats[i]]
        data.qpos[qpos_addr] = joints[i]
        data.qvel[:] = 0.
        data.qvel[3:6] = gyro[i]
        data.qvel[dof_addr] = joint_vel[i]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        for foot, geoms in enumerate(spheres):
            sphere = min(geoms, key=lambda j: data.geom_xpos[j, 2] - model.geom_size[j, 0])
            point = data.geom_xpos[sphere].copy()
            point[2] -= model.geom_size[sphere, 0]
            rel[i, foot] = point
            jac = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jac, None, point, feet[foot])
            foot_v[i, foot] = jac @ data.qvel
    return rel, foot_v


def fuse(times, acc_world, rel, foot_v, bias_scale, r_scale=1., increments=None):
    """Identical filter to the recorded-fusion evaluator, driven by variable dt."""
    n = len(times)
    x = np.zeros(6)
    P = np.diag([.01, .01, .1, .1, .01, .01])
    est = np.zeros((n, 6))
    imu = np.zeros((n, 4))
    H = np.zeros((2, 6))
    H[0, 2] = H[1, 3] = 1
    identity = np.eye(6)
    accepted = switches = rejected_innovation = reduced_confidence = 0
    selected = np.full(n, -1, dtype=int)
    last = None
    last_switch_time = None
    for i in range(1, n):
        dt = times[i] - times[i - 1]
        if dt <= 0.:
            est[i] = x
            continue
        F = np.eye(6)
        F[0, 2] = F[1, 3] = dt
        F[0, 4] = F[1, 5] = -.5 * dt ** 2
        F[2, 4] = F[3, 5] = -dt
        B = np.zeros((6, 2))
        B[0, 0] = B[1, 1] = .5 * dt ** 2
        B[2, 0] = B[3, 1] = dt
        Q = Q_BASE * (dt / DT_REF)
        if increments is None:
            u = acc_world[i, :2]
            x = F @ x + B @ u
            P = F @ P @ F.T + Q
            imu[i, :2] = imu[i - 1, :2] + imu[i - 1, 2:] * dt + .5 * u * dt ** 2
            imu[i, 2:] = imu[i - 1, 2:] + u * dt
        else:
            # Native-rate strapdown collapsed into one increment per update epoch.
            delta_p, delta_v = increments[i]
            x = F @ x
            x[0:2] += delta_p
            x[2:4] += delta_v
            P = F @ P @ F.T + Q
            imu[i, :2] = imu[i - 1, :2] + imu[i - 1, 2:] * dt + .5 * delta_v * dt
            imu[i, 2:] = imu[i - 1, 2:] + delta_v
        low = np.min(rel[i, :, 2])
        candidates = []
        for foot in range(2):
            measurement = -foot_v[i, foot, :2]
            if (rel[i, foot, 2] - low <= .015 and abs(foot_v[i, foot, 2]) <= .12
                    and np.linalg.norm(measurement) <= 1.5):
                candidates.append((np.linalg.norm(measurement - H @ x), foot, measurement))
        if candidates:
            diff, foot, z = min(candidates, key=lambda c: c[0])
            switch_too_soon = (last is not None and foot != last
                               and last_switch_time is not None
                               and times[i] - last_switch_time < bias_scale['dwell'])
            if diff < .6 and not switch_too_soon:
                innovation = z - H @ x
                effective_r = R_BASE * r_scale
                if bias_scale['slip_guard']:
                    S0 = H @ P @ H.T + effective_r
                    nis = float(innovation @ np.linalg.solve(S0, innovation))
                    if nis > 9.21:
                        rejected_innovation += 1
                        est[i] = x
                        continue
                    disagreement = (np.linalg.norm(candidates[0][2] - candidates[1][2])
                                    if len(candidates) == 2 else 0.)
                    scale = min(16., max(1., (diff / .1) ** 2, (disagreement / .15) ** 2))
                    if scale > 1.01:
                        reduced_confidence += 1
                    effective_r = R_BASE * r_scale * scale
                S = H @ P @ H.T + effective_r
                K = np.linalg.solve(S, (P @ H.T).T).T
                x = x + K @ innovation
                P = (identity - K @ H) @ P @ (identity - K @ H).T + K @ effective_r @ K.T
                accepted += 1
                if last is not None and foot != last:
                    switches += 1
                    last_switch_time = times[i]
                last = foot
                selected[i] = foot
        est[i] = x
    return est, imu, {'accepted': accepted, 'switches': switches,
                      'rejected_innovation': rejected_innovation,
                      'reduced_confidence': reduced_confidence}


def metrics(error):
    return {'p50_m': float(np.percentile(error, 50)), 'p95_m': float(np.percentile(error, 95)),
            'max_m': float(error.max()), 'final_m': float(error[-1])}


def preintegrate(times, acc_world, epochs):
    """Cumulative strapdown at native rate, differenced onto the update epochs."""
    v = np.zeros((len(times), 3))
    p = np.zeros((len(times), 3))
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        half = .5 * (acc_world[i - 1] + acc_world[i]) * dt
        v_next = v[i - 1] + half
        p[i] = p[i - 1] + .5 * (v[i - 1] + v_next) * dt
        v[i] = v_next
    idx = np.searchsorted(times, epochs, side='left')
    idx = np.clip(idx, 0, len(times) - 1)
    out = np.zeros((len(epochs), 2, 2))
    for k in range(1, len(epochs)):
        a, b = idx[k - 1], idx[k]
        out[k] = [p[b, :2] - p[a, :2], v[b, :2] - v[a, :2]]
    return out


def static_drift(times, truth_xy, estimate_xy, speed_limit=.02, window=1.):
    """Error change across windows where the true base was static (evaluation only)."""
    out = []
    t = times[0]
    while t + window < times[-1]:
        sel = (times >= t) & (times <= t + window)
        if sel.sum() > 5:
            moved = np.linalg.norm(truth_xy[sel][-1] - truth_xy[sel][0])
            if moved < speed_limit * window:
                delta = np.linalg.norm(estimate_xy[sel][-1] - estimate_xy[sel][0])
                out.append(float(delta))
        t += window
    return {'windows': len(out), 'drift_p50_m': float(np.percentile(out, 50)) if out else None,
            'drift_max_m': float(max(out)) if out else None}


def run_variant(times, acc_world, rel, foot_v, truth_t, truth_xy, bias_scale,
                r_scale=1., increments=None):
    est, imu, counters = fuse(times, acc_world, rel, foot_v, bias_scale,
                              r_scale=r_scale, increments=increments)
    truth_at = np.column_stack([np.interp(times, truth_t, truth_xy[:, 0]),
                                np.interp(times, truth_t, truth_xy[:, 1])])
    base = truth_at[0]
    xy = truth_at - base
    error = np.linalg.norm(est[:, :2] - xy, axis=1)
    imu_error = np.linalg.norm(imu[:, :2] - xy, axis=1)
    return {'fusion': metrics(error), 'strapdown_only': metrics(imu_error),
            'counters': counters, 'static': static_drift(times, xy, est[:, :2]),
            'estimate': est, 'error': error, 'xy': xy}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('record', 'profile', 'scene', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--foot-dwell', type=float, default=.2)
    p.add_argument('--warmup', type=float, default=1.)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('Expected a new output file')

    imu_t, imu = load_stream(a.record / 'imu.jsonl',
                             ('q_xyzw', 'w', 'a', 'recv'))
    odom_t, odom = load_stream(a.record / 'odom.jsonl', ('position',))
    imu_t, q_all, gyro_all, acc_all, recv = dedupe(
        imu_t, *(imu[f] for f in ('q_xyzw', 'w', 'a', 'recv')))
    odom_t, odom = dedupe(odom_t, odom['position'])
    name_list = json.loads(a.profile.read_text())['joint_names']
    joint_t, joint_rows = load_joint_timeline(
        a.record / 'joints.jsonl', name_list,
        lambda r: float(np.interp(r, recv, imu_t)))
    nj = len(name_list)

    keep = imu_t >= imu_t[0] + a.warmup
    imu_t = imu_t[keep]
    q = q_all[keep]
    gyro = gyro_all[keep]
    acc = acc_all[keep]
    joints = support_latest(joint_t, joint_rows, imu_t)
    joint_pos, joint_vel = joints[:, :nj], joints[:, nj:]

    rotation = Rotation.from_quat(q)
    acc_world = rotation.apply(acc) + np.array([0., 0., -G])

    model = mujoco.MjModel.from_xml_path(str(a.scene))
    data = mujoco.MjData(model)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in name_list]
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, s + '_ankle_roll_link')
            for s in ('left', 'right')]
    if min(*ids, *feet) < 0:
        raise ValueError('Missing model joint or foot')
    qpos_addr = [model.jnt_qposadr[i] for i in ids]
    dof_addr = [model.jnt_dofadr[i] for i in ids]
    rel, foot_v_raw = foot_geometry(model, data, qpos_addr, dof_addr, feet,
                                    q[:, [3, 0, 1, 2]], joint_pos, joint_vel, gyro)
    # Rate-invariant low-pass on the contact-Jacobian foot velocity.
    foot_v = np.zeros_like(foot_v_raw)
    for i in range(1, len(imu_t)):
        alpha = np.exp(-(imu_t[i] - imu_t[i - 1]) / TAU_FOOT_VEL)
        foot_v[i] = (1. - alpha) * foot_v_raw[i] + alpha * foot_v[i - 1]
    truth_xy = np.column_stack([odom[:, 0], odom[:, 1]]) - np.column_stack([odom[0, 0], odom[0, 1]])

    gate = {'dwell': a.foot_dwell, 'slip_guard': True}
    native = run_variant(imu_t, acc_world, rel, foot_v, odom_t, truth_xy, gate)

    # Same recording, collapsed to the deployed 20 ms tick before any processing.
    grid = np.arange(imu_t[0], imu_t[-1], DT_REF)
    idx = np.searchsorted(imu_t, grid, side='right') - 1
    ok = idx > 0
    snap = run_variant(grid[ok], acc_world[idx[ok]], rel[idx[ok]], foot_v[idx[ok]],
                       odom_t, truth_xy, gate)
    # The original pipeline: geometry, low-pass and propagation all evaluated on the
    # 20 ms grid. This is the baseline the historical 0.176-0.48 m numbers came from.
    epochs = grid[ok]
    idx_epoch = np.clip(np.searchsorted(imu_t, epochs, side='right') - 1, 0, len(imu_t) - 1)
    joints_epoch = support_latest(joint_t, joint_rows, epochs)
    rel_snap, raw_snap = foot_geometry(model, data, qpos_addr, dof_addr, feet,
                                       q[idx_epoch][:, [3, 0, 1, 2]],
                                       joints_epoch[:, :nj], joints_epoch[:, nj:],
                                       gyro[idx_epoch])
    fv_snap = np.zeros_like(raw_snap)
    for i in range(1, len(epochs)):
        fv_snap[i] = .5 * raw_snap[i] + .5 * fv_snap[i - 1]
    acc_snap = acc_world[idx_epoch]
    orig = run_variant(epochs, acc_snap, rel_snap, fv_snap, odom_t, truth_xy, gate)
    pre = preintegrate(imu_t, acc_world, epochs)
    fused = run_variant(epochs, acc_world[idx[ok]], rel[idx[ok]], foot_v[idx[ok]],
                        odom_t, truth_xy, gate, increments=pre)
    native_r = run_variant(imu_t, acc_world, rel, foot_v, odom_t, truth_xy, gate,
                           r_scale=np.median(np.diff(imu_t)) / DT_REF)
    # Same 20 ms measurement stream, native-rate propagation: isolates propagation quality.
    pre_snap = run_variant(epochs, acc_snap, rel_snap, fv_snap, odom_t, truth_xy, gate,
                           increments=pre)

    report = {
        'native_rate_hz': 1. / float(np.median(np.diff(imu_t))),
        'native_samples': int(len(imu_t)),
        'compared_seconds': float(imu_t[-1] - imu_t[0]),
        'foot_dwell_s': a.foot_dwell,
        'native': {'fusion': native['fusion'], 'strapdown_only': native['strapdown_only'],
                   'counters': native['counters'], 'static': native['static']},
        'snapshot_50hz': {'fusion': snap['fusion'], 'strapdown_only': snap['strapdown_only'],
                          'counters': snap['counters'], 'static': snap['static']},
        'native_preintegration_update_50hz': {'fusion': fused['fusion'],
                                              'strapdown_only': fused['strapdown_only'],
                                              'counters': fused['counters'],
                                              'static': fused['static']},
        'snapshot_geometry_50hz': {'fusion': orig['fusion'], 'strapdown_only': orig['strapdown_only'],
                                   'counters': orig['counters'], 'static': orig['static']},
        'preintegration_snapshot_geometry': {'fusion': pre_snap['fusion'],
                                             'counters': pre_snap['counters'],
                                             'static': pre_snap['static']},
        'native_with_rate_scaled_measurement_noise': {'fusion': native_r['fusion'],
                                                     'counters': native_r['counters'],
                                                     'static': native_r['static']},
        'rate_normalisation': ('foot-velocity low-pass re-expressed as a time constant, support dwell '
                               'in seconds, process noise scaled by dt; measurement noise unchanged'),
        'limitations': ('Offline replay only. Joint samples are aligned by local receive time because the '
                        'recording kept no joint header stamp. Support is inferred geometrically, not from '
                        'measured contact. No control loop, no hardware validation.'),
    }
    a.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
