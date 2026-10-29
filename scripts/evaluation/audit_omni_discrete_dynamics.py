"""Separate native integrator consistency from finite contact-solver residuals.

Replays an existing command trace with its original forward/step ordering.
No model setting, physical threshold or recorded artifact is changed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        p.error('New output and explicit CPU-only execution required')
    import mujoco
    import numpy as np
    meta = json.loads((a.trace / 'result.json').read_text())
    model = Path(meta['configuration']['model'])
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    if sha(model) != meta['model_sha256']:
        raise ValueError('Recorded model hash changed')
    m = mujoco.MjModel.from_binary_path(str(model))
    if (m.nq, m.nv, m.nu, m.na) != (38, 37, 31, 0) or m.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST:
        raise ValueError('Original X2 implicitfast servo model required')
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra', check_assets=True)
    from supermimic.policy.native_inverse import implicitfast_acceleration_map
    d, inv = mujoco.MjData(m), mujoco.MjData(m)
    with np.load(a.trace / 'trajectory.npz') as z:
        states, controls, dt = z['state'].copy(), z['controls'].copy(), float(z['dt'])
    if states.shape != (len(controls) + 1, m.nq + m.nv) or dt != m.opt.timestep:
        raise ValueError('Invalid original trace layout/clock')
    cols = m.jnt_qposadr[m.actuator_trnid[:, 0]]
    vcols = m.jnt_dofadr[m.actuator_trnid[:, 0]]
    d.qpos[:], d.qvel[:] = states[0, :m.nq], states[0, m.nq:]
    d.ctrl[:] = d.qpos[cols] - m.actuator_biasprm[:, 2] * d.qvel[vcols] / m.actuator_gainprm[:, 0]
    mujoco.mj_forward(m, d)
    mass = np.zeros((m.nv, m.nv))
    rows = []
    for i, control in enumerate(controls):
        mujoco.mj_forward(m, d)
        d.ctrl[:] = control
        mujoco.mj_forward(m, d)
        mapping = implicitfast_acceleration_map(m, d)
        mujoco.mj_fullM(m, d, mass)
        q, v = d.qpos.copy(), d.qvel.copy()
        mujoco.mj_step(m, d)
        # implicitfast solves against smooth+constraint forces, not M*qacc.
        rhs = d.qfrc_smooth + d.qfrc_constraint
        derivative = np.zeros((m.nv, m.nv))
        for j in range(m.nv):
            adr, size = int(m.D_rowadr[j]), int(m.D_rownnz[j])
            derivative[j, m.D_colind[adr:adr + size]] = d.qDeriv[adr:adr + size]
        observed = (d.qvel - v) / dt
        predicted = np.linalg.solve(mass - dt * derivative, rhs)
        instantaneous = d.qacc.copy()
        inv.qpos[:], inv.qvel[:] = q, v
        mujoco.mj_forward(m, inv)
        inv.qacc[:] = mapping @ observed
        mujoco.mj_inverse(m, inv)
        inverse_error = inv.qfrc_inverse - d.qfrc_actuator
        solver_residual = mass @ instantaneous - rhs
        rows.append(dict(interval=i,
            exact_integrator_acceleration_error=float(abs(predicted - observed).max()),
            assumed_Mqacc_force_identity_error=float(abs(solver_residual).max()),
            assumed_acceleration_map_error=float(abs(mapping @ observed - instantaneous).max()),
            inverse_root_force_error=float(abs(inverse_error[:6]).max()),
            inverse_motor_force_error=float(abs(inverse_error[6:]).max()),
            solver_iterations=int(d.solver_niter.max()),
            replay_position_error=float(abs(d.qpos - states[i + 1, :m.nq]).max()),
            replay_velocity_error=float(abs(d.qvel - states[i + 1, m.nq:]).max()),
            external_force_max=float(max(abs(d.qfrc_applied).max(), abs(d.xfrc_applied).max()))))
        mujoco.mj_forward(m, d)
    maximum = {k: max(r[k] for r in rows) for k in rows[0] if k != 'interval'}
    report = dict(model_sha256=sha(model), trajectory_sha256=sha(a.trace / 'trajectory.npz'),
        mujoco_version=mujoco.__version__, model_options=dict(iterations=m.opt.iterations,
            tolerance=m.opt.tolerance, ls_iterations=m.opt.ls_iterations, integrator=int(m.opt.integrator)),
        intervals=len(rows), maximum=maximum,
        exact_integrator_and_replay_verified=bool(maximum['exact_integrator_acceleration_error'] <= 1e-10
            and maximum['replay_position_error'] <= 1e-6 and maximum['replay_velocity_error'] <= 1e-5),
        inverse_identity_passed_unchanged_1e_minus5=bool(maximum['inverse_root_force_error'] <= 1e-5
            and maximum['inverse_motor_force_error'] <= 1e-5),
        interpretation='Finite contact-solver residual prevents treating M*qacc as the exact force RHS. '
            'Use the unchanged simulator step for trajectory optimization and continuous replay.',
        complete_climb_verified=False, training_approved=False, rows=rows)
    a.out.mkdir(parents=True)
    (a.out / 'as-run.py').write_bytes(Path(__file__).read_bytes())
    (a.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
