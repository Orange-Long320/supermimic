"""Offline 200Hz collocation with original compliant inverse and servo dynamics.

Positions define discrete velocities; the initial pose and velocity are fixed.
Every interval uses the checked implicitfast servo acceleration map. Optional
phase-I root/motor violation variables are offline search errors, never applied
forces, successful physics, or training approval. Original geometry/joint/speed
bounds remain hard throughout line search. Full actual replay is still required.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['reference', 'model', 'controller', 'solver-python', 'solver-script', 'out']:
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--iterations', type=int, default=6)
    p.add_argument('--restore-feasibility', action='store_true')
    p.add_argument('--maximum-frames', type=int, default=241)
    a = p.parse_args()
    if a.out.exists() or a.iterations < 1 or a.maximum_frames < 5:
        p.error('New output and bounded positive budgets required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Dedicated server CPU-only run required')
    import mujoco
    import numpy as np
    import scipy.sparse as sp
    from scipy.spatial.transform import Rotation
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra', check_assets=True)
    from supermimic.robots.dimensions import PROFILE
    spec = importlib.util.spec_from_file_location('private_native_map', a.controller)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    m = mujoco.MjModel.from_binary_path(str(a.model))
    if (m.nq, m.nv, m.nu, m.neq) != (38, 37, 31, 0) or abs(m.opt.timestep - .005) > 1e-12:
        raise ValueError('Original actuated200Hz native model required')
    d = mujoco.MjData(m)
    dt = float(m.opt.timestep)
    with np.load(a.reference) as z:
        source = z['qpos'].copy()
        initial_velocity = z['qvel'][0].copy()
        fps = float(np.asarray(z['fps']).ravel()[0])
    if fps != 200. or not 5 <= len(source) <= a.maximum_frames or abs(initial_velocity).max() > 1e-12:
        raise ValueError('Explicit bounded200Hz rest-start reference required')
    qs = source.copy()
    n = len(qs)
    names = [m.joint(i).name.removeprefix('robot/') for i in range(1, 32)]
    if names != PROFILE['joint_names']:
        raise ValueError('Unexpected joint order')
    effort = np.array([PROFILE['effort_limit'][k] for k in names])
    speed = np.array([PROFILE['velocity_limit'][k] for k in names])
    np.testing.assert_allclose(m.actuator_forcerange, np.c_[-effort, effort], rtol=0, atol=1e-9)
    root = int(m.jnt_bodyid[0])
    weight = float(m.body_mass.sum() * 9.81)
    scale = np.r_[np.full(3, weight), np.full(3, weight * .7), effort]
    centre = m.jnt_range[1:].mean(1)
    radius = np.diff(m.jnt_range[1:], axis=1).ravel() * .45
    loq, hiq = centre - radius, centre + radius
    robot = [g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == root
             and (m.geom_contype[g] or m.geom_conaffinity[g])]
    terrain = [g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] != root
               and (m.geom_contype[g] or m.geom_conaffinity[g])]
    feet = [m.body('robot/' + s + '_ankle_roll_link').id for s in ['left', 'right']]
    d.qpos[:] = source[0]
    mujoco.mj_forward(m, d)
    anchors = [(d.xpos[b].copy(), Rotation.from_matrix(d.xmat[b].reshape(3, 3))) for b in feet]
    # Both feet are explicitly planted in this bounded input prefix. Normal
    # deflection is allowed under the original1mm collision/contact law;
    # tangential position and orientation retain tight source task bounds.
    witness = np.zeros(6)
    jp, jr = np.zeros((3, 37)), np.zeros((3, 37))
    pos_size = (n - 1) * 37
    count = n - 1
    root_offset = pos_size
    motor_offset = root_offset + (count * 6 if a.restore_feasibility else 0)
    total = motor_offset + (count * 31 if a.restore_feasibility else 0)
    a.out.mkdir(parents=True)
    (a.out / 'as-run.py').write_bytes(Path(__file__).read_bytes())
    (a.out / 'native-map-as-run.py').write_bytes(a.controller.read_bytes())
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    provenance = {'configuration': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        'sha256': {str(f): sha(f) for f in [a.reference, a.model, a.controller, a.solver_script]},
        'initial_state_fixed': True, 'initial_velocity_zero': True, 'fps': fps,
        'velocity_stencil': 'Backward manifold difference; q0 velocity explicitly zero',
        'root_force_variables_applied': False, 'training_approved': False, 'scope': __doc__}
    (a.out / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    history = []
    started = time.monotonic()

    def velocity(q, i):
        v = initial_velocity.copy()
        if i:
            mujoco.mj_differentiatePos(m, v, dt, q[i-1], q[i])
        return v

    def force(q, i):
        v = velocity(q, i)
        after = velocity(q, i+1)
        d.qpos[:], d.qvel[:] = q[i], v
        mujoco.mj_forward(m, d)
        mapping = native.implicitfast_acceleration_map(m, d)
        d.qacc[:] = mapping @ ((after - v) / dt)
        mujoco.mj_inverse(m, d)
        return d.qfrc_inverse.copy() / scale

    def measure(q):
        forces = np.array([force(q, i) for i in range(count)])
        vs = np.array([velocity(q, i) for i in range(n)])
        penetration, self_pen, xy, rotation, contact_gap = 0., 0., 0., 0., 0.
        for pose in q:
            d.qpos[:], d.qvel[:] = pose, 0.
            mujoco.mj_forward(m, d)
            for g in robot:
                for o in terrain:
                    penetration = max(penetration, -float(mujoco.mj_geomDistance(m, d, g, o, .02, witness)))
            for c in d.contact:
                if c.geom1 in robot and c.geom2 in robot:
                    self_pen = max(self_pen, -float(c.dist))
            for b, (position, rot) in zip(feet, anchors):
                xy = max(xy, float(abs(d.xpos[b, :2] - position[:2]).max()))
                rotation = max(rotation, float((rot.inv() * Rotation.from_matrix(d.xmat[b].reshape(3, 3))).magnitude()))
                gaps = [mujoco.mj_geomDistance(m, d, g, o, .02, witness)
                        for g in robot if m.geom_bodyid[g] == b for o in terrain]
                contact_gap = max(contact_gap, float(min(gaps)))
        soft = float(np.maximum(abs(q[:, 7:] - centre) - radius, 0).max())
        excess = np.maximum(abs(forces[:, 6:]) - 1., 0.)
        return dict(root_max=float(abs(forces[:, :6]).max()), motor_max=float(abs(forces[:, 6:]).max()),
            merit=float(np.square(forces[:, :6]).sum() + 10*np.square(excess).sum()),
            penetration_m=penetration, self_penetration_m=self_pen, sticking_xy_error_m=xy,
            foot_orientation_error_rad=rotation, planted_foot_gap_m=contact_gap, soft_violation_rad=soft,
            speed_ratio=float((abs(vs[:, 6:])/speed).max())), forces

    def geometry_ok(metrics):
        return (metrics['penetration_m'] <= .001 and metrics['self_penetration_m'] <= .001
                and metrics['sticking_xy_error_m'] <= .0001 and metrics['foot_orientation_error_rad'] <= .001
                and metrics['planted_foot_gap_m'] <= .0001 and metrics['soft_violation_rad'] <= 1e-8
                and metrics['speed_ratio'] <= 1.)

    def progress(status, **extra):
        metrics, _ = measure(qs)
        report = dict(status=status, history=history, metrics=metrics, elapsed_seconds=time.monotonic()-started,
            model_sha256=sha(a.model), source_sha256=sha(a.reference),
            necessary_native_node_equations_passed=bool(geometry_ok(metrics) and metrics['root_max'] <= 1e-7
                and metrics['motor_max'] <= 1.), actual_continuous_execution_verified=False,
            complete_climb_verified=False, training_approved=False, **extra)
        (a.out / 'progress.json').write_text(json.dumps(report, indent=2))
        return metrics

    current = progress('running')
    if not geometry_ok(current):
        raise ValueError('Input violates unchanged geometry/task/joint/speed bounds')
    trust = 1.
    termination = 'iteration_budget'
    for iteration in range(a.iterations):
        rows, cols, vals, low, high = [], [], [], [], []

        def add(indices, values, lower, upper):
            row = len(low)
            indices, values = np.asarray(indices, dtype=int).ravel(), np.asarray(values, dtype=float).ravel()
            use = abs(values) > 1e-14
            rows.extend([row]*int(use.sum()))
            cols.extend(indices[use].tolist())
            vals.extend(values[use].tolist())
            low.append(float(lower))
            high.append(float(upper))

        original = []
        epsilon = 1e-7
        for i in range(count):
            baseline = force(qs, i)
            original.append(baseline)
            ids, blocks = [], []
            nodes = list(range(max(1, i-1), min(n-1, i+1)+1))
            for node in nodes:
                block = np.zeros((37, 37))
                trial = qs.copy()
                for j in range(37):
                    delta = np.zeros(37)
                    delta[j] = epsilon
                    trial[node] = qs[node]
                    mujoco.mj_integratePos(m, trial[node], delta, 1.)
                    plus = force(trial, i)
                    trial[node] = qs[node]
                    mujoco.mj_integratePos(m, trial[node], delta, -1.)
                    block[:, j] = (plus-force(trial, i))/(2*epsilon)
                blocks.append(block)
                ids.extend(range((node-1)*37, node*37))
            derivative = np.concatenate(blocks, axis=1)
            if iteration == 0 and i == count//2:
                rng = np.random.default_rng(42)
                perturb = rng.normal(size=len(ids))*1e-8
                trial = qs.copy()
                for k, node in enumerate(nodes):
                    mujoco.mj_integratePos(m, trial[node], perturb[k*37:(k+1)*37], 1.)
                error = float(abs(force(trial, i)-baseline-derivative@perturb).max())
                (a.out/'jacobian-check.json').write_text(json.dumps({'maximum_normalized_error':error,'tolerance':1e-4},indent=2))
                if error > 1e-4:
                    raise ValueError('Native discrete dynamics Jacobian failed independent perturbation')
            for j in range(37):
                ii, dd = list(ids), derivative[j].copy()
                if j < 6:
                    if a.restore_feasibility:
                        ii.append(root_offset+i*6+j)
                        dd = np.r_[dd, -1.]
                    add(ii, dd, -baseline[j], -baseline[j])
                elif a.restore_feasibility:
                    violation = motor_offset+i*31+j-6
                    add(ii+[violation], np.r_[dd, -1.], -np.inf, 1.-baseline[j])
                    add(ii+[violation], np.r_[dd, 1.], -1.-baseline[j], np.inf)
                else:
                    add(ii, dd, -1.-baseline[j], 1.-baseline[j])
            if i % 40 == 0:
                print(json.dumps({'iteration':iteration,'linearized_interval':i,'intervals':count}),flush=True)
        bound_lo, bound_hi = np.full(total, -np.inf), np.full(total, np.inf)
        for node in range(1,n):
            indices = np.arange((node-1)*37, node*37)
            delta = np.zeros(37)
            mujoco.mj_differentiatePos(m, delta, 1., source[node], qs[node])
            step = np.r_[np.full(3,.0005), np.full(34,.005)]*trust
            domain = np.r_[np.full(3,.1), np.full(3,.15), np.full(31,np.deg2rad(15.))]
            lower, upper = np.maximum(-step,-domain-delta), np.minimum(step,domain-delta)
            lower[6:] = np.maximum(lower[6:],loq-qs[node,7:])
            upper[6:] = np.minimum(upper[6:],hiq-qs[node,7:])
            bound_lo[indices],bound_hi[indices] = lower,upper
            v = velocity(qs,node)
            for j in range(31):
                ii,dd = [(node-1)*37+6+j],[1./(dt*speed[j])]
                if node>1:
                    ii.append((node-2)*37+6+j)
                    dd.append(-1./(dt*speed[j]))
                add(ii,dd,-1.-v[6+j]/speed[j],1.-v[6+j]/speed[j])
            d.qpos[:],d.qvel[:] = qs[node],0.
            mujoco.mj_forward(m,d)
            for b,(position,rotation) in zip(feet,anchors):
                mujoco.mj_jacBody(m,d,jp,jr,b)
                error = d.xpos[b,:2]-position[:2]
                for j in range(2):add(indices,jp[j],-.0001-error[j],.0001-error[j])
                error = (rotation.inv()*Rotation.from_matrix(d.xmat[b].reshape(3,3))).as_rotvec()
                angular = rotation.as_matrix().T@jr
                for j in range(3):add(indices,angular[j],-.0005-error[j],.0005-error[j])
            for g in robot:
                for o in terrain:
                    distance=float(mujoco.mj_geomDistance(m,d,g,o,.02,witness))
                    if distance>=.005:continue
                    vector=witness[:3]-witness[3:]
                    length=float(np.linalg.norm(vector))
                    if length<1e-12:raise ValueError('Undefined contact witness normal')
                    normal=np.sign(distance)*vector/length
                    mujoco.mj_jac(m,d,jp,jr,witness[:3],int(m.geom_bodyid[g]))
                    add(indices,normal@jp,-.001-distance,np.inf)
        if a.restore_feasibility:bound_lo[motor_offset:]=0.
        if np.any(bound_lo>bound_hi):raise ValueError('Empty source/trust/original joint domain')
        A=sp.coo_matrix((vals,(rows,cols)),shape=(len(low),total)).tocsc()
        A=sp.vstack([A,sp.eye(total,format='csc')],format='csc')
        lower,upper=np.r_[low,bound_lo],np.r_[high,bound_hi]
        if iteration==0 and a.restore_feasibility:
            witness_x=np.zeros(total)
            original=np.asarray(original)
            witness_x[root_offset:motor_offset]=original[:,:6].ravel()
            witness_x[motor_offset:]=np.maximum(abs(original[:,6:])-1.,0).ravel()+1e-10
            ax=A@witness_x
            error=float(max(np.maximum(lower-ax,0).max(),np.maximum(ax-upper,0).max()))
            (a.out/'feasibility-witness.json').write_text(json.dumps({'linear_constraint_error':error,'scope':'Offline nonzero violations only; no physical approval'},indent=2))
            if error>1e-7:raise ValueError('Phase-I matrix failed its explicit feasible witness')
        diagonal=np.r_[np.tile(np.r_[np.full(3,1e4),np.ones(34)],n-1),
            np.full(count*6,1e5) if a.restore_feasibility else np.empty(0),
            np.full(count*31,1e6) if a.restore_feasibility else np.empty(0)]
        P=sp.diags(diagonal,format='csc')
        # Penalize curvature of increments without attracting the solution
        # back to the dynamically invalid source after feasibility improves.
        D=sp.diags([np.ones(n-2),-2*np.ones(n-1),np.ones(n-2)],[-1,0,1],shape=(n-1,n-1),format='csc')
        D=sp.kron(D,sp.diags(np.r_[np.full(3,1e3),np.full(34,10.)]),format='csc')
        if total>pos_size:D=sp.hstack([D,sp.csc_matrix((pos_size,total-pos_size))],format='csc')
        P=P+D.T@D
        directory=a.out/f'qp-{iteration:02d}'
        directory.mkdir()
        sp.save_npz(directory/'P.npz',sp.triu(P,format='csc'))
        sp.save_npz(directory/'A.npz',A)
        np.savez_compressed(directory/'vectors.npz',q=np.zeros(total),lower=lower,upper=upper)
        env=os.environ.copy()
        env.pop('PYTHONPATH',None)
        with (directory/'solver.log').open('w') as f:
            code=subprocess.run([str(a.solver_python),str(a.solver_script),str(directory),'--backend','clarabel'],
                stdout=f,stderr=subprocess.STDOUT,env=env,timeout=100).returncode
        if code:raise RuntimeError('Isolated QP process failed')
        solver=json.loads((directory/'solver.json').read_text())
        if solver['status_value'] not in [1,2] or solver['maximum_linear_constraint_violation'] is None or solver['maximum_linear_constraint_violation']>1e-5:
            termination='QP not numerically solved'
            history.append({'iteration':iteration,'solver':solver,'accepted':False})
            break
        with np.load(directory/'solution.npz') as z:solution=z['x'].copy()
        accepted=False
        trials=[]
        for alpha in [1.,.5,.25,.125,.0625]:
            trial=qs.copy()
            for node in range(1,n):mujoco.mj_integratePos(m,trial[node],solution[(node-1)*37:node*37],alpha)
            metrics,_=measure(trial)
            passed=geometry_ok(metrics)
            trials.append(dict(alpha=alpha,hard_geometry_tasks_passed=passed,**metrics))
            if passed and metrics['merit']<current['merit']*(1.-1e-6):
                qs,current,accepted=trial,metrics,True
                break
        history.append({'iteration':iteration,'accepted':accepted,'solver':solver,'trials':trials})
        if not accepted:trust*=.5
        np.testing.assert_array_equal(qs[0],source[0])
        np.savez_compressed(a.out/'candidate-latest.npz',qpos=qs,qvel=np.array([velocity(qs,i) for i in range(n)]),fps=np.array([200.]))
        progress('running')
        print(json.dumps({'iteration':iteration,'accepted':accepted,'trust':trust,**current}),flush=True)
        if current['root_max']<=1e-7 and current['motor_max']<=1.:
            termination='necessary_native_node_equations_converged'
            break
        if trust<.125:
            termination='Hard geometry line search stalled'
            break
    progress('completed',termination_reason=termination)


if __name__=='__main__':
    main()
