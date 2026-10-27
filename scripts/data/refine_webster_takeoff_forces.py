"""Server CPU takeoff collocation with contact forces and motor effort bounds.

Uses the actual X2 mass/armature and effort limits. Contact friction is a
conservative pyramid. The aerial reference is fixed, so before/after QA must
check launch seams independently. A local solve is not a feasibility guarantee.
"""
import argparse
import hashlib
import importlib
import time
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-evaluations", type=int, default=80)
    p.add_argument("--takeoff-frame", type=int, default=15)
    p.add_argument("--switch-frame", type=int, default=5)
    p.add_argument("--landing-frame", type=int, help="Also jointly optimize the aerial phase up to this fixed landing pose")
    p.add_argument("--warmstart-state", type=Path)
    p.add_argument("--contact-patch", action="store_true")
    p.add_argument("--contact-anchor", choices=("toe", "lowest"), default="toe",
                   help="Fixed support pivot; lowest uses the actual initially lowest collision vertex")
    p.add_argument("--initialization-evaluations", type=int, default=0)
    p.add_argument("--free-com", action="store_true", help="Let dynamics determine COM flight instead of locking the camera-derived path")
    p.add_argument("--free-contact-path", action="store_true", help="Use contact velocity/complementarity instead of fixed camera-derived foot anchors")
    p.add_argument("--linear-iterations", type=int, default=300)
    p.add_argument("--linear-tolerance", type=float, default=1e-5)
    p.add_argument("--force-regularization", type=float, default=.25)
    p.add_argument("--pose-regularization", type=float, default=1.)
    p.add_argument("--rotation-regularization", type=float, default=2.)
    p.add_argument("--smooth-regularization", type=float, default=5.)
    args = p.parse_args()
    from supermimic.data.motion_convert import load_g1_model, convert_csv_to_npz
    from supermimic.robots.dimensions import PROFILE
    m = load_g1_model()
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT) | int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
    for name in PROFILE["joint_names"]:
        m.dof_armature[m.jnt_dofadr[m.joint(name).id]] = PROFILE["armature"][name]
    d = mujoco.MjData(m)
    z = np.load(args.input)
    fps = float(z["fps"].reshape(-1)[0])
    root = int(m.jnt_bodyid[0])
    r = z["body_names"].tolist().index(m.body(root).name)
    base = np.c_[z["body_pos_w"][:, r], z["body_quat_w"][:, r], z["joint_pos"]].astype(float)
    full_flight = args.landing_frame is not None
    support_count = args.takeoff_frame + 1
    count, nj = (args.landing_frame if full_flight else support_count), m.nv - 6
    constraint_count = count if full_flight else count + 1
    npoints = 4 if args.contact_patch else 1
    nlambda = 4 * npoints
    width = m.nv + nlambda
    limits = np.array([PROFILE["effort_limit"][m.joint(j).name] for j in range(1, m.njnt)])
    rotations = Rotation.from_quat(base[:count, [4, 5, 6, 3]])
    meshes, anchors = [], {}
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0 or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        mesh = m.geom_dataid[g]
        a, n = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
        vertices = m.mesh_vert[a:a + n].astype(float)
        # The minimum of any linear height function is attained on the hull.
        vertices = vertices[ConvexHull(vertices).vertices]
        meshes.append((g, vertices))
    print("Exact convex hull vertices:", sum(len(v) for _, v in meshes), flush=True)
    radii = {g: float(np.linalg.norm(v, axis=1).max()) for g, v in meshes}
    from supermimic.data.centroidal_projection import sole_patches
    patches = sole_patches(m) if args.contact_patch else {}
    for frame, side in ((0, "right"), (args.switch_frame, "left")):
        d.qpos[:] = base[frame]
        mujoco.mj_forward(m, d)
        g, vertices = next(item for item in meshes if m.geom(item[0]).name.startswith("foot_") and m.body(m.geom_bodyid[item[0]]).name.startswith(side))
        world = vertices @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        k = world[:, 2].argmin()
        local = patches[g][0] if args.contact_patch and args.contact_anchor == "toe" else vertices[k]
        anchor = d.geom_xmat[g].reshape(3, 3) @ local + d.geom_xpos[g]
        anchor[2] = 0.
        anchors[side] = (g, local, anchor)
    com_target = []
    for frame in range(constraint_count):
        d.qpos[:] = base[frame]
        mujoco.mj_forward(m, d)
        com_target.append(d.subtree_com[root].copy())
    com_target = np.array(com_target)
    friction = 1. / np.sqrt(2)
    rays = np.array([[a * friction, b * friction, 1.] for a, b in ((1, 1), (1, -1), (-1, 1), (-1, -1))]).T
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    initial = np.c_[base[:count, :3], np.zeros((count, 3)), base[:count, 7:], np.full((count, nlambda), m.body_subtreemass[root] * 9.81 / nlambda)]
    centre = m.jnt_range[1:].mean(axis=1)
    half = .45 * np.diff(m.jnt_range[1:], axis=1).ravel()
    lower = np.tile(np.r_[[-5, -5, .1], np.full(3, -1.3), centre - half + 1e-5, np.zeros(nlambda)], count)
    upper = np.tile(np.r_[[5, 5, 2], np.full(3, 1.3), centre + half - 1e-5, np.full(nlambda, 2000)], count)
    if full_flight:
        initial[support_count:, -nlambda:] = 0.
        for frame in range(support_count, count):
            lower[frame * width + 3:frame * width + 6] = -.7
            upper[frame * width + 3:frame * width + 6] = .7
            upper[frame * width + width - nlambda:(frame + 1) * width] = 1e-9
        if not args.free_contact_path:
            lower[:m.nv], upper[:m.nv] = initial[0, :m.nv] - 1e-8, initial[0, :m.nv] + 1e-8
    locked = ("head_yaw_joint", "head_pitch_joint") if full_flight else ("waist_pitch_joint", "waist_roll_joint", "head_yaw_joint", "head_pitch_joint")
    for name in locked:
        column = m.jnt_dofadr[m.joint(name).id]
        lower[column::width], upper[column::width] = -1e-7, 1e-7

    blocksize = 3 + 3 + len(meshes) + nj + 3 + 6 + nj + nj + nlambda + 5 * npoints + 1
    evaluations = 0
    latest = {}
    initializing = False

    def decode(flat):
        x = flat.reshape(count, width)
        q = base.copy()
        q[:count, :3] = x[:, :3]
        q[:count, 3:7] = (rotations * Rotation.from_rotvec(x[:, 3:6])).as_quat()[:, [3, 0, 1, 2]]
        q[:count, 7:] = x[:, 6:6 + nj]
        v = np.zeros((count + 3, m.nv))
        acceleration = np.zeros_like(v)
        forward, backward = np.zeros(m.nv), np.zeros(m.nv)
        for i in range(1, count + 2):
            mujoco.mj_differentiatePos(m, forward, 1 / fps, q[i], q[i + 1])
            mujoco.mj_differentiatePos(m, backward, -1 / fps, q[i], q[i - 1])
            v[i], acceleration[i] = (forward + backward) / 2, (forward - backward) * fps
        mujoco.mj_differentiatePos(m, v[0], 1 / fps, q[0], q[1])
        acceleration[0] = acceleration[1]
        return x, q, v, acceleration

    # Aerial one-sided velocity at takeoff is independent of all optimized states.
    target_velocity = np.zeros(m.nv)
    v1, v2 = np.zeros(m.nv), np.zeros(m.nv)
    boundary_index = support_count - 1
    mujoco.mj_differentiatePos(m, v1, 1 / fps, base[boundary_index], base[boundary_index + 1])
    mujoco.mj_differentiatePos(m, v2, 2 / fps, base[boundary_index], base[boundary_index + 2])
    target_velocity[:] = 2 * v1 - v2

    def residual(flat):
        nonlocal evaluations, latest
        x, q, v, acceleration = decode(flat)
        blocks, diagnostics = [], []
        mg = m.body_subtreemass[root] * 9.81
        for i in range(constraint_count):
            d.qpos[:], d.qvel[:], d.qacc[:] = q[i], v[i], acceleration[i]
            if initializing:
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
            else:
                mujoco.mj_inverse(m, d)
            if i < support_count:
                g, local, anchor = anchors["right" if i < args.switch_frame else "left"]
                point = d.geom_xmat[g].reshape(3, 3) @ local + d.geom_xpos[g]
                required = np.zeros(m.nv) if initializing else d.qfrc_inverse.copy()
                gaps, powers, contact_velocities = [], [], []
                normal_force = 0.
                locations = patches[g][1] if args.contact_patch else [local]
                for k, contact_local in enumerate(locations):
                    if initializing and not args.free_contact_path:
                        gaps.append(0.)
                        powers.append(0.)
                        contact_velocities.extend([0., 0., 0.])
                        continue
                    contact = d.geom_xmat[g].reshape(3, 3) @ contact_local + d.geom_xpos[g]
                    mujoco.mj_jac(m, d, jp, jr, contact, int(m.geom_bodyid[g]))
                    force = rays @ x[i, -nlambda:].reshape(npoints, 4)[k]
                    if not initializing:
                        required -= jp.T @ force
                    velocity = jp @ v[i]
                    normal_force += force[2]
                    gaps.append(0. if initializing else max(contact[2] - .001, 0) * force[2] * 50)
                    powers.append(0. if initializing else max(float(force[:2] @ velocity[:2]), 0) * .5)
                    contact_velocities.extend(velocity * force[2] / mg * 1000 if args.free_contact_path else np.zeros(3))
                contact_error = np.zeros(3) if args.free_contact_path else point - anchor
                orientation_error = x[i, 3:6]
                # Scale by a nominal share of body weight. An almost-zero
                # penalty leaves large force-distribution nullspaces and makes
                # the linear systems needlessly ill-conditioned.
                force_penalty = x[i, -nlambda:] / (mg / nlambda) * args.force_regularization
            else:
                # Also constrain the first fixed aerial pose. It depends on the
                # last adjustable supported pose and must have no contact force.
                required = np.zeros(m.nv) if initializing else d.qfrc_inverse.copy()
                contact_error, force_penalty = np.zeros(3), np.zeros(nlambda)
                gaps, powers = np.zeros(npoints), np.zeros(npoints)
                contact_velocities = np.zeros(3 * npoints)
                normal_force = 0.
                orientation_error = x[i, 3:6] if i < count else np.zeros(3)
            ratio = required[6:] / limits
            thresholds = np.array([(-.0005 if i < support_count else .008) if m.geom(g).name.startswith("foot_") else .008 for g, _ in meshes])
            heights = []
            for k, (g, vertices) in enumerate(meshes):
                lower_bound = d.geom_xpos[g, 2] - radii[g]
                # Exact broadphase: a bounding sphere entirely above the margin
                # cannot contribute to the floor inequality.
                heights.append(lower_bound if lower_bound > thresholds[k] else
                               (vertices @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min())
            heights = np.array(heights)
            balance = required[:6] * np.r_[np.full(3, 400 / mg), np.full(3, 800 / mg)]
            violations = np.maximum(np.abs(ratio) - .99, 0) * 80
            if i == 0:
                balance[:] = 0
                violations[:] = 0
            blocks.append(np.r_[
                (d.subtree_com[root] - com_target[i]) * (20 if args.free_com else (100000 if i >= support_count - 1 else (100 if initializing else 20))),
                contact_error * (1000 if initializing else 5000),
                np.minimum(heights - thresholds, 0) * (1000 if initializing else 5000),
                (q[i, 7:] - base[i, 7:]) * (2 if initializing else args.pose_regularization),
                orientation_error * (1 if initializing else args.rotation_regularization),
                balance, violations, ratio * .05, force_penalty, gaps, powers, contact_velocities,
                max(.2 * mg - normal_force, 0) / mg * 100 if args.free_contact_path and i < support_count - 2 else 0.,
            ])
            diagnostics.append({"frame": i, "force_residual_N": float(np.linalg.norm(required[:3])),
                                "torque_residual_Nm": float(np.linalg.norm(required[3:6])),
                                "motor_ratio": float(np.abs(ratio).max()),
                                "normal_force_N": float(normal_force),
                                "support_error_m": float(np.linalg.norm(contact_error)),
                                "minimum_height_m": float(heights.min())})
        # Finite-difference velocity continuity at the fixed launch pose.
        if args.free_contact_path:
            initial_next = np.zeros(m.nv)
            mujoco.mj_differentiatePos(m, initial_next, 1 / fps, q[1], q[2])
            rest_scale = np.r_[np.full(6, 200), np.full(nj, 50)]
            boundary = np.r_[v[0] * rest_scale, initial_next * rest_scale]
        else:
            boundary = np.zeros(m.nv) if full_flight else (v[boundary_index] - target_velocity) * np.r_[np.full(3, 5), np.full(3, 2), np.full(nj, 1)]
        # Include the first fixed pose in the stencil. Otherwise the final
        # adjustable state can drift away and create an artificial landing jump.
        fixed_tail = np.r_[base[count, :3], np.zeros(3), base[count, 7:]][None]
        smooth = np.diff(np.r_[x[:, :6 + nj], fixed_tail], n=2, axis=0) * (
            np.r_[np.full(3, 2000), np.full(3, 200), np.full(nj, 100)] if initializing else args.smooth_regularization)
        evaluations += 1
        latest = {"frames": diagnostics, "launch_velocity_error": (v[boundary_index] - target_velocity).tolist()}
        return np.r_[np.concatenate(blocks), boundary, smooth.ravel()]

    boundary_size = 2 * m.nv if args.free_contact_path else m.nv
    rows = constraint_count * blocksize + boundary_size + (count - 1) * (6 + nj)
    sparsity = lil_matrix((rows, count * width), dtype=int)
    for i in range(constraint_count):
        for j in range(max(0, i - 1), min(count, i + 2)):
            sparsity[i * blocksize:(i + 1) * blocksize, j * width:(j + 1) * width] = 1
    if args.free_contact_path:
        r = constraint_count * blocksize
        sparsity[r:r + m.nv, :2 * width] = 1
        sparsity[r + m.nv:r + 2 * m.nv, width:3 * width] = 1
    elif not full_flight:
        sparsity[constraint_count * blocksize:constraint_count * blocksize + m.nv, (count - 2) * width:count * width] = 1
    for i in range(count - 1):
        a = constraint_count * blocksize + boundary_size + i * (6 + nj)
        sparsity[a:a + 6 + nj, i * width:min(i + 3, count) * width] = 1
    start = initial.ravel()
    source_sha = hashlib.sha256(args.input.read_bytes()).hexdigest()
    if args.warmstart_state:
        state = np.load(args.warmstart_state)
        assert str(state["source_sha256"]) == source_sha
        start = state["parameters"]
        assert start.shape == initial.ravel().shape
    if args.initialization_evaluations:
        initializing = True
        print("KINEMATIC INITIALIZATION: motor diagnostics disabled in this phase", flush=True)
        init_fit = least_squares(residual, np.clip(start, lower, upper), jac_sparsity=sparsity.tocsr(),
                                bounds=(lower, upper), max_nfev=args.initialization_evaluations,
                                diff_step=1e-4, x_scale="jac", ftol=1e-4, xtol=1e-6,
                                tr_options={"atol": 1e-5, "btol": 1e-5, "maxiter": 300}, verbose=2)
        start = init_fit.x
        initializing = False
        print("FULL DYNAMICS: contact forces and motor limits enabled", flush=True)
    progress = args.out.with_name(args.out.name + "_progress")
    progress.mkdir(parents=True, exist_ok=False)
    accepted = 0

    def save_progress(intermediate_result):
        nonlocal accepted
        accepted += 1
        if accepted != 1 and accepted % 5:
            return
        values = residual(intermediate_result.x)
        checkpoint = progress / "latest.tmp.npz"
        with checkpoint.open("wb") as output:
            np.savez(output, parameters=intermediate_result.x, source_sha256=np.array(source_sha))
        checkpoint.replace(progress / "latest.npz")
        snapshot = dict(latest)
        snapshot.update({"accepted_iterations": accepted, "cost": float(values @ values / 2),
                         "source": str(args.input.resolve()), "source_sha256": source_sha,
                         "takeoff_frame": args.takeoff_frame, "landing_frame": args.landing_frame})
        temporary = progress / "latest.tmp.json"
        temporary.write_text(json.dumps(snapshot, indent=2))
        temporary.replace(progress / "latest.json")

    trf_module = importlib.import_module("scipy.optimize._lsq.trf")
    original_lsmr = trf_module.lsmr

    def traced_lsmr(*solver_args, **solver_kwargs):
        started = time.monotonic()
        result = original_lsmr(*solver_args, **solver_kwargs)
        record = {"stop_code": int(result[1]), "iterations": int(result[2]),
                  "residual_norm": float(result[3]), "condition_estimate": float(result[6]),
                  "seconds": time.monotonic() - started}
        with (progress / "linear-solves.jsonl").open("a") as output:
            output.write(json.dumps(record) + "\n")
        return result

    trf_module.lsmr = traced_lsmr
    fit = least_squares(residual, np.clip(start, lower, upper), jac_sparsity=sparsity.tocsr(),
                        bounds=(lower, upper), max_nfev=args.max_evaluations, diff_step=1e-4,
                        ftol=2e-5, xtol=1e-6, gtol=1e-5, x_scale="jac", verbose=2,
                        callback=save_progress, tr_options={"atol": args.linear_tolerance, "btol": args.linear_tolerance, "maxiter": args.linear_iterations})
    _, q, _, _ = decode(fit.x)
    residual(fit.x)
    args.out.mkdir(parents=True, exist_ok=False)
    np.savetxt(args.out / "webster.csv", np.c_[q[:, :3], q[:, [4, 5, 6, 3]], q[:, 7:]], delimiter=",", fmt="%.10f")
    convert_csv_to_npz(args.out / "webster.csv", args.out / "webster.npz", fps, fps)
    np.savez(args.out / "optimization-state.npz", parameters=fit.x,
             source_sha256=np.array(source_sha),
             support_force_coefficients=fit.x.reshape(count, width)[:, -nlambda:])
    latest.update({"source": str(args.input.resolve()), "cost": float(fit.cost), "nfev": fit.nfev,
                   "contact_anchor": args.contact_anchor,
                   "joint_takeoff_and_flight_fit": full_flight, "takeoff_frame": args.takeoff_frame,
                   "landing_frame": args.landing_frame,
                   "actual_residual_evaluations": evaluations, "aerial_poses_preserved": bool(np.array_equal(q[count:], base[count:])),
                   "limitations": "Local constrained least squares with consistent second differences; initial boundary and landing impulses unoptimized. Independent checks and closed-loop validation required."})
    (args.out / "force-fit.json").write_text(json.dumps(latest, indent=2))
    print(json.dumps(latest), flush=True)


if __name__ == "__main__":
    main()
