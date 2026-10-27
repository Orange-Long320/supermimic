"""CPU inverse-dynamics diagnostic with unilateral friction-pyramid foot forces.

The residual and joint torques are approximate (finite differences, estimated
contacts). A low residual is necessary, not a proof of dynamic executability.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import nnls
from scipy.spatial.transform import Rotation
from supermimic.data.contact_wrench import solve_contact_effort


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--friction", type=float, default=.8)
    p.add_argument("--exact-contact-wrench", action="store_true",
                   help="Also minimize peak motor effort while requiring exact floating-base force/torque balance")
    args = p.parse_args()
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.robots.dimensions import PROFILE
    model = load_g1_model()
    for name in PROFILE["joint_names"]:
        model.dof_armature[model.jnt_dofadr[model.joint(name).id]] = PROFILE["armature"][name]
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT) | int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
    data = mujoco.MjData(model)
    z = np.load(args.npz)
    fps = float(z["fps"].reshape(-1)[0])
    root = int(model.jnt_bodyid[0])
    root_index = z["body_names"].tolist().index(model.body(root).name)
    q = np.c_[z["body_pos_w"][:, root_index], z["body_quat_w"][:, root_index], z["joint_pos"]].astype(float)
    v = np.zeros((len(q), model.nv))
    acceleration = np.zeros_like(v)
    forward, backward = np.zeros(model.nv), np.zeros(model.nv)
    for i in range(1, len(q) - 1):
        mujoco.mj_differentiatePos(model, forward, 1 / fps, q[i], q[i + 1])
        mujoco.mj_differentiatePos(model, backward, -1 / fps, q[i], q[i - 1])
        v[i] = (forward + backward) / 2
        # Direct second differences avoid the even/odd aliasing of applying
        # a centered first derivative twice. Both tangents use body frame i.
        acceleration[i] = (forward - backward) * fps
    v[0], v[-1] = v[1], v[-2]
    feet = []
    for g in range(model.ngeom):
        if not model.geom(g).name.startswith("foot_"):
            continue
        mesh = model.geom_dataid[g]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        feet.append((g, model.mesh_vert[start:start + count]))
    # Standalone XML has no mjlab-generated position actuators. Read the exact
    # effort limits and armatures used by get_robot_cfg(), rather than treating
    # absent XML limits as unlimited motors.
    limits = np.array([PROFILE["effort_limit"][model.joint(j).name] for j in range(1, model.njnt)])
    rows = []
    if args.friction <= 0:
        p.error("friction must be positive")
    friction = args.friction / np.sqrt(2)
    rays = np.array([[x * friction, y * friction, 1] for x, y in ((1, 1), (1, -1), (-1, 1), (-1, -1))]).T
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    momentum_matrix = np.zeros((3, model.nv))
    for i in range(2, len(q) - 2):
        data.qpos[:], data.qvel[:], data.qacc[:] = q[i], v[i], acceleration[i]
        mujoco.mj_inverse(model, data)
        required = data.qfrc_inverse.copy()
        mujoco.mj_angmomMat(model, data, momentum_matrix, root)
        momentum = momentum_matrix @ v[i]
        world_moment = data.xmat[root].reshape(3, 3) @ required[3:6] + np.cross(q[i, :3], required[:3])
        required_cop = (np.array([-world_moment[1], world_moment[0]]) / required[2]).tolist() if required[2] > 1e-6 else None
        columns = []
        foot_heights = {}
        contact_points = []
        exact_wrench = None
        for g, local in feet:
            world = local @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            foot_heights[model.body(int(model.geom_bodyid[g])).name] = float(world[:, 2].min())
            near = world[world[:, 2] < .012]
            if len(near) > 12:
                near = near[np.linspace(0, len(near) - 1, 12).astype(int)]
            for point in near:
                contact_points.append(point[:2].tolist())
                mujoco.mj_jac(model, data, jp, jr, point, int(model.geom_bodyid[g]))
                columns.append(jp.T @ rays)
        if columns:
            basis = np.concatenate(columns, axis=1)
            # Ground-force distribution is underdetermined. Favor low motor
            # effort as well as the floating-base balance, rather than choosing
            # a least-force solution that can exaggerate joint requirements.
            weight = np.r_[np.full(3, 100 / (model.body_subtreemass[root] * 9.81)),
                           np.full(3, 200 / (model.body_subtreemass[root] * 9.81))]
            matrix = np.r_[basis[:6] * weight[:, None], basis[6:] / limits[:, None],
                           np.eye(basis.shape[1]) * 1e-5]
            target = np.r_[required[:6] * weight, required[6:] / limits, np.zeros(basis.shape[1])]
            forces, _ = nnls(matrix, target, maxiter=10000)
            residual = required - basis @ forces
            normal = float(forces.sum())
            if args.exact_contact_wrench:
                # Separate impossible ground wrenches from motor-effort demand.
                # NNLS above trades base imbalance against motor cost, so its
                # peak ratio alone cannot answer either question.
                fit = solve_contact_effort(basis, required, limits)
                exact_wrench = {"feasible_contact_wrench": bool(fit.success),
                                "solver_status": int(fit.status),
                                "minimum_peak_motor_ratio": float(fit.x[-1]) if fit.success else None,
                                "note": "Exact pose-derived base wrench in estimated friction cone; finite-difference/contact uncertainty still applies."}
        else:
            residual = required
            normal = 0.
        ratios = np.abs(residual[6:]) / np.where(limits > 0, limits, np.inf)
        rows.append({"frame": i, "contact_ray_count": 4 * sum(1 for _ in columns) // 4,
                     "exact_contact_wrench": exact_wrench,
                     "required_ground_force_world_N": required[:3].tolist(),
                     "required_center_of_pressure_xy_m": required_cop,
                     "estimated_contact_points_xy_m": contact_points,
                     "com_position_m": data.subtree_com[root].tolist(),
                     "angular_momentum_world_kg_m2_per_s": momentum.tolist(),
                     "root_angular_velocity_body_rad_per_s": v[i, 3:6].tolist(),
                     "foot_minimum_height_m": foot_heights,
                     "joint_velocity_rad_per_s": v[i, 6:].tolist(),
                     "joint_acceleration_rad_per_s2": acceleration[i, 6:].tolist(),
                     "required_normal_force_N": normal,
                     "floating_force_residual_N": float(np.linalg.norm(residual[:3])),
                     "floating_torque_residual_Nm": float(np.linalg.norm(residual[3:6])),
                     "max_joint_torque_to_limit": float(ratios.max()),
                     "worst_joint": model.joint(int(ratios.argmax()) + 1).name,
                     "required_joint_torque_Nm": residual[6:].tolist()})
    report = {"npz": str(args.npz.resolve()), "mass_kg": float(model.body_subtreemass[root]),
              "friction": args.friction,
              "actuator_limits_available": bool(np.all(limits > 0)),
              "rows": rows,
              "limitations": "Finite-difference inverse dynamics; contact patch estimated within12mm with the stated friction. Residual is a diagnostic, not a controller or a feasibility certificate; impacts need separate treatment."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        json.dump(report, output, indent=2)
    print(json.dumps({k:v for k,v in report.items() if k != "rows"}))
    print("peak_torque_ratio", max(row["max_joint_torque_to_limit"] for row in rows))


if __name__ == "__main__":
    main()
