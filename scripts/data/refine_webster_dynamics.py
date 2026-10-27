"""Server CPU projection of a Webster candidate onto contact/centroidal constraints.

This constructs a new candidate, preserving the input. It does not certify motor
limits or train a controller. All approximation errors are saved for review.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicHermiteSpline, CubicSpline
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--contacts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--lock-low-torque-joints", action="store_true")
    args = p.parse_args()
    from supermimic.data.motion_convert import load_g1_model, convert_csv_to_npz
    model = load_g1_model()
    data = mujoco.MjData(model)
    z = np.load(args.input)
    fps = float(z["fps"].reshape(-1)[0])
    dt = 1 / fps
    n = len(z["joint_pos"])
    t = np.arange(n) * dt
    annotation = json.loads(args.contacts.read_text())
    takeoff = round(annotation["takeoff_reference_frame"] / annotation["reference_fps"] * fps)
    landing = round(annotation["landing_reference_frame"] / annotation["reference_fps"] * fps)
    root = int(model.jnt_bodyid[0])
    root_index = z["body_names"].tolist().index(model.body(root).name)
    original = np.concatenate([z["body_pos_w"][:, root_index], z["body_quat_w"][:, root_index], z["joint_pos"]], axis=1).astype(float)
    q = original.copy()
    q[:, 7:] = np.clip(savgol_filter(q[:, 7:], 7, 3, axis=0), model.jnt_range[1:, 0] + 1e-5, model.jnt_range[1:, 1] - 1e-5)
    locked = []
    if args.lock_low_torque_joints:
        for name in ("waist_pitch_joint", "waist_roll_joint", "head_yaw_joint", "head_pitch_joint"):
            j = model.joint(name).id
            q[:, model.jnt_qposadr[j]] = 0.
            locked.append(name)
    baseline = q.copy()
    meshes = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0 or not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        mesh = model.geom_dataid[g]
        if mesh < 0:
            raise ValueError("This candidate expects the existing X2 collision meshes")
        a, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        meshes.append((g, model.mesh_vert[a:a + count].copy()))
    feet = [item for item in meshes if model.geom(item[0]).name.startswith("foot_")]

    def fk(pose):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        return data.subtree_com[root].copy()

    def vertices(item):
        g, v = item
        return v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]

    original_com = np.array([fk(row) for row in original])
    # Fix the actual source-derived support points instead of independently
    # snapping only the lowest foot's height on every frame.
    anchors = {}
    for frame, side in ((0, "right"), (5, "left")):
        fk(q[frame])
        item = next(item for item in feet if model.body(model.geom_bodyid[item[0]]).name.startswith(side))
        world = vertices(item)
        k = int(world[:, 2].argmin())
        anchor = world[k].copy()
        anchor[2] = 0.005
        anchors[side] = (item[0], item[1][k].copy(), anchor)

    target_takeoff = original_com[takeoff].copy()
    target_takeoff[2] += 0.15
    horizontal_velocity = np.array([1.66, 0.22])
    target_takeoff[:2] = original_com[0, :2] + horizontal_velocity * t[takeoff]
    flight_duration = t[landing] - t[takeoff]
    velocity = np.r_[horizontal_velocity, (original_com[landing, 2] - target_takeoff[2]) / flight_duration + 0.5 * 9.81 * flight_duration]
    start_velocity = np.r_[horizontal_velocity, -0.35]
    com_spline = CubicHermiteSpline(
        [t[0], t[takeoff]], [original_com[0], target_takeoff], [start_velocity, velocity]
    )
    stance_target = com_spline(t[:takeoff + 1])
    stance_fit = []

    def solve_stance():
        previous = q[0, 7:19].copy()
        for frame in range(takeoff + 1):
            reference = original[frame]
            side = "right" if frame < 5 else "left"
            g, local, anchor = anchors[side]
            indices = np.r_[np.arange(3), np.arange(7, 19)]
            x0 = q[frame, indices].copy()
            lower = np.r_[[-5, -5, .1], model.jnt_range[1:13, 0] + 1e-5]
            upper = np.r_[[5, 5, 2], model.jnt_range[1:13, 1] - 1e-5]

            def residual(x):
                pose = q[frame].copy()
                pose[indices] = x
                com = fk(pose)
                point = data.geom_xmat[g].reshape(3, 3) @ local + data.geom_xpos[g]
                # Only legs are adjustable in this initializer. Arm floor
                # constraints belong to the following full-body fit.
                floor = np.array([vertices(item)[:, 2].min() for item in feet])
                return np.r_[
                    (com - stance_target[frame]) * 150,
                    (point - anchor) * 150,
                    np.minimum(floor - 0.003, 0) * 100,
                    (x[3:] - reference[7:19]) * .08,
                    (x[3:] - previous) * .1,
                ]

            fit = least_squares(residual, np.clip(x0, lower, upper), bounds=(lower, upper),
                                max_nfev=60, diff_step=1e-5, ftol=1e-8, xtol=1e-8, gtol=1e-8)
            q[frame, indices] = fit.x
            previous = fit.x[3:].copy()
            com = fk(q[frame])
            point = data.geom_xmat[g].reshape(3, 3) @ local + data.geom_xpos[g]
            stance_fit.append({"frame": frame, "com_error_m": float(np.linalg.norm(com - stance_target[frame])),
                               "support_error_m": float(np.linalg.norm(point - anchor))})

    print("Initializing support contacts", flush=True)
    solve_stance()
    print("Shooting constant aerial angular momentum", flush=True)
    # Blend the corrected push-off leg posture into the recovered aerial tuck.
    joint_delta = q[takeoff, 7:] - baseline[takeoff, 7:]
    for frame in range(takeoff + 1, landing):
        u = min((frame - takeoff) / 10, 1)
        q[frame, 7:] += joint_delta * (1 - 3 * u**2 + 2 * u**3)
    q[:, 7:] = np.clip(q[:, 7:], model.jnt_range[1:, 0] + 1e-5, model.jnt_range[1:, 1] - 1e-5)

    momentum_matrix = np.empty((3, model.nv))
    flight_t = t[takeoff:landing + 1]
    reference_rot = Rotation.from_quat(original[takeoff:landing + 1, [4, 5, 6, 3]])
    initial_quat = q[takeoff, 3:7].copy()

    def make_integrator():
        joints = CubicSpline(t, q[:, 7:], axis=0)

        def omega(time, quat, momentum):
            pose = q[takeoff].copy()
            pose[3:7] = quat / np.linalg.norm(quat)
            pose[7:] = joints(time)
            fk(pose)
            mujoco.mj_angmomMat(model, data, momentum_matrix, root)
            return np.linalg.solve(momentum_matrix[:, 3:6], momentum - momentum_matrix[:, 6:] @ joints(time, 1))

        def integrate(momentum):
            def derivative(time, quat):
                w = omega(time, quat, momentum)
                a, v = quat[0], quat[1:]
                return .5 * np.r_[-v @ w, a * w + np.cross(v, w)]
            solution = solve_ivp(derivative, (flight_t[0], flight_t[-1]), initial_quat,
                                 t_eval=flight_t, rtol=2e-7, atol=1e-9, max_step=.005)
            if not solution.success:
                raise RuntimeError(solution.message)
            rotations = solution.y.T
            return rotations / np.linalg.norm(rotations, axis=1, keepdims=True)
        return integrate, omega

    integrate, omega = make_integrator()
    def shoot(momentum):
        rotations = integrate(momentum)
        rotation = Rotation.from_quat(rotations[:, [1, 2, 3, 0]])
        error = (reference_rot.inv() * rotation).as_rotvec()
        return np.r_[error[1:-1].reshape(-1) * .15, error[-1] * 6]

    fit = least_squares(shoot, np.array([4., 24., 10.]), max_nfev=35, diff_step=1e-4,
                        ftol=1e-7, xtol=1e-7, gtol=1e-7)
    momentum = fit.x
    rotations = integrate(momentum)
    print("First aerial shooting complete", flush=True)
    # Root angular velocity must also join continuously at push-off. Apply a
    # local Hermite correction to the preceding root rotations, then refit contact.
    v_old = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, v_old, 2 * dt, original[takeoff - 1], original[takeoff + 1])
    delta_omega = omega(t[takeoff], initial_quat, momentum) - v_old[3:6]
    for frame in range(takeoff):
        u = frame / takeoff
        correction = Rotation.from_rotvec(delta_omega * (t[frame] - t[takeoff]) * u**2)
        original_rotation = Rotation.from_quat(original[frame, [4, 5, 6, 3]])
        q[frame, 3:7] = (original_rotation * correction).as_quat()[[3, 0, 1, 2]]
    solve_stance()
    print("Contact refit complete; final aerial shooting", flush=True)
    # Update the joint interpolation after the contact refit, shoot once more.
    integrate, omega = make_integrator()
    fit = least_squares(shoot, momentum, max_nfev=25, diff_step=1e-4)
    momentum = fit.x
    rotations = integrate(momentum)
    q[takeoff:landing + 1, 3:7] = rotations
    com_targets = target_takeoff + (flight_t - flight_t[0])[:, None] * velocity
    com_targets[:, 2] -= .5 * 9.81 * (flight_t - flight_t[0])**2
    for frame, target in zip(range(takeoff, landing + 1), com_targets):
        q[frame, :3] += target - fk(q[frame])
    # Translate the later recovery consistently with the new landing position.
    offset = q[landing, :3] - original[landing, :3]
    for frame in range(landing + 1, n):
        q[frame, :3] += offset

    args.out.mkdir(parents=True, exist_ok=False)
    csv = np.c_[q[:, :3], q[:, [4, 5, 6, 3]], q[:, 7:]]
    np.savetxt(args.out / "webster.csv", csv, delimiter=",", fmt="%.10f")
    # Use the existing conversion, with floor/joint repairs disabled: this audit
    # must expose problems rather than silently alter the optimized candidate.
    convert_csv_to_npz(args.out / "webster.csv", args.out / "webster.npz", input_fps=fps, output_fps=fps)
    com = np.array([fk(row) for row in q])
    before_v = (original_com[takeoff] - original_com[takeoff - 1]) / dt
    after_v = (original_com[takeoff + 1] - original_com[takeoff]) / dt
    new_before = (com[takeoff] - com[takeoff - 1]) / dt
    new_after = (com[takeoff + 1] - com[takeoff]) / dt
    orientation_error = (reference_rot.inv() * Rotation.from_quat(rotations[:, [1, 2, 3, 0]])).magnitude()
    report = {
        "source": str(args.input.resolve()), "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "frames": n, "fps": fps, "takeoff_frame": takeoff, "landing_frame": landing,
        "neutralized_low_torque_joints": locked,
        "constant_flight_angular_momentum": momentum.tolist(),
        "flight_orientation_change_max_degrees": float(np.rad2deg(orientation_error.max())),
        "landing_orientation_change_degrees": float(np.rad2deg(orientation_error[-1])),
        "source_takeoff_com_velocity_jump_m_per_s": (after_v - before_v).tolist(),
        "candidate_takeoff_com_velocity_jump_m_per_s": (new_after - new_before).tolist(),
        "takeoff_com_velocity": velocity.tolist(), "stance_fits": stance_fit,
        "max_support_error_m": max(row["support_error_m"] for row in stance_fit[-takeoff - 1:]),
        "max_stance_com_error_m": max(row["com_error_m"] for row in stance_fit[-takeoff - 1:]),
        "limitations": "Centroidal/contact projection candidate. Requires independent collision, momentum, actuator/contact-force audit and policy evaluation. Landing impulse is not optimized.",
    }
    (args.out / "projection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k != "stance_fits"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
