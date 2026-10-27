"""Robot-adapt Webster preparation, preserving the aerial motion's duration.

The user permits retiming. A linear aerial COM-height change preserves gravity
and adds only constant translational velocity; landing height stays unchanged.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import mujoco
from scipy.interpolate import PchipInterpolator, CubicHermiteSpline
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation, RotationSpline


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--contacts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--preparation-seconds", type=float, default=.6)
    p.add_argument("--takeoff-com-lift", type=float, default=.15)
    p.add_argument("--coupled-initialization", action="store_true")
    p.add_argument("--interpolation-only", action="store_true")
    p.add_argument("--ease-in", action="store_true", help="Preserve original launch speed while stretching preparation")
    args = p.parse_args()
    from supermimic.data.motion_convert import load_g1_model, convert_csv_to_npz
    m = load_g1_model()
    z = np.load(args.input)
    fps = float(z["fps"].reshape(-1)[0])
    annotation = json.loads(args.contacts.read_text())
    old_takeoff = round(annotation["takeoff_reference_frame"] / annotation["reference_fps"] * fps)
    old_landing = round(annotation["landing_reference_frame"] / annotation["reference_fps"] * fps)
    new_takeoff = round(args.preparation_seconds * fps)
    if new_takeoff <= old_takeoff:
        p.error("Preparation must be longer than the original")
    root = z["body_names"].tolist().index(m.body(int(m.jnt_bodyid[0])).name)
    raw = np.c_[z["body_pos_w"][:, root], z["body_quat_w"][:, root], z["joint_pos"]].astype(float)
    sample_time = np.linspace(0, old_takeoff / fps, new_takeoff + 1)
    if args.ease_in:
        ratio = (new_takeoff / fps) / (old_takeoff / fps)
        sample_time = old_takeoff / fps * (np.arange(new_takeoff + 1) / new_takeoff)**ratio
    source_time = np.arange(old_takeoff + 1) / fps
    scalar = PchipInterpolator(source_time, np.c_[raw[:old_takeoff + 1, :3], raw[:old_takeoff + 1, 7:]], axis=0)(sample_time)
    rotation = RotationSpline(source_time, Rotation.from_quat(raw[:old_takeoff + 1, [4, 5, 6, 3]]))(sample_time)
    support = np.c_[scalar[:, :3], rotation.as_quat()[:, [3, 0, 1, 2]], scalar[:, 3:]]
    support[:, 2] += args.takeoff_com_lift * (np.arange(new_takeoff + 1) / new_takeoff)**2
    q = np.r_[support, raw[old_takeoff + 1:]]
    offset = new_takeoff - old_takeoff
    new_landing = old_landing + offset
    for frame in range(new_takeoff + 1, new_landing + 1):
        q[frame, 2] += args.takeoff_com_lift * (new_landing - frame) / (new_landing - new_takeoff)
    if args.interpolation_only:
        args.out.mkdir(parents=True, exist_ok=False)
        np.savetxt(args.out / "webster.csv", np.c_[q[:, :3], q[:, [4, 5, 6, 3]], q[:, 7:]], delimiter=",", fmt="%.10f")
        convert_csv_to_npz(args.out / "webster.csv", args.out / "webster.npz", fps, fps)
        annotation.update({"reference_fps": fps, "takeoff_reference_frame": new_takeoff,
                           "landing_reference_frame": new_landing,
                           "retiming": {"preparation_seconds": args.preparation_seconds,
                                        "airborne_duration_preserved": True,
                                        "takeoff_com_lift_m": args.takeoff_com_lift,
                                        "ease_in": args.ease_in,
                                        "launch_time_scale_derivative": 1. if args.ease_in else old_takeoff / new_takeoff},
                           "method": "Smooth original poses used only as an unconstrained initial guess; no per-frame leg IK."})
        (args.out / "contacts.json").write_text(json.dumps(annotation, indent=2))
        return
    # Initialize contact-constrained leg extension at the requested COM height.
    # Without this step, raising the pelvis only creates floating support feet.
    d = mujoco.MjData(m)
    root_body = int(m.jnt_bodyid[0])
    feet = []
    for g in range(m.ngeom):
        if m.geom(g).name.startswith("foot_"):
            mesh = m.geom_dataid[g]
            a, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
            vertices = m.mesh_vert[a:a + count].astype(float)
            feet.append((g, vertices[ConvexHull(vertices).vertices]))

    def fk(pose):
        d.qpos[:] = pose
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        return d.subtree_com[root_body].copy()

    switch = round(5 * new_takeoff / old_takeoff)
    first_com, launch_com = fk(q[0]), fk(q[new_takeoff])
    after1, after2 = fk(q[new_takeoff + 1]), fk(q[new_takeoff + 2])
    launch_velocity = (4 * after1 - after2 - 3 * launch_com) * fps / 2
    initial_velocity = (launch_com - first_com) / args.preparation_seconds
    initial_velocity[2] = 0. if args.coupled_initialization else -.35
    stance_com = CubicHermiteSpline([0, args.preparation_seconds], [first_com, launch_com],
                                    [initial_velocity, launch_velocity])(np.arange(new_takeoff + 1) / fps)
    anchors = {}
    from supermimic.data.centroidal_projection import sole_patches
    patches = sole_patches(m) if args.coupled_initialization else {}
    for frame, side in ((0, "right"), (switch, "left")):
        fk(q[frame])
        g, vertices = next(x for x in feet if m.body(m.geom_bodyid[x[0]]).name.startswith(side))
        world = vertices @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        k = world[:, 2].argmin()
        local = patches[g][0] if args.coupled_initialization else vertices[k]
        anchor = d.geom_xmat[g].reshape(3, 3) @ local + d.geom_xpos[g]
        anchor[2] = 0.
        anchors[side] = g, local, anchor
    swing_targets = []
    for frame in range(new_takeoff + 1):
        fk(q[frame])
        points = {}
        for side in ("left", "right"):
            g, local, _ = anchors[side]
            points[side] = d.geom_xmat[g].reshape(3, 3) @ local + d.geom_xpos[g]
        swing_targets.append(points)
    left_start = swing_targets[0]["left"].copy()
    right_end = swing_targets[-1]["right"].copy()
    right_g, right_local, right_anchor = anchors["right"]
    right_after = []
    for frame in (new_takeoff + 1, new_takeoff + 2):
        fk(q[frame])
        right_after.append(d.geom_xmat[right_g].reshape(3, 3) @ right_local + d.geom_xpos[right_g])
    right_velocity = (4 * right_after[0] - right_after[1] - 3 * right_end) * fps / 2
    left_curve = CubicHermiteSpline([0, switch / fps], [left_start, anchors["left"][2]], [np.zeros(3), np.zeros(3)])
    right_curve = CubicHermiteSpline([switch / fps, new_takeoff / fps], [right_anchor, right_end], [np.zeros(3), right_velocity])
    columns = np.r_[np.arange(3), np.arange(7, 19)]
    joint_centre = m.jnt_range[1:13].mean(axis=1)
    half = .45 * np.diff(m.jnt_range[1:13], axis=1).ravel()
    lower = np.r_[[-5, -5, .1], joint_centre - half + 1e-5]
    upper = np.r_[[5, 5, 2], joint_centre + half - 1e-5]
    def fit_stance():
        errors = []
        previous_legs = q[0, 7:19].copy()
        for frame in range(new_takeoff + 1):
            target = stance_com[frame]
            side = "right" if frame < switch else "left"
            reference = q[frame, columns].copy()

            def residual(x):
                row = q[frame].copy()
                row[columns] = x
                com = fk(row)
                points = {}
                for limb in ("left", "right"):
                    g, local, _ = anchors[limb]
                    points[limb] = d.geom_xmat[g].reshape(3, 3) @ local + d.geom_xpos[g]
                swing = "left" if side == "right" else "right"
                swing_goal = left_curve(frame / fps) if swing == "left" else right_curve(frame / fps)
                heights = np.array([(v @ d.geom_xmat[k].reshape(3, 3).T + d.geom_xpos[k])[:, 2].min() for k, v in feet])
                return np.r_[(com - target) * 1000, (points[side] - anchors[side][2]) * 1000,
                             (points[swing] - swing_goal) * (500 if args.coupled_initialization else 0),
                             np.minimum(heights + .0005, 0) * 1000, (x - reference) * .05,
                             (x[3:] - previous_legs) * .1]

            fit = least_squares(residual, np.clip(reference, lower, upper), bounds=(lower, upper), max_nfev=60, diff_step=1e-5)
            q[frame, columns] = fit.x
            previous_legs = fit.x[3:].copy()
            error = residual(fit.x)
            errors.append({"frame": frame, "com_error_m": float(np.linalg.norm(error[:3]) / 1000),
                           "contact_error_m": float(np.linalg.norm(error[3:6]) / 1000)})
        return errors

    init_errors = fit_stance()
    # The pushed-off leg starts extended and tucks over 0.12s. Preserve the
    # intended ballistic COM while changing its internal limb configuration.
    air_template = q.copy()
    projection_reports = []
    for refinement in range(2 if args.coupled_initialization else 1):
        if args.coupled_initialization:
            v_start = (3 * q[new_takeoff, 7:13] - 4 * q[new_takeoff - 1, 7:13] + q[new_takeoff - 2, 7:13]) * fps / 2
            v_end = (air_template[new_takeoff + 7, 7:13] - air_template[new_takeoff + 5, 7:13]) * fps / 2
            tuck = CubicHermiteSpline([0, 6 / fps], [q[new_takeoff, 7:13], air_template[new_takeoff + 6, 7:13]], [v_start, v_end])
        for step in range(1, 7):
            frame = new_takeoff + step
            target_com = fk(air_template[frame])
            u = step / 6
            blend = 3 * u**2 - 2 * u**3
            q[frame, 7:13] = tuck(step / fps) if args.coupled_initialization else q[new_takeoff, 7:13] * (1 - blend) + air_template[frame, 7:13] * blend
            q[frame, :3] += target_com - fk(q[frame])
        if args.coupled_initialization:
            from supermimic.data.centroidal_projection import project_flight
            q, h, omega = project_flight(m, q, new_takeoff, new_landing, fps, np.array([5.6, 25.2, -8.3]))
            projection_reports.append({"angular_momentum": h.tolist(), "launch_angular_velocity_body": omega.tolist()})
            if refinement == 0:
                r0 = Rotation.from_quat(q[0, [4, 5, 6, 3]])
                rt = Rotation.from_quat(q[new_takeoff, [4, 5, 6, 3]])
                tangent = (r0.inv() * rt).as_rotvec()
                theta = np.linalg.norm(tangent)
                cross = np.array([[0, -tangent[2], tangent[1]], [tangent[2], 0, -tangent[0]], [-tangent[1], tangent[0], 0]])
                jr = np.eye(3) if theta < 1e-7 else np.eye(3) - (1 - np.cos(theta)) / theta**2 * cross + (theta - np.sin(theta)) / theta**3 * cross @ cross
                prefix = CubicHermiteSpline([0, args.preparation_seconds], [np.zeros(3), tangent], [np.zeros(3), np.linalg.solve(jr, omega)])
                q[:new_takeoff, 3:7] = (r0 * Rotation.from_rotvec(prefix(np.arange(new_takeoff) / fps))).as_quat()[:, [3, 0, 1, 2]]
                init_errors = fit_stance()
    args.out.mkdir(parents=True, exist_ok=False)
    np.savetxt(args.out / "webster.csv", np.c_[q[:, :3], q[:, [4, 5, 6, 3]], q[:, 7:]], delimiter=",", fmt="%.10f")
    convert_csv_to_npz(args.out / "webster.csv", args.out / "webster.npz", fps, fps)
    annotation.update({"reference_fps": fps, "takeoff_reference_frame": new_takeoff,
                       "landing_reference_frame": new_landing,
                       "retiming": {"original_preparation_seconds": old_takeoff / fps,
                                    "preparation_seconds": new_takeoff / fps,
                                    "takeoff_com_lift_m": args.takeoff_com_lift,
                                    "airborne_duration_preserved": True,
                                    "aerial_vertical_velocity_delta_m_per_s": -args.takeoff_com_lift / ((old_landing - old_takeoff) / fps)},
                       "method": "Robot-adapted timing/height candidate; contact and motor constrained optimization must follow."})
    (args.out / "contacts.json").write_text(json.dumps(annotation, indent=2))
    report = {"source": str(args.input.resolve()), "frames": len(q), "fps": fps,
              "takeoff_frame": new_takeoff, "landing_frame": new_landing,
              "switch_frame": switch, "leg_extension_initialization": init_errors,
              "retiming": annotation["retiming"], "launch_com_velocity": launch_velocity.tolist(),
              "tuck_blend_seconds": 6 / fps,
              "centroidal_projections": projection_reports,
              "limitation": "Contact/COM leg initialization only; requires unified motor/dynamics/collision checks before training."}
    all_limits = m.jnt_range[1:]
    centre = all_limits.mean(axis=1)
    half = .45 * np.diff(all_limits, axis=1).ravel()
    report["soft_limit_violation_rad"] = float(np.maximum(np.maximum(centre - half - q[:, 7:], q[:, 7:] - centre - half), 0).max())
    report["maximum_adjacent_joint_change_rad"] = float(np.abs(np.diff(q[:, 7:], axis=0)).max())
    (args.out / "retiming.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
