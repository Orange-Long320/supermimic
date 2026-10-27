"""Fit a robot reference's root to observed contacts and free-flight COM.

Run with scripts/robot.py x2_ultra in the main server environment. This changes
only root translation; recovered orientation and all actuated joints are kept.
Contact timing is an explicit source-video annotation, never inferred silently.
The result is a candidate for simulation, not a dynamically validated motion.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def kinematics(model, raw, foot_geometry):
    data = mujoco.MjData(model)
    com, soles = [], []
    root_body = int(model.jnt_bodyid[0])
    for frame in raw:
        data.qpos[:3] = frame[:3]
        data.qpos[3:7] = frame[[6, 3, 4, 5]]
        data.qpos[7:] = frame[7:]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        com.append(data.subtree_com[root_body].copy())
        soles.append(min(float((vertices @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])[:, 2].min()) for g, vertices in foot_geometry))
    return np.array(com), np.array(soles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--contacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--output-fps", type=float, default=50)
    parser.add_argument("--input-fps", type=float)
    args = parser.parse_args()
    from supermimic.data.motion_convert import load_csv, load_g1_model, resample_motion

    annotation = json.loads(args.contacts.read_text())
    source_raw = load_csv(args.input)
    if args.output_fps <= 0:
        parser.error("output-fps must be positive")
    sampled = resample_motion(source_raw, args.input_fps or annotation["reference_fps"], args.output_fps)
    raw = np.concatenate([sampled.root_pos, sampled.root_quat[:, [1, 2, 3, 0]], sampled.joint_pos], axis=1)
    model = load_g1_model()
    feet = []
    for geom in range(model.ngeom):
        if not model.geom(geom).name.startswith("foot_"):
            continue
        mesh = model.geom_dataid[geom]
        if mesh < 0:
            raise ValueError("Expected established mesh foot collision geometry")
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        feet.append((geom, model.mesh_vert[start:start + count].copy()))
    if not feet:
        raise ValueError("No robot foot collision meshes")
    fps, gravity = args.output_fps, annotation["gravity_m_per_s2"]
    takeoff = round(annotation["takeoff_reference_frame"] / annotation["reference_fps"] * fps)
    landing = round(annotation["landing_reference_frame"] / annotation["reference_fps"] * fps)
    if not np.allclose(model.opt.gravity, [0, 0, -gravity]):
        raise ValueError("Annotated gravity differs from the robot model")
    if not 0 < takeoff < landing < len(raw) - 1:
        raise ValueError("Contact frames outside the reference")
    clearance = annotation["sole_clearance_m"]
    com, sole = kinematics(model, raw, feet)
    relative_com = com - raw[:, :3]
    corrected = raw.copy()
    supported = np.ones(len(raw), dtype=bool)
    supported[takeoff + 1:landing] = False
    corrected[supported, 2] += clearance - sole[supported]
    start_com = corrected[takeoff, :3] + relative_com[takeoff]
    end_com = corrected[landing, :3] + relative_com[landing]
    duration = (landing - takeoff) / fps
    velocity = (end_com - start_com) / duration
    velocity[2] += 0.5 * gravity * duration
    times = np.arange(landing - takeoff + 1) / fps
    target_com = start_com + times[:, None] * velocity
    target_com[:, 2] -= 0.5 * gravity * times**2
    corrected[takeoff:landing + 1, :3] = target_com - relative_com[takeoff:landing + 1]
    final_com, final_sole = kinematics(model, corrected, feet)
    flight_acceleration = np.diff(final_com[takeoff:landing + 1], n=2, axis=0) * fps**2
    error = float(np.max(np.abs(flight_acceleration - np.array([0, 0, -gravity]))))
    report = {
        "source": str(args.input.resolve()), "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "contacts": annotation, "frames": len(raw), "fps": fps,
        "takeoff_output_frame": takeoff, "landing_output_frame": landing,
        "takeoff_com_velocity_m_per_s": velocity.tolist(), "flight_duration_s": duration,
        "initial_minimum_sole_height_m": float(final_sole[0]),
        "flight_minimum_sole_height_m": float(final_sole[takeoff + 1:landing].min()),
        "flight_com_acceleration_max_error_m_per_s2": error,
        "max_root_translation_change_m": float(np.linalg.norm(corrected[:, :3] - raw[:, :3], axis=1).max()),
        "resampled_root_orientation_and_joint_angles_preserved": bool(np.array_equal(raw[:, 3:], corrected[:, 3:])),
        "limitations": "Contact timing and camera depth are estimates; angular momentum, actuator feasibility and policy success are not validated.",
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "fit.json").write_text(json.dumps(report, indent=2) + "\n")
    if final_sole[takeoff + 1:landing].min() < -0.005 or error > 1e-4:
        raise ValueError("Contact/ballistic candidate rejected; see fit.json")
    np.savetxt(args.out / "webster.csv", corrected, delimiter=",", fmt="%.10f")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
