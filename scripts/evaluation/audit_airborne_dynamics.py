"""CPU diagnostic: airborne centroidal momentum from reference poses.

Run on the server with the selected robot. Finite differences of an estimated
video motion are approximate; this is a consistency audit, not inverse control.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--contacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from supermimic.data.motion_convert import load_g1_model
    model = load_g1_model()
    data = mujoco.MjData(model)
    motion = np.load(args.npz)
    contact = json.loads(args.contacts.read_text())
    fps = float(motion["fps"].reshape(-1)[0])
    start = round(contact["takeoff_reference_frame"] / contact["reference_fps"] * fps)
    end = round(contact["landing_reference_frame"] / contact["reference_fps"] * fps)
    root_body = int(model.jnt_bodyid[0])
    root = motion["body_names"].tolist().index(model.body(root_body).name)
    poses = np.concatenate([
        motion["body_pos_w"][:, root], motion["body_quat_w"][:, root], motion["joint_pos"]
    ], axis=1).astype(float)
    results = []
    # Two finite-difference widths check that large changes are not just a
    # single-frame differentiation artifact. Exclude contact boundary stencils.
    for radius in (1, 2):
        frames, momentum = [], []
        for frame in range(start + radius + 1, end - radius):
            data.qpos[:] = poses[frame]
            mujoco.mj_differentiatePos(
                model, data.qvel, 2 * radius / fps,
                poses[frame - radius], poses[frame + radius],
            )
            # mj_differentiatePos expresses angular velocity in the first
            # quaternion's frame; the momentum matrix uses this sample's frame.
            previous = Rotation.from_quat(poses[frame - radius, [4, 5, 6, 3]])
            current = Rotation.from_quat(poses[frame, [4, 5, 6, 3]])
            data.qvel[3:6] = (current.inv() * previous).apply(data.qvel[3:6])
            mujoco.mj_forward(model, data)
            mujoco.mj_subtreeVel(model, data)
            frames.append(frame)
            momentum.append(data.subtree_angmom[root_body].copy())
        momentum = np.array(momentum)
        centre = momentum.mean(axis=0)
        span = np.ptp(momentum, axis=0)
        results.append({
            "difference_radius_frames": radius,
            "frames": frames,
            "angular_momentum_world_kg_m2_per_s": momentum.tolist(),
            "mean_angular_momentum": centre.tolist(),
            "axis_peak_to_peak": span.tolist(),
            "max_deviation_from_mean": float(np.linalg.norm(momentum - centre, axis=1).max()),
            "mean_magnitude": float(np.linalg.norm(momentum, axis=1).mean()),
            "external_torque_estimate_peak_Nm": float(np.linalg.norm(np.diff(momentum, axis=0) * fps, axis=1).max()),
        })
    report = {
        "npz": str(args.npz.resolve()), "fps": fps,
        "takeoff_frame": start, "landing_frame": end,
        "robot_mass_kg": float(model.body_subtreemass[root_body]),
        "angular_momentum_checks": results,
        "interpretation": "For contact-free flight under uniform gravity, total angular momentum about COM should be constant. Large variation flags reference dynamic inconsistency; numerical differentiation and estimated pose errors affect values. Not a proof of the policy failure cause or actuator feasibility.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        json.dump(report, output, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
