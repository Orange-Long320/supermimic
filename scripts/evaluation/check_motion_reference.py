"""Check robot floor clearance, joint limits and airborne COM from a final NPZ."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", required=True)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--contacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--joint-limit-policy",choices=("soft","hard"),default="soft",
                        help="Report both; explicitly select whether soft posture limits are mandatory")
    args = parser.parse_args()
    from supermimic.robots.paths import select_robot
    select_robot(args.robot, check_assets=True)
    from supermimic.data.motion_convert import load_g1_model
    model = load_g1_model()
    data = mujoco.MjData(model)
    motion = np.load(args.npz)
    annotation = json.loads(args.contacts.read_text())
    fps = float(motion["fps"].reshape(-1)[0])
    takeoff = round(annotation["takeoff_reference_frame"] / annotation["reference_fps"] * fps)
    landing = round(annotation["landing_reference_frame"] / annotation["reference_fps"] * fps)
    body_id = int(model.jnt_bodyid[0])
    root = motion["body_names"].tolist().index(model.body(body_id).name)
    minimum = (float("inf"), None, None)
    nonfoot = (float("inf"), None, None)
    com = []
    for frame in range(len(motion["joint_pos"])):
        data.qpos[:3] = motion["body_pos_w"][frame, root]
        data.qpos[3:7] = motion["body_quat_w"][frame, root]
        data.qpos[7:] = motion["joint_pos"][frame]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        com.append(data.subtree_com[body_id].copy())
        for geom in range(model.ngeom):
            if model.geom_bodyid[geom] == 0 or not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                continue
            matrix = data.geom_xmat[geom].reshape(3, 3)
            mesh = model.geom_dataid[geom]
            kind = model.geom_type[geom]
            if mesh >= 0:
                start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                height = float((model.mesh_vert[start:start + count] @ matrix.T + data.geom_xpos[geom])[:, 2].min())
            elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                height = float(data.geom_xpos[geom, 2] - model.geom_size[geom, 0])
            elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                height = float(data.geom_xpos[geom, 2] - abs(matrix[2, 2]) * model.geom_size[geom, 1] - model.geom_size[geom, 0])
            elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                height = float(data.geom_xpos[geom, 2] - np.abs(matrix[2]) @ model.geom_size[geom])
            else:
                raise ValueError(f"Unsupported collision geometry: {kind}")
            name = model.geom(geom).name
            if height < minimum[0]:
                minimum = (height, frame, name)
            if not name.startswith("foot_") and height < nonfoot[0]:
                nonfoot = (height, frame, name)
    com = np.array(com)
    acceleration = np.diff(com[takeoff:landing + 1], n=2, axis=0) * fps**2
    error = float(np.max(np.abs(acceleration - np.array([0, 0, -annotation["gravity_m_per_s2"]]))))
    values = motion["joint_pos"]
    limits = model.jnt_range[1:]
    violation = float(np.maximum(np.maximum(limits[:, 0] - values, values - limits[:, 1]), 0).max())
    from supermimic.robots.assets import get_robot_cfg
    factor = get_robot_cfg().articulation.soft_joint_pos_limit_factor
    centre, half = limits.mean(axis=1), factor * (limits[:, 1] - limits[:, 0]) / 2
    soft_violation = float(np.maximum(np.maximum(centre - half - values, values - centre - half), 0).max())
    finite = all(np.isfinite(motion[key]).all() for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"))
    report = {"frames": len(com), "fps": fps, "finite": bool(finite),
              "minimum_collision_height_m": minimum, "minimum_nonfoot_collision_height_m": nonfoot,
              "flight_com_acceleration_max_error_m_per_s2": error, "joint_limit_violation_rad": violation,
              "soft_joint_limit_factor": factor, "soft_joint_limit_violation_rad": soft_violation,
              "joint_limit_policy":args.joint_limit_policy,
              "approved_for_bounded_simulation": bool(finite and minimum[0] > -.005 and nonfoot[0] > .005 and error < .02 and violation < 1e-4 and (args.joint_limit_policy=="hard" or soft_violation < 1e-4)),
              "limitations": "Geometric/contact/COM checks only; angular momentum, actuator feasibility and policy success remain unvalidated."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["approved_for_bounded_simulation"]:
        raise SystemExit("Reference rejected before training")


if __name__ == "__main__":
    main()
