"""Align a reference with the same 0.9 joint soft limits used by X2 resets.

Preserve each frame's COM while clipping joints, then regenerate all velocities.
This is preprocessing only; collision/dynamic QA must follow.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    from supermimic.data.motion_convert import load_g1_model, convert_csv_to_npz
    from supermimic.robots.assets import get_robot_cfg
    m = load_g1_model()
    d = mujoco.MjData(m)
    z = np.load(args.input)
    root = int(m.jnt_bodyid[0])
    ri = z["body_names"].tolist().index(m.body(root).name)
    q = np.c_[z["body_pos_w"][:, ri], z["body_quat_w"][:, ri], z["joint_pos"]].astype(float)
    source = q.copy()
    factor = get_robot_cfg().articulation.soft_joint_pos_limit_factor
    limits = m.jnt_range[1:]
    centre = limits.mean(axis=1)
    lower = centre - factor * (limits[:, 1] - limits[:, 0]) / 2 + 1e-5
    upper = centre + factor * (limits[:, 1] - limits[:, 0]) / 2 - 1e-5
    for row in q:
        d.qpos[:] = row
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        com = d.subtree_com[root].copy()
        row[7:] = np.clip(row[7:], lower, upper)
        d.qpos[:] = row
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        row[:3] += com - d.subtree_com[root]
    args.out.mkdir(parents=True, exist_ok=False)
    np.savetxt(args.out / "webster.csv", np.c_[q[:, :3], q[:, [4, 5, 6, 3]], q[:, 7:]], delimiter=",", fmt="%.10f")
    fps = float(z["fps"].reshape(-1)[0])
    convert_csv_to_npz(args.out / "webster.csv", args.out / "webster.npz", fps, fps)
    report = {"source": str(args.input.resolve()), "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "soft_limit_factor": factor, "max_joint_change_rad": float(np.abs(q[:, 7:] - source[:, 7:]).max()),
              "initial_max_joint_change_rad": float(np.abs(q[0, 7:] - source[0, 7:]).max()),
              "max_com_preserving_root_adjustment_m": float(np.linalg.norm(q[:, :3] - source[:, :3], axis=1).max()),
              "limitations": "Limits aligned with reset behavior. Independent collision/contact/force and policy checks required."}
    (args.out / "soft-limit-alignment.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
