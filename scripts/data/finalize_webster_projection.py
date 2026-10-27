"""Preserve a fitted stance, join COM ballistics exactly, align recovery soles.

No rotational or joint repairs; independent QA remains mandatory.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--contacts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    from supermimic.data.motion_convert import load_csv, load_g1_model, convert_csv_to_npz
    raw = load_csv(args.input)
    q = np.c_[raw[:, :3], raw[:, [6, 3, 4, 5]], raw[:, 7:]]
    model = load_g1_model()
    data = mujoco.MjData(model)
    ann = json.loads(args.contacts.read_text())
    fps = 50.
    start = round(ann["takeoff_reference_frame"] / ann["reference_fps"] * fps)
    end = round(ann["landing_reference_frame"] / ann["reference_fps"] * fps)
    root = int(model.jnt_bodyid[0])
    feet = []
    for g in range(model.ngeom):
        if model.geom(g).name.startswith("foot_"):
            mesh = model.geom_dataid[g]
            a, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            feet.append((g, model.mesh_vert[a:a + count]))

    def fk(row):
        data.qpos[:] = row
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        sole = min((v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])[:, 2].min() for g, v in feet)
        return data.subtree_com[root].copy(), float(sole)

    for i in range(end, len(q)):
        _, sole = fk(q[i])
        q[i, 2] += .005 - sole
    a, b = fk(q[start])[0], fk(q[end])[0]
    duration = (end - start) / fps
    velocity = (b - a) / duration
    velocity[2] += .5 * 9.81 * duration
    for i in range(start + 1, end):
        time = (i - start) / fps
        target = a + time * velocity - np.array([0, 0, .5 * 9.81 * time**2])
        q[i, :3] += target - fk(q[i])[0]
    args.out.mkdir(parents=True, exist_ok=False)
    np.savetxt(args.out / "webster.csv", np.c_[q[:, :3], q[:, [4, 5, 6, 3]], q[:, 7:]], delimiter=",", fmt="%.10f")
    convert_csv_to_npz(args.out / "webster.csv", args.out / "webster.npz", fps, fps)
    com = np.array([fk(row)[0] for row in q])
    oldq = np.c_[raw[:, :3], raw[:, [6, 3, 4, 5]], raw[:, 7:]]
    report = {"source": str(args.input.resolve()), "takeoff_velocity": velocity.tolist(),
              "takeoff_com_discrete_velocity_jump": ((com[start + 1] - 2 * com[start] + com[start - 1]) * fps).tolist(),
              "max_root_translation_adjustment_m": float(np.linalg.norm(q[:, :3] - oldq[:, :3], axis=1).max()),
              "joint_and_rotation_preserved": bool(np.array_equal(q[:, 3:], oldq[:, 3:])),
              "limitations": "Landing contact impulses remain unoptimized. Motor/force constraints require an independent audit."}
    (args.out / "finalize.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
