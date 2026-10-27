"""Bounded arm IK for collision-free video-derived references; server CPU only."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=50)
    args = parser.parse_args()
    from supermimic.data.motion_convert import load_csv, load_g1_model, G1_JOINT_NAMES
    model = load_g1_model()
    data = mujoco.MjData(model)
    raw = load_csv(args.input)
    corrected = raw.copy()
    changes = []
    for side in ("left", "right"):
        ids = [i for i, name in enumerate(G1_JOINT_NAMES) if name.startswith(side + "_") and any(part in name for part in ("shoulder", "elbow", "wrist"))]
        body_ids = {i for i in range(model.nbody) if model.body(i).name.startswith(side + "_") and any(part in model.body(i).name for part in ("shoulder", "elbow", "wrist"))}
        geometries = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] not in body_ids or not (model.geom_contype[g] or model.geom_conaffinity[g]):
                continue
            mesh = model.geom_dataid[g]
            if mesh < 0:
                raise ValueError("Expected existing arm collision meshes")
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            geometries.append((g, model.mesh_vert[start:start + count].copy()))
        limits = model.jnt_range[[model.joint(G1_JOINT_NAMES[i]).id for i in ids]]
        for frame in range(len(raw)):
            data.qpos[:3] = corrected[frame, :3]
            data.qpos[3:7] = corrected[frame, [6, 3, 4, 5]]
            data.qpos[7:] = corrected[frame, 7:]
            original = corrected[frame, 7 + np.array(ids)].copy()

            def heights(x):
                data.qpos[7 + np.array(ids)] = x
                mujoco.mj_kinematics(model, data)
                return np.array([float((vertices @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])[:, 2].min()) for g, vertices in geometries])

            before = heights(original)
            if before.min() >= 0.015:
                continue
            lower = np.maximum(limits[:, 0], original - 0.6)
            upper = np.minimum(limits[:, 1], original + 0.6)

            def residual(x):
                return np.concatenate([(x - original) * 0.1, np.maximum(0.02 - heights(x), 0) * 80])

            solution = least_squares(residual, np.clip(original, lower, upper), bounds=(lower, upper), diff_step=1e-4, max_nfev=100, ftol=1e-9, xtol=1e-9, gtol=1e-9)
            after = heights(solution.x)
            if after.min() < 0.015:
                raise ValueError(f"Bounded {side} arm clearance infeasible at frame {frame}: {after.min()}")
            corrected[frame, 7 + np.array(ids)] = solution.x
            changes.append({"frame": frame, "side": side, "before_min_m": float(before.min()), "after_min_m": float(after.min()), "max_joint_change_rad": float(np.abs(solution.x - original).max())})
    args.out.mkdir(parents=True, exist_ok=False)
    np.savetxt(args.out / "webster.csv", corrected, delimiter=",", fmt="%.10f")
    report = {
        "source": str(args.input.resolve()), "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "fps": args.fps, "frames": len(raw), "corrected_frames": changes,
        "max_joint_change_rad": float(np.abs(corrected[:, 7:] - raw[:, 7:]).max()),
        "joint_speed_p99_rad_per_s": float(np.quantile(np.abs(np.gradient(corrected[:, 7:], 1 / args.fps, axis=0)), .99)),
        "root_pose_preserved": bool(np.array_equal(corrected[:, :7], raw[:, :7])),
        "method": "Bounded least-squares arm IK with existing collision meshes; floor margin 15mm, joint changes <=0.6rad; root and leg poses preserved",
    }
    (args.out / "arm-floor-fit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
