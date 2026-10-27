"""Project aerial interior poses onto free-flight dynamics and motor limits.

Server CPU; takeoff pose, first aerial pose and landing boundary stay fixed.
Waist compliance is allowed rather than demanding an infinitely stiff torso.
"""
import argparse
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
    p.add_argument("--max-evaluations", type=int, default=60)
    p.add_argument("--boundary-source", type=Path)
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
    ri = z["body_names"].tolist().index(m.body(root).name)
    base = np.c_[z["body_pos_w"][:, ri], z["body_quat_w"][:, ri], z["joint_pos"]].astype(float)
    active = np.arange(18, 39)
    checked = np.arange(17, 40)
    active_index = {int(frame): i for i, frame in enumerate(active)}
    if args.boundary_source:
        boundary = np.load(args.boundary_source)
        boundary_q = np.c_[boundary["body_pos_w"][:, ri], boundary["body_quat_w"][:, ri], boundary["joint_pos"]]
        assert boundary_q.shape == base.shape
        fixed = np.ones(len(base), dtype=bool)
        fixed[active] = False
        base[fixed] = boundary_q[fixed]
    reference_q = boundary_q if args.boundary_source else base
    count, width, nj = len(active), m.nv, m.nv - 6
    rotations = Rotation.from_quat(base[active][:, [4, 5, 6, 3]])
    limits = np.array([PROFILE["effort_limit"][m.joint(j).name] for j in range(1, m.njnt)])
    meshes = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0 or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        mesh = m.geom_dataid[g]
        a, n = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
        v = m.mesh_vert[a:a + n].astype(float)
        meshes.append((g, v[ConvexHull(v).vertices]))

    def derivatives(q):
        v = np.zeros((len(q), m.nv))
        acceleration = np.zeros_like(v)
        forward, backward = np.zeros(m.nv), np.zeros(m.nv)
        for i in range(1, len(q) - 1):
            mujoco.mj_differentiatePos(m, forward, 1 / fps, q[i], q[i + 1])
            mujoco.mj_differentiatePos(m, backward, -1 / fps, q[i], q[i - 1])
            v[i], acceleration[i] = (forward + backward) / 2, (forward - backward) * fps
        v[0], v[-1] = v[1], v[-2]
        return v, acceleration

    v0 = derivatives(reference_q)[0]
    com_target, momentum = [], []
    am = np.zeros((3, m.nv))
    for frame in checked:
        d.qpos[:], d.qvel[:] = reference_q[frame], v0[frame]
        mujoco.mj_forward(m, d)
        mujoco.mj_angmomMat(m, d, am, root)
        com_target.append(d.subtree_com[root].copy())
        momentum.append(am @ v0[frame])
    com_target, target_h = np.array(com_target), np.mean(momentum, axis=0)
    initial = np.c_[base[active, :3], np.zeros((count, 3)), base[active, 7:]]
    centre = m.jnt_range[1:].mean(axis=1)
    half = .45 * np.diff(m.jnt_range[1:], axis=1).ravel()
    lower = np.tile(np.r_[[-5, -5, .1], np.full(3, -.5), centre - half + 1e-5], count)
    upper = np.tile(np.r_[[5, 5, 2], np.full(3, .5), centre + half - 1e-5], count)
    for name in ("head_yaw_joint", "head_pitch_joint"):
        j = m.jnt_dofadr[m.joint(name).id]
        lower[j::width], upper[j::width] = -1e-7, 1e-7
    for name in ("waist_pitch_joint", "waist_roll_joint"):
        j = m.jnt_dofadr[m.joint(name).id]
        lower[j::width] = np.maximum(lower[j::width], -.5)
        upper[j::width] = np.minimum(upper[j::width], .5)
    blocksize = 3 + len(meshes) + 3 + nj + 6 + nj + nj + 3
    latest = []

    def decode(flat):
        x = flat.reshape(count, width)
        q = base.copy()
        q[active, :3] = x[:, :3]
        q[active, 3:7] = (rotations * Rotation.from_rotvec(x[:, 3:6])).as_quat()[:, [3, 0, 1, 2]]
        q[active, 7:] = x[:, 6:]
        v, acceleration = derivatives(q)
        return x, q, v, acceleration

    def residual(flat):
        nonlocal latest
        x, q, v, acceleration = decode(flat)
        blocks, latest = [], []
        mg = m.body_subtreemass[root] * 9.81
        for j, frame in enumerate(checked):
            d.qpos[:], d.qvel[:], d.qacc[:] = q[frame], v[frame], acceleration[frame]
            mujoco.mj_inverse(m, d)
            required = d.qfrc_inverse.copy()
            ratio = required[6:] / limits
            heights = np.array([(vertices @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min() for g, vertices in meshes])
            mujoco.mj_angmomMat(m, d, am, root)
            h = am @ v[frame]
            blocks.append(np.r_[
                (d.subtree_com[root] - com_target[j]) * 100000,
                np.minimum(heights - .008, 0) * 2000,
                (x[active_index[int(frame)], 3:6] if int(frame) in active_index else np.zeros(3)) * .5,
                (q[frame, 7:] - base[frame, 7:]) * .1,
                required[:6] * np.r_[np.full(3, 400 / mg), np.full(3, 800 / mg)],
                np.maximum(np.abs(ratio) - .98, 0) * 80,
                ratio * .02, (h - target_h) * 3,
            ])
            latest.append({"frame": int(frame), "force_residual_N": float(np.linalg.norm(required[:3])),
                           "torque_residual_Nm": float(np.linalg.norm(required[3:6])),
                           "motor_ratio": float(np.abs(ratio).max()),
                           "com_error_m": float(np.linalg.norm(d.subtree_com[root] - com_target[j]))})
        return np.concatenate(blocks)

    sparsity = lil_matrix((len(checked) * blocksize, count * width), dtype=int)
    for i, frame in enumerate(checked):
        for j, variable_frame in enumerate(active):
            if abs(int(variable_frame - frame)) <= 1:
                sparsity[i * blocksize:(i + 1) * blocksize, j * width:(j + 1) * width] = 1
    fit = least_squares(residual, np.clip(initial.ravel(), lower, upper), jac_sparsity=sparsity.tocsr(),
                        bounds=(lower, upper), x_scale="jac", max_nfev=args.max_evaluations,
                        diff_step=1e-4, ftol=2e-5, xtol=1e-6, gtol=1e-5, verbose=2)
    _, q, _, _ = decode(fit.x)
    residual(fit.x)
    args.out.mkdir(parents=True, exist_ok=False)
    np.savetxt(args.out / "webster.csv", np.c_[q[:, :3], q[:, [4, 5, 6, 3]], q[:, 7:]], delimiter=",", fmt="%.10f")
    convert_csv_to_npz(args.out / "webster.csv", args.out / "webster.npz", fps, fps)
    report = {"source": str(args.input.resolve()), "frames": latest, "cost": float(fit.cost), "nfev": fit.nfev,
              "target_world_angular_momentum": target_h.tolist(),
              "limitations": "Local finite-difference inverse-dynamics fit; boundary stencils and landing impacts remain approximate. Independent audit and closed-loop evaluation required."}
    (args.out / "flight-fit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
