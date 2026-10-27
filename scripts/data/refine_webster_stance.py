"""Jointly smooth supported poses with fixed contact points and a C1 COM launch.

Server CPU only. Retains the projected aerial rotations and reports all residuals.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--projection", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--lock-low-torque-joints", action="store_true")
    args = p.parse_args()
    from supermimic.data.motion_convert import load_csv, load_g1_model, convert_csv_to_npz
    m = load_g1_model()
    d = mujoco.MjData(m)
    report = json.loads(args.projection.read_text())
    raw = load_csv(args.input)
    q = np.c_[raw[:, :3], raw[:, [6, 3, 4, 5]], raw[:, 7:]]
    root = int(m.jnt_bodyid[0])
    count = report["takeoff_frame"] + 1
    fps = report["fps"]
    frames = np.arange(count) / fps
    if args.lock_low_torque_joints:
        # Recovered root rotations contain acceleration spikes. Fit one smooth
        # launch rotation that matches the existing aerial orientation and its
        # one-sided angular velocity; do not retain per-frame video jitter.
        start_rotation = Rotation.from_quat(q[0, [4, 5, 6, 3]])
        end_rotation = Rotation.from_quat(q[count - 1, [4, 5, 6, 3]])
        r = (start_rotation.inv() * end_rotation).as_rotvec()
        after1 = (end_rotation.inv() * Rotation.from_quat(q[count, [4, 5, 6, 3]])).as_rotvec()
        after2 = (end_rotation.inv() * Rotation.from_quat(q[count + 1, [4, 5, 6, 3]])).as_rotvec()
        omega = (4 * after1 - after2) * fps / 2
        theta = np.linalg.norm(r)
        cross = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
        right_jacobian = np.eye(3) if theta < 1e-7 else (
            np.eye(3) - (1 - np.cos(theta)) / theta**2 * cross
            + (theta - np.sin(theta)) / theta**3 * cross @ cross
        )
        tangent = CubicHermiteSpline([0, frames[-1]], [np.zeros(3), r],
                                    [np.zeros(3), np.linalg.solve(right_jacobian, omega)])
        rotation = start_rotation * Rotation.from_rotvec(tangent(frames))
        q[:count, 3:7] = rotation.as_quat()[:, [3, 0, 1, 2]]
    meshes = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0 or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        mesh = m.geom_dataid[g]
        a, n = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
        meshes.append((g, m.mesh_vert[a:a + n]))

    def fk(pose):
        d.qpos[:] = pose
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        return d.subtree_com[root].copy()

    anchors = {}
    for frame, side in ((0, "right"), (5, "left")):
        fk(q[frame])
        g, v = next(x for x in meshes if m.geom(x[0]).name.startswith("foot_") and m.body(m.geom_bodyid[x[0]]).name.startswith(side))
        world = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        k = world[:, 2].argmin()
        anchor = world[k].copy()
        anchor[2] = .005
        anchors[side] = (g, v[k], anchor)
    target = CubicHermiteSpline(
        [0, frames[-1]], [fk(q[0]), fk(q[count - 1])],
        [np.r_[report["takeoff_com_velocity"][:2], -.35], report["takeoff_com_velocity"]]
    )(frames)
    indices = np.r_[np.arange(3), np.arange(7, m.nq)]
    reference = q[:count, indices].copy()
    width = len(indices)
    blocksize = 6 + len(meshes) + width
    calls = 0

    def residual(flat):
        nonlocal calls
        x = flat.reshape(count, width)
        blocks = []
        for frame in range(count):
            pose = q[frame].copy()
            pose[indices] = x[frame]
            com = fk(pose)
            g, local, anchor = anchors["right" if frame < 5 else "left"]
            point = d.geom_xmat[g].reshape(3, 3) @ local + d.geom_xpos[g]
            heights = np.array([(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min() for g, v in meshes])
            blocks.append(np.r_[(com - target[frame]) * 1000,
                                (point - anchor) * 1000,
                                np.minimum(heights - .006, 0) * 1000,
                                (x[frame] - reference[frame]) * .1])
        smooth = np.diff(x, n=2, axis=0) * 4
        # Preserve aerial joint position and derivative at the launch boundary.
        boundary = np.r_[(x[-1, 3:] - reference[-1, 3:]) * 50,
                         ((x[-1, 3:] - x[-2, 3:]) - (q[count, 7:] - x[-1, 3:])) * 10]
        calls += 1
        return np.r_[np.concatenate(blocks), smooth.ravel(), boundary]

    rows = count * blocksize + (count - 2) * width + 2 * (width - 3)
    sparsity = lil_matrix((rows, count * width), dtype=int)
    for i in range(count):
        sparsity[i * blocksize:(i + 1) * blocksize, i * width:(i + 1) * width] = 1
    for i in range(count - 2):
        r = count * blocksize + i * width
        sparsity[r:r + width, i * width:(i + 3) * width] = 1
    sparsity[-2 * (width - 3):, -2 * width:] = 1
    # Use MuJoCo's exact point/COM Jacobians. Numerical differences over all
    # vertices are needlessly expensive for this local constrained problem.
    tangent = np.r_[np.arange(3), np.arange(6, m.nv)]

    def jacobian(flat):
        x = flat.reshape(count, width)
        jac = lil_matrix((rows, count * width), dtype=float)
        jp, jr, jc = np.zeros((3, m.nv)), np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for frame in range(count):
            pose = q[frame].copy()
            pose[indices] = x[frame]
            fk(pose)
            g, local, _ = anchors["right" if frame < 5 else "left"]
            point = d.geom_xmat[g].reshape(3, 3) @ local + d.geom_xpos[g]
            mujoco.mj_jacSubtreeCom(m, d, jc, root)
            mujoco.mj_jac(m, d, jp, jr, point, int(m.geom_bodyid[g]))
            block = np.zeros((blocksize, width))
            block[:3] = jc[:, tangent] * 1000
            block[3:6] = jp[:, tangent] * 1000
            for k, (geom, vertices) in enumerate(meshes):
                world = vertices @ d.geom_xmat[geom].reshape(3, 3).T + d.geom_xpos[geom]
                point = world[world[:, 2].argmin()]
                if point[2] < .006:
                    mujoco.mj_jac(m, d, jp, jr, point, int(m.geom_bodyid[geom]))
                    block[6 + k] = jp[2, tangent] * 1000
            block[-width:] = np.eye(width) * .1
            jac[frame * blocksize:(frame + 1) * blocksize, frame * width:(frame + 1) * width] = block
        for i in range(count - 2):
            r = count * blocksize + i * width
            for delta, weight in ((0, 4), (1, -8), (2, 4)):
                jac[r:r + width, (i + delta) * width:(i + delta + 1) * width] = np.eye(width) * weight
        r = count * blocksize + (count - 2) * width
        nj = width - 3
        jac[r:r + nj, (count - 1) * width + 3:] = np.eye(nj) * 50
        jac[r + nj:, (count - 2) * width + 3:(count - 1) * width] = np.eye(nj) * -10
        jac[r + nj:, (count - 1) * width + 3:] = np.eye(nj) * 20
        return jac.tocsr()
    lower = np.tile(np.r_[[-5, -5, .1], m.jnt_range[1:, 0] + 1e-5], count)
    upper = np.tile(np.r_[[5, 5, 2], m.jnt_range[1:, 1] - 1e-5], count)
    if args.lock_low_torque_joints:
        for name in ("waist_pitch_joint", "waist_roll_joint", "head_yaw_joint", "head_pitch_joint"):
            col = int(np.where(indices == m.jnt_qposadr[m.joint(name).id])[0][0])
            lower[col::width], upper[col::width] = -1e-7, 1e-7
    fit = least_squares(residual, np.clip(reference.ravel(), lower, upper),
                        jac=jacobian, bounds=(lower, upper),
                        max_nfev=60, ftol=1e-6, xtol=1e-7, gtol=1e-6,
                        verbose=2)
    q[:count, indices] = fit.x.reshape(count, width)
    args.out.mkdir(parents=True, exist_ok=False)
    np.savetxt(args.out / "webster.csv", np.c_[q[:, :3], q[:, [4, 5, 6, 3]], q[:, 7:]], delimiter=",", fmt="%.10f")
    convert_csv_to_npz(args.out / "webster.csv", args.out / "webster.npz", fps, fps)
    values = residual(fit.x)[:count * blocksize].reshape(count, blocksize)
    result = {"max_com_error_m": float(np.abs(values[:, :3]).max() / 1000),
              "max_contact_error_m": float(np.abs(values[:, 3:6]).max() / 1000),
              "cost": float(fit.cost), "nfev": fit.nfev,
              "source": str(args.input.resolve()), "projection": str(args.projection.resolve()),
              "limitation": "Contact/COM/smoothness fit; force, torque, flight momentum and collision checks must be rerun."}
    (args.out / "stance-fit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
