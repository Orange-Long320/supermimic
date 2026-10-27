"""Centroidal shooting helper for server reference preparation."""
import mujoco
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def sole_patches(model):
    """Four sole corners and a virtual toe-edge centre, in geometry coordinates."""
    d = mujoco.MjData(model)
    d.qpos[:] = model.qpos0
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[7:] = 0
    mujoco.mj_kinematics(model, d)
    result = {}
    for g in range(model.ngeom):
        if not model.geom(g).name.startswith("foot_"):
            continue
        mesh = model.geom_dataid[g]
        a, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[a:a + count].astype(float)
        world = vertices @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        candidates = np.flatnonzero(world[:, 2] < world[:, 2].min() + .003)
        corners = []
        for direction in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            corners.append(candidates[np.argmax(world[candidates, :2] @ direction)])
        centre = np.array([world[candidates, 0].max(), world[candidates, 1].mean(), world[:, 2].min()])
        local_centre = d.geom_xmat[g].reshape(3, 3).T @ (centre - d.geom_xpos[g])
        result[g] = (local_centre, vertices[corners])
    return result


def project_flight(model, poses, start, end, fps, seed):
    q = poses.copy()
    d = mujoco.MjData(model)
    root = int(model.jnt_bodyid[0])
    times = np.arange(len(q)) / fps
    joints = CubicSpline(times, q[:, 7:], axis=0)
    am = np.zeros((3, model.nv))
    target = Rotation.from_quat(q[start:end + 1, [4, 5, 6, 3]])
    # Endpoint orientation alone cannot distinguish one full forward flip
    # from a small backward rotation. Preserve the guide's signed cumulative
    # world-Y rotation, including its winding, throughout shooting.
    def forward_progress(rotation):
        increments=(rotation[1:]*rotation[:-1].inv()).as_rotvec()[:,1]
        return np.r_[0.,np.cumsum(increments)]
    target_progress=forward_progress(target)
    com = []
    for row in q[start:end + 1]:
        d.qpos[:] = row
        mujoco.mj_kinematics(model, d)
        mujoco.mj_comPos(model, d)
        com.append(d.subtree_com[root].copy())

    def omega(time, quaternion, momentum):
        d.qpos[:] = q[start]
        d.qpos[3:7] = quaternion / np.linalg.norm(quaternion)
        d.qpos[7:] = joints(time)
        mujoco.mj_kinematics(model, d)
        mujoco.mj_comPos(model, d)
        mujoco.mj_angmomMat(model, d, am, root)
        return np.linalg.solve(am[:, 3:6], momentum - am[:, 6:] @ joints(time, 1))

    def integrate(momentum):
        def derivative(time, quaternion):
            w = omega(time, quaternion, momentum)
            return .5 * np.r_[-quaternion[1:] @ w,
                              quaternion[0] * w + np.cross(quaternion[1:], w)]
        solution = solve_ivp(derivative, (times[start], times[end]), q[start, 3:7],
                             t_eval=times[start:end + 1], rtol=1e-6, atol=1e-8, max_step=.01)
        if not solution.success:
            raise RuntimeError(solution.message)
        r = solution.y.T
        return r / np.linalg.norm(r, axis=1, keepdims=True)

    def error(momentum):
        rotation = Rotation.from_quat(integrate(momentum)[:, [1, 2, 3, 0]])
        delta = (target.inv() * rotation).as_rotvec()
        progress=forward_progress(rotation)-target_progress
        return np.r_[delta[1:-1].ravel() * .1, delta[-1] * 10,
                     progress[1:-1],progress[-1]*10]

    fit = least_squares(error, seed, max_nfev=30, diff_step=1e-4, ftol=1e-6, xtol=1e-6, gtol=1e-6)
    q[start:end + 1, 3:7] = integrate(fit.x)
    for i, target_com in zip(range(start, end + 1), com):
        d.qpos[:] = q[i]
        mujoco.mj_kinematics(model, d)
        mujoco.mj_comPos(model, d)
        q[i, :3] += target_com - d.subtree_com[root]
    return q, fit.x, omega(times[start], q[start, 3:7], fit.x)
