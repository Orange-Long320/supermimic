"""Compare IMU-to-pelvis conversion against MuJoCo Jacobian truth on server."""
import argparse
import json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from supermimic.deploy.mujoco_imu_mount import MujocoImuMount
from supermimic.deploy.pelvis_state import UrdfPelvisMount, pelvis_angular_velocity, pelvis_from_imu, pelvis_velocity_from_imu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--mount-urdf', required=True)
    a = ap.parse_args()
    names = ('waist_yaw_joint', 'waist_pitch_joint', 'waist_roll_joint')
    mount = MujocoImuMount(a.model, 'imu_1', names)
    urdf_mount = UrdfPelvisMount(a.mount_urdf, names, [-.0248787, .00198528, .105938], np.eye(3))
    m, d = mount.model, mujoco.MjData(mount.model)
    body, site = mount.pelvis, mount.site
    jp, jr, bp, br = (np.zeros((3, m.nv)) for _ in range(4))
    rng = np.random.default_rng(1102)
    report = {}
    for name in ('static', 'pelvis_rotation', 'waist_motion', 'combined'):
        errors = []
        for _ in range(100):
            d.qpos[:] = m.qpos0
            d.qpos[:3] = rng.uniform(-1, 1, 3)
            d.qpos[3:7] = Rotation.random(random_state=rng).as_quat()[[3, 0, 1, 2]]
            q = rng.uniform(-.2, .2, 3)
            d.qpos[mount.qadr] = q
            d.qvel[:] = 0.
            if name in ('pelvis_rotation', 'combined'):
                d.qvel[:6] = rng.uniform(-.5, .5, 6)
            if name in ('waist_motion', 'combined'):
                d.qvel[mount.dadr] = rng.uniform(-.5, .5, 3)
            mujoco.mj_fwdPosition(m, d)
            mujoco.mj_jacSite(m, d, jp, jr, site)
            mujoco.mj_jacBody(m, d, bp, br, body)
            rwi, rwp = d.site_xmat[site].reshape(3, 3), d.xmat[body].reshape(3, 3)
            ppi, rpi, wm, vm = mount.mount_with_velocity(q, d.qvel[mount.dadr])
            up, ur, uw, uv = urdf_mount.mount_with_velocity(q, d.qvel[mount.dadr])
            if max(np.linalg.norm(x-y) for x, y in ((ppi, up), (rpi, ur), (wm, uw), (vm, uv))) > 1e-10:
                raise ValueError('URDF mount disagrees with simulator')
            p, r = pelvis_from_imu(d.site_xpos[site], rwi, ppi, rpi)
            omega = pelvis_angular_velocity(rwi.T @ (jr @ d.qvel), rpi, wm)
            v = pelvis_velocity_from_imu(jp @ d.qvel, r, ppi, omega, vm)
            errors.append([np.linalg.norm(p-d.xpos[body]), np.linalg.norm(r-rwp),
                           np.linalg.norm(v-bp@d.qvel), np.linalg.norm(omega-rwp.T@(br@d.qvel))])
        report[name] = dict(zip(('position_m', 'rotation_matrix', 'velocity_m_s', 'angular_rad_s'),
                                np.max(errors, axis=0).tolist()))
    with open(a.out, 'x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
