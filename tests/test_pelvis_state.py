"""Synthetic geometry and causality checks, independent of robot data."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from supermimic.deploy.pelvis_state import UrdfPelvisMount,pelvis_from_imu,pelvis_angular_velocity,latest_causal,gravity_level_rotation


class PelvisStateTest(unittest.TestCase):
    def test_initial_imu_frame_is_not_assumed_z_up(self):
        for up in ([8.17,.18,5.43],[0,0,-1]):
            r=gravity_level_rotation(up)
            np.testing.assert_allclose(r@(np.asarray(up)/np.linalg.norm(up)),[0,0,1],atol=1e-12)
            np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(r),1.)

    def test_moving_mount_round_trip(self):
        rwp=Rotation.from_euler('xyz',[.2,-.3,1.2]).as_matrix(); pwp=np.array([1.,2.,.7])
        for pitch in (-.4,0,.6):
            rpi=Rotation.from_euler('xyz',[.1,pitch,.7]).as_matrix(); ppi=np.array([.1,0,.5])
            p,r=pelvis_from_imu(pwp+rwp@ppi,rwp@rpi,ppi,rpi)
            np.testing.assert_allclose(p,pwp,atol=1e-12);np.testing.assert_allclose(r,rwp,atol=1e-12)

    def test_waist_motion_is_not_pelvis_rotation(self):
        rpi=Rotation.from_euler('y',.5).as_matrix(); waist=np.array([0.,.4,0.])
        np.testing.assert_allclose(pelvis_angular_velocity(rpi.T@waist,rpi,waist),0,atol=1e-12)

    def test_late_arrival_and_stale_data_rejected(self):
        rows=[{'received_ns':100_000_000,'header_ns':90_000_000},
              {'received_ns':300_000_000,'header_ns':180_000_000}]
        self.assertEqual(latest_causal(rows,190_000_000,max_age_s=.15),rows[0])
        with self.assertRaises(ValueError): latest_causal(rows,290_000_000,max_age_s=.15)
        with self.assertRaises(ValueError): latest_causal(rows,50_000_000,max_age_s=.15)

    def mount_model(self, directory):
        path=Path(directory)/'mount.urdf'
        path.write_text('''<robot>
          <joint name="base" type="fixed"><parent link="base_link"/><child link="pelvis"/><origin xyz="10 20 30"/></joint>
          <joint name="yaw" type="revolute"><parent link="pelvis"/><child link="waist"/><origin xyz="0 0 0.1"/><axis xyz="0 0 1"/></joint>
          <joint name="pitch" type="revolute"><parent link="waist"/><child link="torso_link"/><origin xyz="0.1 0 0.2" rpy="0.1 0.2 0.3"/><axis xyz="0 1 0"/></joint>
        </robot>''')
        return UrdfPelvisMount(path,('yaw','pitch'),[.2,0,.1],Rotation.from_euler('x',.4).as_matrix())

    def test_device_mount_uses_pelvis_frame_and_sensor_transform(self):
        with tempfile.TemporaryDirectory() as directory:
            mount=self.mount_model(directory)
            p,r,w=mount.mount([0,0])
            expected=Rotation.from_euler('xyz',[.1,.2,.3]).as_matrix()
            np.testing.assert_allclose(p,np.array([.1,0,.3])+expected@np.array([.2,0,.1]),atol=1e-12)
            np.testing.assert_allclose(r,expected@Rotation.from_euler('x',.4).as_matrix(),atol=1e-12)
            np.testing.assert_allclose(w,0)
            rwp=Rotation.from_euler('xyz',[.3,-.2,.8]).as_matrix();pwp=np.array([1.,2.,.7])
            recovered_p,recovered_r=pelvis_from_imu(pwp+rwp@p,rwp@r,p,r)
            np.testing.assert_allclose(recovered_p,pwp,atol=1e-12)
            np.testing.assert_allclose(recovered_r,rwp,atol=1e-12)

    def test_device_mount_angular_velocity_matches_finite_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            mount=self.mount_model(directory)
            q=np.array([.4,-.2]);qd=np.array([.2,.1]);epsilon=1e-6
            _,r,w=mount.mount(q,qd)
            _,plus,_=mount.mount(q+epsilon*qd)
            _,minus,_=mount.mount(q-epsilon*qd)
            skew=((plus-minus)/(2*epsilon))@r.T
            np.testing.assert_allclose(w,[skew[2,1],skew[0,2],skew[1,0]],atol=1e-9)

    def test_analytic_mount_linear_velocity_matches_finite_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            mount = self.mount_model(directory)
            rng = np.random.default_rng(103)
            for _ in range(50):
                q, qd = rng.uniform(-1., 1., (2, 2)); epsilon = 1e-6
                p, r, omega, velocity = mount.mount_with_velocity(q, qd)
                expected_p, expected_r, expected_omega = mount.mount(q, qd)
                plus = mount.mount(q+epsilon*qd)[0]
                minus = mount.mount(q-epsilon*qd)[0]
                np.testing.assert_allclose(p, expected_p, atol=1e-12)
                np.testing.assert_allclose(r, expected_r, atol=1e-12)
                np.testing.assert_allclose(omega, expected_omega, atol=1e-12)
                np.testing.assert_allclose(velocity, (plus-minus)/(2*epsilon), atol=1e-9)

    def test_device_mount_rejects_bad_measurements_and_rotation(self):
        with tempfile.TemporaryDirectory() as directory:
            mount=self.mount_model(directory)
            for q in ([float('nan'),0],[0],[[0,0]]):
                with self.assertRaises(ValueError):mount.mount(q)
            with self.assertRaises(ValueError):
                UrdfPelvisMount(Path(directory)/'mount.urdf',('yaw','pitch'),[0,0,0],np.zeros((3,3)))


if __name__=='__main__':unittest.main()
