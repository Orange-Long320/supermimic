"""Synthetic geometry and causality checks, independent of robot data."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from supermimic.deploy.pelvis_state import pelvis_from_imu,pelvis_angular_velocity,latest_causal,gravity_level_rotation


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


if __name__=='__main__':unittest.main()
