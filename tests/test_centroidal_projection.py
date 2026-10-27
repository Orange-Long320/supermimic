"""A full forward flip must retain its winding even from a backward seed."""

import numpy as np
from scipy.spatial.transform import Rotation

from supermimic.data.centroidal_projection import project_flight
from supermimic.data.motion_convert import load_g1_model


def test_projection_preserves_full_forward_rotation_from_backward_seed():
    model = load_g1_model()
    poses = np.tile(model.qpos0, (31, 1))
    poses[:, 2] = 1.
    rotvec = np.zeros((31, 3))
    rotvec[:, 1] = np.linspace(0., 6., 31)
    poses[:, 3:7] = Rotation.from_rotvec(rotvec).as_quat()[:, [3, 0, 1, 2]]
    result, _, _ = project_flight(model, poses, 0, 30, 50., np.array([0., -1., 0.]))
    actual = Rotation.from_quat(result[:, [4, 5, 6, 3]])
    increments = (actual[1:] * actual[:-1].inv()).as_rotvec()
    assert increments[:, 1].sum() > 5.5
    endpoint_error = (actual[-1].inv() * Rotation.from_euler("y", 6.)).magnitude()
    assert endpoint_error < .03
