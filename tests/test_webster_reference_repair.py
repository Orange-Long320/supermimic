"""Guard against frame-alternating wrist jitter hidden by central differences."""
import importlib.util
import os
from pathlib import Path

import numpy as np


def load_repair():
    path=Path(os.environ.get('SM_REFERENCE_REPAIR_SCRIPT',
        Path(__file__).resolve().parents[1]/'scripts/data/repair_webster_standing_reference.py'))
    spec=importlib.util.spec_from_file_location('webster_reference_repair',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_smoothing_preserves_linear_motion_other_joints_and_boundary_poses():
    module=load_repair();q=np.zeros((61,9));q[:,3]=1.
    q[:,7]=np.linspace(-.3,.5,len(q));q[:,8]=np.sin(np.arange(len(q)))
    result=module.smooth_wrists(q,['left_wrist_yaw_joint','left_knee_joint'])
    np.testing.assert_allclose(result,q,atol=1e-13)
    np.testing.assert_array_equal(result[[0,-1]],q[[0,-1]])


def test_frame_alternating_noise_is_removed_even_when_central_speed_is_zero():
    module=load_repair();q=np.zeros((61,9));q[:,3]=1.
    q[:,7]=.6*(-1.)**np.arange(len(q));q[:,8]=np.arange(len(q))
    assert abs(np.gradient(q[:,7])[1:-1]).max()==0
    result=module.smooth_wrists(q,['left_wrist_yaw_joint','left_knee_joint'])
    assert np.max(abs(np.diff(result[10:-10,7])))<.01
    assert np.sqrt(np.mean(result[10:-10,7]**2))<.01
    np.testing.assert_array_equal(result[:,8],q[:,8])
    np.testing.assert_array_equal(result[[0,-1]],q[[0,-1]])
