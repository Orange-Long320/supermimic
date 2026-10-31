import numpy as np
import pytest

from supermimic.control.idle_velocity_feedback import RootVelocityEstimate


def test_velocity_uses_sample_times_and_is_translation_invariant():
    a, b = RootVelocityEstimate(), RootVelocityEstimate()
    velocity = np.array([.12, -.03, .02])
    for t in [0., .019, .041, .063, .081, .102, .123, .143]:
        va = a.update(1000+t, np.array([1., 2., .65])+velocity*t)
        vb = b.update(1000+t, np.array([5., -3., .65])+velocity*t)
    np.testing.assert_allclose(va, [.12, -.03, 0.], atol=1e-12)
    np.testing.assert_allclose(va, vb, atol=1e-12)


def test_static_root_has_no_velocity_correction_and_duplicate_sample_is_ok():
    estimator = RootVelocityEstimate()
    for t in [1., 1.02, 1.04, 1.06]:
        np.testing.assert_allclose(estimator.update(t, [1., 2., .65]), 0., atol=1e-14)
    np.testing.assert_allclose(estimator.update(1.06, [1., 2., .65]), 0., atol=1e-14)
    with pytest.raises(ValueError, match='Nonmonotonic'):
        estimator.update(1.05, [1., 2., .65])
    with pytest.raises(ValueError, match='Invalid odometry'):
        estimator.update(1.08, [np.nan, 2., .65])


def test_original_observation_cannot_distinguish_rigid_translation_speed():
    import torch
    from supermimic.robots.dimensions import NJ, NQ, STATE_DIM, XI_DIM
    from supermimic.world_model.heads import analytic_observation
    still = torch.zeros(1, STATE_DIM)
    still[:, 3] = 1.
    moving = still.clone()
    moving[:, NQ] = .05
    xi = torch.zeros(1, XI_DIM)
    previous = torch.zeros(1, NJ)
    default = torch.zeros(NJ)
    torch.testing.assert_close(analytic_observation(still, previous, xi, default),
                               analytic_observation(moving, previous, xi, default), rtol=0, atol=0)
    # Timestamped global translation history supplies the missing information;
    # both histories finish at the same current root position and orientation.
    a, b = RootVelocityEstimate(), RootVelocityEstimate()
    for t in np.arange(6)*.02:
        va = a.update(t, [0., 0., .65])
        vb = b.update(t, [.05*(t-.1), 0., .65])
    np.testing.assert_allclose(va, [0,0,0], atol=1e-14)
    np.testing.assert_allclose(vb, [.05,0,0], atol=1e-14)
