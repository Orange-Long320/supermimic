import pytest
import torch

from supermimic.envs.mdp.rewards import _sq_quat_err
from mjlab.utils.lab_api.math import quat_box_minus


def test_identical_and_antipodal_rotations_have_zero_finite_gradient():
    for sign in (1.,-1.):
        q=torch.tensor([[1.,0.,0.,0.],[0.,1.,0.,0.]],dtype=torch.float64,requires_grad=True)
        other=q.detach()*sign
        error=_sq_quat_err(q,other)
        error.sum().backward()
        assert torch.equal(error,torch.zeros_like(error))
        assert torch.equal(q.grad,torch.zeros_like(q))


def test_nonzero_angles_match_existing_geodesic_and_gradcheck_at_identity():
    generator=torch.Generator().manual_seed(8)
    q=torch.randn(64,4,generator=generator,dtype=torch.float64)
    r=torch.randn(64,4,generator=generator,dtype=torch.float64)
    q=q/q.norm(dim=-1,keepdim=True);r=r/r.norm(dim=-1,keepdim=True)
    torch.testing.assert_close(_sq_quat_err(q,r),quat_box_minus(q,r).square().sum(-1),rtol=1e-10,atol=1e-10)
    q=torch.tensor([[1.,0.,0.,0.]],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda x:_sq_quat_err(x,q.detach()),(q,))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='needs one explicitly assigned/shared GPU')
def test_compiled_rotation_error_has_finite_gradient_at_zero():
    compiled=torch.compile(_sq_quat_err,dynamic=False)
    q=torch.tensor([[1.,0.,0.,0.],[0.,1.,0.,0.]],device='cuda',requires_grad=True)
    r=q.detach().clone()
    compiled(q,r).sum().backward()
    assert torch.isfinite(q.grad).all() and torch.equal(q.grad,torch.zeros_like(q))
