"""Keep GPU diagnostics equivalent at mechanical-limit decision boundaries."""
import numpy as np
import pytest
import torch
from supermimic.data.design_limits import joint_limit_statistics


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_limits_match_independent_scalar_reference(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    rng=np.random.default_rng(42);n,joints=7,31
    lo=rng.uniform(-1,-.2,joints);hi=rng.uniform(.2,1,joints)
    softlo=lo+.1;softhi=hi-.1;speed=rng.uniform(3,30,joints);effort=rng.uniform(20,120,joints)
    q=np.zeros((n,joints+7),dtype=np.float32);q[:,3]=1;q[:,7:]=rng.uniform(lo,hi,(n,joints))
    q[0,7]=lo[0]-1e-4;q[1,8]=hi[1]+1e-4;q[2,9]=lo[2]-1e-7
    v=rng.uniform(-40,40,(n,joints+6)).astype(np.float32)
    force=rng.uniform(-150,150,(n,joints)).astype(np.float32)
    expected=[]
    for k in range(n):
        expected.append([max(max(softlo[i]-float(q[k,7+i]),float(q[k,7+i])-softhi[i],0) for i in range(joints)),
                         max(max(lo[i]-float(q[k,7+i]),float(q[k,7+i])-hi[i],0) for i in range(joints)),
                         max(abs(float(v[k,6+i]))/speed[i] for i in range(joints)),
                         max(abs(float(force[k,i]))/effort[i] for i in range(joints))])
    tensor=lambda x:torch.as_tensor(x,device=device)
    actual=joint_limit_statistics(tensor(q),tensor(v),tensor(force),*[tensor(x) for x in [softlo,softhi,lo,hi,speed,effort]])
    assert actual.dtype==torch.float64
    np.testing.assert_allclose(actual.cpu().numpy(),expected,rtol=1e-12,atol=1e-12)


def test_body_contact_cannot_buy_a_lower_core_preserving_score():
    from copy import deepcopy
    from supermimic.data.design_limits import candidate_priority
    safe={'cost':100.,'physical_candidate_passed':False,'takeoff_candidate_passed':False,
          'maxima':{'minimum_nonfoot_height':.02,'speed':.9,'effort':1.}}
    contact=deepcopy(safe);contact['cost']=1.;contact['maxima']['minimum_nonfoot_height']=-.01
    assert candidate_priority(safe,preserve_core=True)<candidate_priority(contact,preserve_core=True)
    assert candidate_priority(contact)<candidate_priority(safe)
    complete=deepcopy(safe);complete['cost']=200.;complete['physical_candidate_passed']=True
    assert candidate_priority(complete,preserve_core=True)<candidate_priority(safe,preserve_core=True)
