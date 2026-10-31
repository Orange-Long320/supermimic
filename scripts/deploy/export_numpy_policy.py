"""Export the immutable selected TorchScript's FP32 tensors; never retrains."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from supermimic.deploy.numpy_policy import SOURCE_POLICY_SHA, parameter_shapes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a=p.parse_args()
    if a.out.exists() or a.out.with_suffix('.json').exists(): raise ValueError('New export required')
    if hashlib.sha256(a.policy.read_bytes()).hexdigest() != SOURCE_POLICY_SHA: raise ValueError('Wrong policy source')
    model=torch.jit.load(str(a.policy),map_location='cpu').eval()
    state=model.state_dict(); shapes=parameter_shapes()
    if set(state)!=set(shapes): raise ValueError('Unexpected parameter set')
    values={}
    for key,shape in shapes.items():
        value=state[key].detach().numpy()
        if value.dtype!=np.float32 or value.shape!=shape or not np.isfinite(value).all(): raise ValueError('Invalid '+key)
        values[key]=value
    # Source SHA pins the actual graph: scale -> SiLU MLP -> tanh encoder;
    # decoder concatenates [latent, scaled observation] then a SiLU MLP.
    values['source_policy_sha256']=np.asarray(SOURCE_POLICY_SHA)
    with a.out.open('xb') as f: np.savez_compressed(f,**values)
    report={'source_policy_sha256':SOURCE_POLICY_SHA,'archive_sha256':hashlib.sha256(a.out.read_bytes()).hexdigest(),
            'parameters':{k:list(v) for k,v in shapes.items()},'dtype':'float32','quantized':False,
            'architecture':'selected 11500 scale/SiLU/tanh/latent-first decoder; tensors unchanged',
            'numerical_equivalence_validated':False,'physical_acceptance_validated':False}
    a.out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
