"""Terrain visibility and function-preserving transfer for the Omni pilot."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from supermimic.envs.box_scene import TerrainFeatureEncoder
from supermimic.robots.dimensions import FEAT_DIM
from supermimic.world_model.dynamics import DynamicsMLP


def test_terrain_heading_and_translation_invariance(tmp_path):
    boxes=[{'center':[x,0.,.3],'half_size':[.4,.5,.3],'rotation_world':np.eye(3).tolist()} for x in [0.,2.]]
    original=tmp_path/'original.json';original.write_text(json.dumps({'boxes':boxes}))
    angle=.7;c,s=np.cos(angle),np.sin(angle);r=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    shift=np.array([4.,-3.,0.]);rotated=tmp_path/'rotated.json'
    rotated.write_text(json.dumps({'boxes':[{**b,'center':(r@b['center']+shift).tolist(),'rotation_world':r.tolist()} for b in boxes]}))
    p=torch.tensor([[.2,.1,1.],[1.8,.2,1.]],dtype=torch.float64,requires_grad=True)
    q=torch.tensor([[1.,0.,0.,0.]]*2,dtype=torch.float64)
    a=TerrainFeatureEncoder(original,'cpu',torch.float64)(p,q)
    transformed=p@torch.tensor(r).T+torch.tensor(shift)
    yaw=torch.tensor([[np.cos(angle/2),0.,0.,np.sin(angle/2)]]*2,dtype=torch.float64)
    b=TerrainFeatureEncoder(rotated,'cpu',torch.float64)(transformed,yaw)
    torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)
    assert not torch.allclose(a[0],a[1])
    a.square().sum().backward();assert torch.isfinite(p.grad).all()


def test_terrain_warmstart_preserves_dynamics_predictions():
    source=Path(__file__).resolve().parents[1]/'scripts/experiments/omni_pilot.py'
    spec=importlib.util.spec_from_file_location('terrain_pilot_test',source)
    entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
    torch.manual_seed(7)
    old=DynamicsMLP(feat_dim=FEAT_DIM,hidden=(16,16));new=DynamicsMLP(feat_dim=FEAT_DIM+18,hidden=(16,16))
    torch.nn.init.normal_(old.mlp.net[-1].weight)
    entry.warmstart_terrain_module(new,old.state_dict(),'mlp.net.0.weight',FEAT_DIM)
    features=torch.randn(5,FEAT_DIM);terrain=torch.randn(5,18)
    # Derive xi size from this robot's declared interface rather than G1 counts.
    from supermimic.robots.dimensions import NJ,XI_DIM
    action=torch.randn(5,NJ);xi=torch.randn(5,XI_DIM)
    torch.testing.assert_close(old(features,action,xi),new(torch.cat([features,terrain],-1),action,xi),rtol=1e-5,atol=1e-6)
