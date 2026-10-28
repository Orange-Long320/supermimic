"""The sole proxy must stay inside the original convex mesh, not its AABB."""
import importlib.util
import os
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull


def load_fit():
    path=Path(os.environ.get('SM_SOLE_BOX_SCRIPT',Path(__file__).resolve().parents[1]/'scripts/evaluation/fit_native_sole_boxes.py'))
    spec=importlib.util.spec_from_file_location('sole_box_fit',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def corners(box,z,hz):
    cx,cy,hx,hy=box
    return np.array([[cx+x*hx,cy+y*hy,z+w*hz] for x in [-1,1] for y in [-1,1] for w in [-1,1]])


def test_rectangular_sole_retains_full_supported_width():
    points=np.array([[x,y,z] for x in [-.1,.1] for y in [-.05,.05] for z in [0.,.02]])
    hull=ConvexHull(points);box,error=load_fit().inscribed_rectangle(hull.equations,.00425,.004,points)
    np.testing.assert_allclose(box,[0,0,.1,.05],atol=1e-7)
    assert error<1e-8
    assert (corners(box,.00425,.004)@hull.equations[:,:3].T+hull.equations[:,3]).max()<1e-8


def test_tapered_sole_cannot_gain_support_from_empty_aabb_corners():
    points=np.array([[x,y,z] for x,y in [(-.1,-.05),(.1,-.05),(0,.05)] for z in [0.,.02]])
    hull=ConvexHull(points);box,error=load_fit().inscribed_rectangle(hull.equations,.00425,.004,points)
    assert error<1e-8
    assert 4*box[2]*box[3]<.006
    assert (corners(box,.00425,.004)@hull.equations[:,:3].T+hull.equations[:,3]).max()<1e-8
