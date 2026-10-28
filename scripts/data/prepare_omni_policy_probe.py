"""Export a 50 Hz diagnostic input for the existing frozen X2 controller.

This does not approve the source for training. The original candidate and its
terrain remain unchanged; all timing changes and hashes are recorded.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('New output required')
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra',check_assets=True)
    import numpy as np
    from scipy.interpolate import PchipInterpolator
    from scipy.spatial.transform import Rotation,Slerp
    from supermimic.robots.assets import get_spec
    from supermimic.data.physics_motion import differentiate_qpos,export_qpos_motion
    with np.load(a.candidate/'x2-reference.npz',allow_pickle=False) as z:
        q=z['qpos'].copy();fps=float(np.asarray(z['fps']).reshape(-1)[0])
    if q.ndim!=2 or q.shape[1]!=38 or len(q)<5 or not np.isfinite(q).all() or fps<=0:raise ValueError('Invalid X2 candidate')
    duration=(len(q)-1)/fps;nsteps=int(np.ceil(duration*50))
    source_time=np.arange(len(q))/fps;query=np.linspace(0.,duration,nsteps+1)
    result=np.zeros((len(query),38));cols=np.r_[np.arange(3),np.arange(7,38)]
    result[:,cols]=PchipInterpolator(source_time,q[:,cols],axis=0)(query)
    result[:,3:7]=Slerp(source_time,Rotation.from_quat(q[:,[4,5,6,3]]))(query).as_quat()[:,[3,0,1,2]]
    np.testing.assert_allclose(result[[0,-1],:3],q[[0,-1],:3],atol=1e-12)
    m=get_spec().compile();mid=m.jnt_range[1:].mean(1);half=np.diff(m.jnt_range[1:],axis=1).reshape(-1)*.45
    soft=float(np.maximum(np.maximum(mid-half-result[:,7:],result[:,7:]-(mid+half)),0).max())
    if soft>1e-8:raise ValueError('Interpolation violates the existing 0.9 reset bounds')
    a.out.mkdir(parents=True)
    velocity=differentiate_qpos(m,result,50.)
    export_qpos_motion(m,result,a.out/'motion-fk.npz',50.,qvel=velocity,source=a.candidate/'x2-reference.npz')
    np.savez_compressed(a.out/'reference-qpos.npz',qpos=result,qvel=velocity,fps=np.array([50.]))
    (a.out/'scene.json').write_bytes((a.candidate/'result.json').read_bytes())
    info={'source_reference_sha256':hashlib.sha256((a.candidate/'x2-reference.npz').read_bytes()).hexdigest(),
        'source_frames':len(q),'source_fps':fps,'source_duration_s':duration,'frames':len(result),'fps':50.,
        'duration_s':nsteps/50,'uniform_time_scale':(nsteps/50)/duration,
        'interpolation':'PCHIP translations/joints and quaternion SLERP; supplied generalized velocities match the 50 Hz output clock',
        'reset_soft_violation_rad':soft,'training_approved':False,'purpose':__doc__}
    (a.out/'manifest.json').write_text(json.dumps(info,indent=2)+'\n');(a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(info,indent=2))


if __name__=='__main__':main()
