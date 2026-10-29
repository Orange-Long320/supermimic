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
    p.add_argument('--native-model',type=Path,required=True,help='Verified matching native terrain model; interpolation must pass geometry again')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference-policy',type=Path,help='Explicit training-reference tolerances; omitted retains legacy strict geometry')
    a=p.parse_args()
    if a.out.exists():p.error('New output required')
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra',check_assets=True)
    import numpy as np
    import mujoco
    from scipy.interpolate import PchipInterpolator
    from scipy.spatial.transform import Rotation,Slerp
    from supermimic.robots.assets import get_spec
    from supermimic.robots.dimensions import PROFILE
    from supermimic.data.physics_motion import differentiate_qpos,export_qpos_motion
    from supermimic.data.reference_acceptance import load_reference_policy,audit_reference_geometry
    policy=load_reference_policy(a.reference_policy)
    with np.load(a.candidate/'x2-reference.npz',allow_pickle=False) as z:
        q=z['qpos'].copy();fps=float(np.asarray(z['fps']).reshape(-1)[0])
    if q.ndim!=2 or q.shape[1]!=38 or len(q)<5 or not np.isfinite(q).all() or fps<=0:raise ValueError('Invalid X2 candidate')
    duration=(len(q)-1)/fps;nsteps=int(np.ceil(duration*50))
    source_time=np.arange(len(q))/fps;query=np.linspace(0.,duration,nsteps*4+1)
    dense=np.zeros((len(query),38));cols=np.r_[np.arange(3),np.arange(7,38)]
    dense[:,cols]=PchipInterpolator(source_time,q[:,cols],axis=0)(query)
    dense[:,3:7]=Slerp(source_time,Rotation.from_quat(q[:,[4,5,6,3]]))(query).as_quat()[:,[3,0,1,2]]
    result=dense[::4].copy()
    np.testing.assert_allclose(result[[0,-1],:3],q[[0,-1],:3],atol=1e-12)
    m=get_spec().compile();mid=m.jnt_range[1:].mean(1);half=np.diff(m.jnt_range[1:],axis=1).reshape(-1)*.45
    soft=float(np.maximum(np.maximum(mid-half-result[:,7:],result[:,7:]-(mid+half)),0).max())
    if soft>1e-8:raise ValueError('Interpolation violates the existing 0.9 reset bounds')
    a.out.mkdir(parents=True)
    native=mujoco.MjModel.from_binary_path(str(a.native_model));d=mujoco.MjData(native)
    if (native.nq,native.nv,native.nu)!=(38,37,31):raise ValueError('Unexpected native robot model')
    if [native.joint(i).name.removeprefix('robot/') for i in range(1,native.njnt)]!=[m.joint(i).name for i in range(1,m.njnt)]:raise ValueError('Native joint order differs')
    check=audit_reference_geometry(native,dense,policy)
    check.update(fps=200.,model_sha256=hashlib.sha256(a.native_model.read_bytes()).hexdigest())
    (a.out/'resampling-geometry.json').write_text(json.dumps(check,indent=2)+'\n')
    if not check['passed']:raise RuntimeError('Resampled reference fails selected contact/body geometry policy; no policy input exported')
    dense_velocity=differentiate_qpos(m,dense,200.)
    velocity=dense_velocity[::4]
    names=list(PROFILE['joint_names'])
    if [m.joint(i).name for i in range(1,m.njnt)]!=names:raise ValueError('Velocity limits do not match native joint order')
    limits=np.array([PROFILE['velocity_limit'][name] for name in names])
    if not np.isfinite(limits).all() or not (limits>0).all():raise ValueError('Invalid original velocity limits')
    if not np.isfinite(dense_velocity).all():raise ValueError('Nonfinite differentiated reference')
    ratios=abs(dense_velocity[:,6:])/limits
    t,j=np.unravel_index(ratios.argmax(),ratios.shape)
    speed_check={'passed':bool(ratios.max()<=1.),'dense_samples':len(dense),'dense_fps':200.,
        'maximum_dense_joint_velocity_ratio':float(ratios[t,j]),
        'maximum_export_joint_velocity_ratio':float((abs(velocity[:,6:])/limits).max()),
        'peak_joint':names[j],'peak_frame':int(t),'peak_time_s':float(t/200.),
        'peak_speed_rad_s':float(dense_velocity[t,6+j]),'original_limit_rad_s':float(limits[j]),
        'scope':'Original actuated joint speed limits checked at dense200Hz and exported50Hz; no dynamics certificate'}
    (a.out/'resampling-velocity.json').write_text(json.dumps(speed_check,indent=2)+'\n')
    if not speed_check['passed']:raise RuntimeError('Resampled reference fails unchanged joint speed limits; no policy input exported')
    export_qpos_motion(m,result,a.out/'motion-fk.npz',50.,qvel=velocity,source=a.candidate/'x2-reference.npz')
    np.savez_compressed(a.out/'reference-qpos.npz',qpos=result,qvel=velocity,fps=np.array([50.]))
    (a.out/'scene.json').write_bytes((a.candidate/'result.json').read_bytes())
    info={'source_reference_sha256':hashlib.sha256((a.candidate/'x2-reference.npz').read_bytes()).hexdigest(),
        'source_frames':len(q),'source_fps':fps,'source_duration_s':duration,'frames':len(result),'fps':50.,
        'duration_s':nsteps/50,'uniform_time_scale':(nsteps/50)/duration,
        'interpolation':'PCHIP translations/joints and quaternion SLERP; generalized velocities differentiated at 200 Hz and sampled at 50 Hz',
        'resampling_geometry':check,
        'resampling_velocity':speed_check,
        'reset_soft_violation_rad':soft,'reference_input_accepted':True,
        'force_diagnostics_are_training_gate':False,'training_approved':False,'purpose':__doc__}
    (a.out/'manifest.json').write_text(json.dumps(info,indent=2)+'\n');(a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(info,indent=2))


if __name__=='__main__':main()
