"""C2 dense resampling and original native-model inverse diagnostics.

No retiming, collision restoration or policy export. Source knots and original
physics stay fixed. Near-surface force-cone feasibility is separate from the
engine's compliant-contact inverse requirement; neither is controller success.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['reference','model','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('New output required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only diagnostics required')
    import mujoco
    import numpy as np
    from scipy.interpolate import CubicSpline
    from scipy.spatial.transform import Rotation,RotationSpline
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra',check_assets=True)
    from supermimic.robots.dimensions import PROFILE
    from supermimic.data.physics_motion import differentiate_qpos
    m=mujoco.MjModel.from_binary_path(str(a.model));d=mujoco.MjData(m)
    if (m.nq,m.nv,m.nu,m.neq)!=(38,37,31,0) or abs(m.opt.timestep-.005)>1e-12:raise ValueError('Original actuated200Hz native model required')
    names=[m.joint(i).name.rsplit('/',1)[-1] for i in range(1,m.njnt)]
    if names!=PROFILE['joint_names']:raise ValueError('Wrong joint order')
    effort=np.array([PROFILE['effort_limit'][n] for n in names]);speed=np.array([PROFILE['velocity_limit'][n] for n in names])
    np.testing.assert_allclose(m.actuator_forcerange,np.c_[-effort,effort],rtol=0,atol=1e-9)
    np.testing.assert_array_equal(m.jnt_dofadr[m.actuator_trnid[:,0]],np.arange(6,37))
    np.testing.assert_allclose(m.opt.gravity,[0,0,-9.81],rtol=0,atol=1e-12)
    with np.load(a.reference) as z:q=z['qpos'].copy();fps=float(np.asarray(z['fps']).reshape(-1)[0])
    if fps not in [50.,200.] or q.shape[1]!=38:raise ValueError('Explicit50Hz or200Hz source required')
    factor=int(200/fps);times=np.arange(len(q))/fps;clock=np.arange((len(q)-1)*factor+1)/200.
    columns=np.r_[np.arange(3),np.arange(7,38)];dense=np.empty((len(clock),38))
    dense[:,columns]=CubicSpline(times,q[:,columns],axis=0)(clock)
    rotations=RotationSpline(times,Rotation.from_quat(q[:,[4,5,6,3]]))
    dense[:,3:7]=rotations(clock).as_quat()[:,[3,0,1,2]]
    error=float(abs(dense[::factor][:,columns]-q[:,columns]).max())
    rotation_error=float((Rotation.from_quat(q[:,[4,5,6,3]]).inv()*rotations(times)).magnitude().max())
    if max(error,rotation_error)>1e-10:raise ValueError('Interpolation changed source knots')
    v=differentiate_qpos(m,dense,200.);acc=np.zeros_like(v)
    forward=np.zeros(37);backward=np.zeros(37)
    for i in range(1,len(dense)-1):
        mujoco.mj_differentiatePos(m,forward,.005,dense[i],dense[i+1]);mujoco.mj_differentiatePos(m,backward,-.005,dense[i],dense[i-1])
        acc[i]=(forward-backward)*200.
    root=int(m.jnt_bodyid[0]);weight=float(m.body_mass.sum()*9.81);scales=np.r_[np.full(3,weight),np.full(3,weight*.7)]
    robot=[g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]]==root and (m.geom_contype[g] or m.geom_conaffinity[g])]
    terrain=[g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]]!=root and (m.geom_contype[g] or m.geom_conaffinity[g])]
    witness=np.zeros(6);peak=0.;self_peak=0.;rows=[];inverse=[]
    for i,pose in enumerate(dense):
        d.qpos[:]=pose;d.qvel[:]=v[i];mujoco.mj_forward(m,d)
        penetration=max(0.,max(-float(mujoco.mj_geomDistance(m,d,g,o,.02,witness)) for g in robot for o in terrain))
        peak=max(peak,penetration)
        for c in d.contact:
            if c.geom1 in robot and c.geom2 in robot:self_peak=max(self_peak,-float(c.dist))
        if 2<=i<len(dense)-2:
            # Reapply the requested acceleration after forward; retain the
            # original contacts/limits when computing the engine inverse.
            d.qacc[:]=acc[i];mujoco.mj_inverse(m,d);required=d.qfrc_inverse.copy();inverse.append(required)
            rows.append(dict(frame200=i,time_s=i/200.,penetration_m=penetration,
                native_root_residual_normalized=float(abs(required[:6]/scales).max()),
                native_required_motor_ratio=float((abs(required[6:])/effort).max())))
        if i%200==0:print(f'checked{i}/{len(dense)}',flush=True)
    centre=m.jnt_range[1:].mean(1);radius=np.diff(m.jnt_range[1:],axis=1).ravel()*.45
    soft=float(np.maximum(abs(dense[:,7:]-centre)-radius,0).max());velocity=float((abs(v[:,6:])/speed).max())
    geometry_passed=bool(peak<=.001 and self_peak<=.001 and soft<=1e-9 and velocity<=1.)
    a.out.mkdir(parents=True);sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    np.savez_compressed(a.out/'dense-qpos.npz',qpos=dense,qvel=v,fps=np.array([200.]))
    np.savez_compressed(a.out/'native-inverse.npz',required_generalized_force=np.asarray(inverse),frames=np.arange(2,len(dense)-2),fps=np.array([200.]))
    report=dict(source_sha256=sha(a.reference),model_sha256=sha(a.model),source_fps=fps,output_fps=200.,frames=len(dense),
        time_scale=1.,source_knot_position_joint_error=error,source_knot_orientation_error_rad=rotation_error,
        interpolation='CubicSpline translations/hinges, RotationSpline orientations; continuous angular rate/acceleration; no collision restoration',
        scipy_rotation_spline_docs='https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.RotationSpline.html',
        environment_penetration_m=peak,self_penetration_m=self_peak,soft_joint_violation_rad=soft,maximum_joint_speed_ratio=velocity,
        dense_geometry_speed_bounds_passed=geometry_passed,
        native_inverse_necessary_passed=bool(geometry_passed and all(r['native_root_residual_normalized']<=1e-5 and r['native_required_motor_ratio']<=1 for r in rows)),
        native_root_residual_max_normalized=max(r['native_root_residual_normalized'] for r in rows),
        native_motor_ratio_max=max(r['native_required_motor_ratio'] for r in rows),rows=rows,
        approved_for_training=False,actual_controller_execution_verified=False,scope=__doc__)
    (a.out/'report.json').write_text(json.dumps(report,indent=2));(a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
