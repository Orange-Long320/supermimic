"""Server-only mechanical regression for the zero-contact wrench LP.

Native X2 ballistic motion must pass; a robot held in the air must fail. Also
compare real corrected/uncorrected unsupported source frames. These are inverse
diagnostics, not controller executions or complete-climb certificates.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['model','original','corrected','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':p.error('New output and explicit CPU required')
    import mujoco
    import numpy as np
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra',check_assets=True)
    from supermimic.robots.dimensions import PROFILE
    from supermimic.data.contact_wrench import solve_contact_effort
    m=mujoco.MjModel.from_binary_path(str(a.model));d=mujoco.MjData(m)
    if (m.nq,m.nv)!=(38,37):raise ValueError('Original X2 required')
    names=[m.joint(i).name.rsplit('/',1)[-1] for i in range(1,m.njnt)]
    if names!=PROFILE['joint_names']:raise ValueError('Joint order changed')
    effort=np.array([PROFILE['effort_limit'][n] for n in names]);zero=np.zeros((37,0))
    m.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_CONTACT)|int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
    with np.load(a.original) as z:
        fps=float(np.asarray(z['fps']).ravel()[0]);original=z['qpos'][::int(fps/50)].copy()
    with np.load(a.corrected) as z:
        if float(np.asarray(z['fps']).ravel()[0])!=50:raise ValueError('Corrected50Hz expected')
        corrected=z['qpos'].copy()
    def evaluate(pose,velocity,acceleration):
        d.qpos[:]=pose;d.qvel[:]=velocity;d.qacc[:]=acceleration;mujoco.mj_inverse(m,d)
        required=d.qfrc_inverse.copy();fit=solve_contact_effort(zero,required,effort)
        return dict(base_feasible=bool(fit.success),within_original_motor_caps=bool(fit.success and fit.x[-1]<=1),
            motor_ratio=float(fit.x[-1]) if fit.success else None,root_wrench_max_abs=float(abs(required[:6]).max()))
    rows=[];forward=np.zeros(37);backward=np.zeros(37)
    for frame in [121,122,123,124,125]:
        item=dict(frame50=frame)
        for name,poses in [('original',original),('corrected',corrected)]:
            mujoco.mj_differentiatePos(m,forward,.02,poses[frame],poses[frame+1])
            mujoco.mj_differentiatePos(m,backward,-.02,poses[frame],poses[frame-1])
            item[name]=evaluate(poses[frame],(forward+backward)/2,(forward-backward)*50.)
        item['held_in_air']=evaluate(corrected[frame],np.zeros(37),np.zeros(37));rows.append(item)
    pose=corrected[123].copy();pose[2]+=10.;acc=np.zeros(37);acc[:3]=m.opt.gravity
    ballistic=evaluate(pose,np.zeros(37),acc);held=evaluate(pose,np.zeros(37),np.zeros(37))
    passed=bool(ballistic['within_original_motor_caps'] and not held['base_feasible']
        and all(r['corrected']['within_original_motor_caps'] and not r['held_in_air']['base_feasible'] for r in rows)
        and any(not r['original']['base_feasible'] for r in rows))
    a.out.mkdir(parents=True);sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    report=dict(passed=passed,synthetic_native_ballistic=ballistic,synthetic_native_held_air=held,rows=rows,
        source_sha256={str(f):sha(f) for f in [a.model,a.original,a.corrected]},
        scope=__doc__,lp_function_unchanged=True,physical_caps_unchanged=True,actual_controller_execution=False)
    (a.out/'report.json').write_text(json.dumps(report,indent=2));(a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report,indent=2))
    if not passed:raise RuntimeError('Zero-contact native mechanics regression failed')


if __name__=='__main__':main()
