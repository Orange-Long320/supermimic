"""Find a static single-foot equilibrium under the native soft-contact law."""
import argparse
import hashlib
import json
from types import SimpleNamespace
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--support-reserve-goal',type=float,help='Seek a lower static support-leg motor fraction before final force-balance polishing')
    p.add_argument('--candidate-from',type=Path,help='Polish a recorded candidate while preserving its contact attitude')
    p.add_argument('--candidate-index',type=int,default=0)
    p.add_argument('--fixed-pressure-grid',action='store_true',help='Balance each fixed foot attitude independently; keep individual states for dynamic screening')
    p.add_argument('--pressure-angles',type=float,nargs='+',default=[-.02,-.012,-.006,.006,.012,.02])
    p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'calibration-source.py').write_text(Path(__file__).read_text())
    from supermimic.robots.dimensions import PROFILE
    m=mujoco.MjModel.from_binary_path(str(args.model));d=mujoco.MjData(m)
    state=np.load(args.initial)['state'][0];q0=state[:m.nq].copy()
    d.qpos[:]=q0;mujoco.mj_forward(m,d)
    foot=m.body('robot/left_ankle_roll_link').id
    foot_position=d.xpos[foot].copy();foot_rotation=Rotation.from_matrix(d.xmat[foot].reshape(3,3))
    local=mujoco.MjData(m)
    mass=float(m.body_mass.sum());scale=np.r_[np.full(3,mass*9.81),np.full(3,mass*9.81*.7)]
    effort=np.array([PROFILE['effort_limit'][n] for n in PROFILE['joint_names']])
    ankle_joints=[m.joint('robot/left_ankle_pitch_joint').id,m.joint('robot/left_ankle_roll_joint').id]
    ankle_qpos=m.jnt_qposadr[ankle_joints]
    joint_lower=np.maximum(m.jnt_range[ankle_joints,0]+.02-q0[ankle_qpos],-.15)
    joint_upper=np.minimum(m.jnt_range[ankle_joints,1]-.02-q0[ankle_qpos],.15)
    meshes=[]
    for g in range(m.ngeom):
        if not (m.geom_contype[g] or m.geom_conaffinity[g]) or m.geom_dataid[g]<0:continue
        k=m.geom_dataid[g];a,n=m.mesh_vertadr[k],m.mesh_vertnum[k]
        side='left' if 'foot_' in m.geom(g).name and 'left' in m.body(int(m.geom_bodyid[g])).name else 'right' if 'foot_' in m.geom(g).name else 'nonfoot'
        meshes.append((g,m.mesh_vert[a:a+n].copy(),side))
    def evaluate(x):
        q=q0.copy()
        q[ankle_qpos]+=x[3:]
        # Place the support foot exactly, independently of body balance.
        # This prevents tiny cancellation errors from changing CCD branches.
        local.qpos[:]=q;local.qpos[:3]=0;local.qpos[3:7]=[1,0,0,0]
        mujoco.mj_kinematics(m,local)
        target_rotation=Rotation.from_rotvec([x[1],x[2],0])*foot_rotation
        root_rotation=target_rotation*Rotation.from_matrix(local.xmat[foot].reshape(3,3)).inv()
        q[3:7]=root_rotation.as_quat()[[3,0,1,2]]
        q[:3]=foot_position+[0,0,x[0]]-root_rotation.apply(local.xpos[foot])
        d.qpos[:]=q;d.qvel[:]=0;d.qacc[:]=0;mujoco.mj_inverse(m,d)
        heights={s:min(float((v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])[:,2].min())
            for g,v,k in meshes if k==s) for s in ['left','right','nonfoot']}
        return q,d.qfrc_inverse.copy(),heights
    regularization=1e-4
    reserve_phase=True
    def residual(x):
        _,force,heights=evaluate(x)
        weight=20. if args.support_reserve_goal is not None and reserve_phase else 1.
        reserve=np.maximum(abs(force[6:12])/effort[:6]-args.support_reserve_goal,0) if args.support_reserve_goal is not None and reserve_phase else np.zeros(6)
        return np.r_[force[:6]/scale*weight,max(.04-heights['right'],0)*50,max(.005-heights['nonfoot'],0)*50,
            max(-.005-heights['left'],0)*50,
            np.maximum(abs(force[6:])/effort-1,0),reserve,x*regularization]
    def jacobian(x):
        # Relative steps at zero rotation stayed below collision tolerances.
        # Use absolute spatial steps inside each active contact branch.
        steps=np.array([1e-6,1e-5,1e-5,1e-5,1e-5]);columns=[]
        for j,step in enumerate(steps):
            delta=np.zeros(len(x));delta[j]=step
            columns.append((residual(x+delta)-residual(x-delta))/(2*step))
        return np.stack(columns,axis=1)
    trials=[]
    lower=np.r_[[-.12,-.05,-.05],joint_lower];upper=np.r_[[.01,.05,.05],joint_upper]
    pressure_angles=[(roll,pitch) for roll in [-.003,.003] for pitch in [-.003,.003]]
    if args.support_reserve_goal is not None:pressure_angles += [(-.025,-.03),(-.02,.02),(.02,-.03),(.02,.02)]
    seeds=[np.array([z,roll,pitch,.06,-.035]) for z in [-.006,-.012] for roll,pitch in pressure_angles]
    if args.fixed_pressure_grid:
        seeds=[np.array([-.006,roll,pitch,.06,-.035]) for roll in args.pressure_angles for pitch in args.pressure_angles]
    if args.candidate_from:
        saved=json.loads(args.candidate_from.read_text())['trials'][args.candidate_index]
        if saved['parameter_names']!=['foot_z','foot_roll','foot_pitch','left_ankle_pitch','left_ankle_roll']:
            raise ValueError('Candidate parameter convention differs')
        seeds=[np.asarray(saved['adjustments'])]
    for seed in seeds:
        regularization=1e-4;reserve_phase=True
        if args.candidate_from or args.fixed_pressure_grid:result=SimpleNamespace(x=seed.copy(),nfev=0,cost=0.)
        else:
            result=least_squares(residual,np.clip(seed,lower,upper),jac=jacobian,bounds=(lower,upper),
                xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=400)
            regularization=0.;reserve_phase=False
            result=least_squares(residual,result.x,jac=jacobian,bounds=(lower,upper),
                xtol=1e-13,ftol=1e-13,gtol=1e-13,max_nfev=100)
        # Separate body balance from pressure attitude so contact vertices do
        # not jump while the final unactuated wrench is being zeroed.
        fixed=result.x.copy();indices=np.array([0,3,4])
        def balance(y):
            x=fixed.copy();x[indices]=y
            return evaluate(x)[1][:6]/scale
        def balance_jac(y):
            steps=np.array([1e-6,1e-5,1e-5]);columns=[]
            for j,step in enumerate(steps):
                delta=np.zeros(3);delta[j]=step
                columns.append((balance(y+delta)-balance(y-delta))/(2*step))
            return np.stack(columns,axis=1)
        polished=least_squares(balance,fixed[indices],jac=balance_jac,bounds=(lower[indices],upper[indices]),
            xtol=1e-13,ftol=1e-13,gtol=1e-13,max_nfev=150)
        fixed[indices]=polished.x
        result=SimpleNamespace(x=fixed,cost=polished.cost,nfev=result.nfev+polished.nfev)
        q,force,heights=evaluate(result.x)
        contact_bodies=sorted(set(m.body(int(m.geom_bodyid[int(g)])).name for c in d.contact[:d.ncon] for g in c.geom if int(m.geom_bodyid[int(g)])!=0))
        item={'parameter_names':['foot_z','foot_roll','foot_pitch','left_ankle_pitch','left_ankle_roll'],
            'adjustments':result.x.tolist(),'cost':float(result.cost),'evaluations':int(result.nfev),
            'root_force_residual_normalized':float(abs(force[:6]/scale).max()),
            'root_force_residual':force[:6].tolist(),'maximum_motor_ratio':float((abs(force[6:])/effort).max()),
            'maximum_support_motor_ratio':float((abs(force[6:12])/effort[:6]).max()),
            'heights':heights,'contact_bodies':contact_bodies}
        item['static_force_feasible']=(item['root_force_residual_normalized']<1e-5 and
            item['maximum_motor_ratio']<=1 and heights['right']>.04 and heights['nonfoot']>.005 and
            all(name=='terrain' or name.endswith('/left_ankle_roll_link') for name in contact_bodies))
        item['strict_geometry_static_feasible']=item['static_force_feasible'] and heights['left']>=-.005
        trials.append((item,q,force))
    if args.support_reserve_goal is None:
        trials.sort(key=lambda t:(not t[0]['strict_geometry_static_feasible'],not t[0]['static_force_feasible'],t[0]['cost']))
    else:
        # This ranks candidates for an actual closed-loop test; it is not a
        # relaxation of force/geometry acceptance or a training certificate.
        trials.sort(key=lambda t:(t[0]['heights']['left']<-.005 or t[0]['heights']['right']<.04 or
            t[0]['heights']['nonfoot']<.005 or t[0]['maximum_motor_ratio']>1,
            1000*t[0]['root_force_residual_normalized']+t[0]['maximum_support_motor_ratio']))
    best,q,force=trials[0]
    if args.fixed_pressure_grid:
        for index,(item,pose,wrench) in enumerate(trials):
            name=f'candidate-{index:03d}.npz';item['state_file']=name
            np.savez(args.out/name,qpos=pose,qvel=np.zeros(m.nv),required_motor_torque=wrench[6:],
                joint_names=np.array(PROFILE['joint_names']))
    np.savez(args.out/'equilibrium.npz',qpos=q,qvel=np.zeros(m.nv),required_motor_torque=force[6:],
        joint_names=np.array(PROFILE['joint_names']))
    report={'purpose':__doc__,'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),
        'initial_sha256':hashlib.sha256(args.initial.read_bytes()).hexdigest(),
        'configuration':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'support_reserve_goal':args.support_reserve_goal,
        'candidate_source':str(args.candidate_from) if args.candidate_from else None,'candidate_index':args.candidate_index,
        'mass_kg':mass,'best':best,'trials':[t[0] for t in trials],'approved_for_training':False,
        'note':'Static inverse-dynamics balance only; closed-loop hold, takeoff and flip remain unproven.'}
    (args.out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps({
        'mass_kg':mass,'best':best,'trials':len(trials),'approved_for_training':False}),flush=True)


if __name__=='__main__':main()
