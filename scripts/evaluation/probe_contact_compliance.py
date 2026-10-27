"""Measure default contact compression during a bounded hold and a gravity drop.

Uses the project's unchanged robot/terrain and actuator gains/limits. Only the
initial height and servo commands vary; this is not a Webster success test.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--drop-height',type=float,default=.3)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    from supermimic.envs import SuperMimicEnvCfgArgs,make_env
    from supermimic.envs.state_io import StateIO
    from supermimic.data.contact_wrench import solve_contact_effort
    from supermimic.data.design_metrics import BatchedDesignMetrics
    from supermimic.robots.dimensions import PROFILE
    env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(args.source)],num_envs=1,
        action_noise_std=0.,obs_noise=False,reset_noise=False,domain_randomization=False,
        sampling_mode='start',record_transitions=False,episode_length_s=1e5),args.device)
    try:
        cmd=env.command_manager.get_term('motion');cmd.forced_frame[:]=cmd.library.clip_start[0]
        env.reset();io=StateIO(env);initial=io.read();nq=env.sim.mj_model.nq
        initial[:,nq:]=0.
        m=env.sim.mj_model;d=mujoco.MjData(m)
        d.qpos[:]=initial[0,:nq].cpu().numpy();mujoco.mj_inverse(m,d)
        required=d.qfrc_inverse.copy();columns=[]
        jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv))
        rays=np.array([[x/np.sqrt(2),y/np.sqrt(2),1] for x,y in [(1,1),(1,-1),(-1,1),(-1,-1)]]).T
        for g in range(m.ngeom):
            if 'foot_' not in m.geom(g).name:continue
            k=m.geom_dataid[g];a,n=m.mesh_vertadr[k],m.mesh_vertnum[k]
            world=m.mesh_vert[a:a+n]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
            near=world[(world[:,2]<.012)&(world[:,2]<world[:,2].min()+.003)]
            if len(near)>12:near=near[np.linspace(0,len(near)-1,12).astype(int)]
            for point in near:
                mujoco.mj_jac(m,d,jp,jr,point,int(m.geom_bodyid[g]));columns.append(jp.T@rays)
        names=list(PROFILE['joint_names'])
        effort=np.array([PROFILE['effort_limit'][n] for n in names])
        basis=np.concatenate(columns,axis=1)
        fit=solve_contact_effort(basis,required,effort)
        if not fit.success:raise RuntimeError('Initial static contact LP infeasible; calibration rejected')
        tau=required[6:]-basis[6:]@fit.x[:-1]
        kp=np.array([PROFILE['stiffness'][n] for n in names])
        default=np.array([PROFILE['default_joint_pos'][n] for n in names])
        scale=PROFILE['action_scale']
        scale=np.array([scale[n] for n in names]) if isinstance(scale,dict) else scale
        qref=initial[0,7:nq].cpu().numpy()
        actions=[torch.tensor((qref+ff*tau/kp-default)/scale,dtype=torch.float32,device=args.device)[None]
                 for ff in [0.,1.]]
        metrics=BatchedDesignMetrics(m,torch.device(args.device))
        lower,upper=m.jnt_range[1:].T
        env.termination_manager.compute=lambda:torch.zeros(1,dtype=torch.bool,device=args.device)
        original_step=env.sim.step
        traces=[];states=[]
        def capture(*a,**kw):
            result=original_step(*a,**kw)
            q=env.sim.data.qpos[0].cpu().numpy().copy()
            v=env.sim.data.qvel[0].cpu().numpy().copy()
            f=env.sim.data.actuator_force[0].cpu().numpy()
            traces.append(np.r_[metrics(env.sim.data)[0].cpu().numpy(),
                np.maximum(np.maximum(lower-q[7:],q[7:]-upper),0).max(),(abs(f)/effort).max()])
            states.append(np.r_[q,v]);return result
        env.sim.step=capture
        reports=[]
        for height in [0.,args.drop_height]:
            env.reset();state=initial.clone();state[:,2]+=height
            io.write(torch.tensor([0],device=args.device),state)
            initial_metrics=metrics(env.sim.data)[0].cpu().numpy()
            traces.clear();states.clear()
            with torch.no_grad():
                for _ in range(round(3./env.step_dt)):
                    near_ground=min((traces[-1] if traces else initial_metrics)[7:9])<.03
                    env.step(actions[int(near_ground)])
            a=np.array(traces);valid=(a[:,12]<1e-4)&(a[:,9]>.005)&(a[:,11]>.9)
            penetration=np.maximum(-a[:,7:9].min(axis=1),0)
            report={'initial_extra_height_m':height,'physics_steps':len(a),
                'maximum_foot_penetration_m':float(penetration.max()),
                'maximum_foot_penetration_with_hard_limits_and_clear_body_m':float(penetration[valid].max()) if valid.any() else None,
                'hard_valid_clear_body_steps':int(valid.sum()),'maximum_hard_violation_rad':float(a[:,12].max()),
                'maximum_effort_ratio':float(a[:,13].max()),'minimum_com_vz':float(a[:,1].min()),
                'tail_max_abs_com_vz':float(abs(a[-100:,1]).max()),'tail_min_root_up_z':float(a[-100:,11].min()),
                'tail_min_com_z':float(a[-100:,0].min())}
            reports.append(report)
            np.savez(args.out/f'height-{height}.npz',state=np.array(states),metrics=a,physics_dt=np.array([m.opt.timestep]))
        report={'purpose':__doc__,'approved_for_training':False,'static_feedforward_peak_effort_ratio':float(fit.x[-1]),
            'cases':reports,'note':'Compression during a diagnostic hold/drop is not proof of a valid jump, flip, or landing.'}
        (args.out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    finally:env.close()


if __name__=='__main__':main()
