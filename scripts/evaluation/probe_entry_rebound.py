"""Batch-test incoming-motion capture in unchanged physics with fixed servo targets."""
import argparse,hashlib,itertools,json,os,time
from pathlib import Path
import mujoco
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--equilibrium',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--seconds',type=float,default=1.);p.add_argument('--vx',type=float,nargs='+',default=[1.,1.5,2.])
    p.add_argument('--vz',type=float,nargs='+',default=[-1.2,-.6,0.]);p.add_argument('--hy',type=float,nargs='+',default=[0.,8.86])
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    source=Path(__file__).read_bytes();(args.out/'probe-source.py').write_bytes(source)
    from supermimic.envs import SuperMimicEnvCfgArgs,make_env
    from supermimic.envs.state_io import StateIO
    from supermimic.data.entry_state import entry_velocity_state
    from supermimic.data.design_metrics import BatchedDesignMetrics
    from supermimic.robots.dimensions import PROFILE
    grid=[(0.,0.,0.)]+list(itertools.product(args.vx,args.vz,args.hy));n=len(grid)
    env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(args.source)],num_envs=n,action_noise_std=0.,
        obs_noise=False,reset_noise=False,domain_randomization=False,sampling_mode='start',
        record_transitions=False,episode_length_s=1e5),args.device)
    states=[];measures=[];forces=[];started=time.monotonic()
    configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    configuration.update(pid=os.getpid(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    (args.out/'configuration.json').write_text(json.dumps(configuration,indent=2))
    try:
        env.command_manager.get_term('motion').forced_frame[:]=0;env.reset();m=env.sim.mj_model;io=StateIO(env);initial=io.read()
        names=list(PROFILE['joint_names']);limits=np.array([PROFILE['velocity_limit'][x] for x in names])
        effort=np.array([PROFILE['effort_limit'][x] for x in names]);kp=np.array([PROFILE['stiffness'][x] for x in names])
        with np.load(args.equilibrium) as eq:
            assert eq['joint_names'].tolist()==names;q0=eq['qpos'].copy();tau0=eq['required_motor_torque'].copy()
        cases=[];velocities=[]
        for vx,vz,hy in grid:
            velocity,audit=entry_velocity_state(m,q0,np.zeros(m.nv),vx,limits,
                com_velocity=np.array([vx,0,vz]),angular_momentum=np.array([0,hy,0]))
            cases.append({'vx':vx,'vz':vz,'hy':hy,'initial_audit':audit});velocities.append(velocity)
        initial[:,:m.nq]=torch.tensor(np.tile(q0,(n,1)),dtype=initial.dtype,device=args.device)
        initial[:,m.nq:]=torch.tensor(np.asarray(velocities),dtype=initial.dtype,device=args.device)
        io.write(torch.arange(n,device=args.device),initial);mujoco.mj_saveModel(m,str(args.out/'native-model.mjb'))
        torch.testing.assert_close(io.read()[:,:m.nq],torch.tensor(np.tile(q0,(n,1)),dtype=initial.dtype,device=args.device),rtol=0,atol=2e-5)
        np.savez(args.out/'initial-state.npz',state=initial.cpu().numpy(),env_origins=env.scene.env_origins.cpu().numpy())
        target=torch.tensor(np.tile(q0[7:]+tau0/kp,(n,1)),dtype=initial.dtype,device=args.device)
        assert np.allclose(m.actuator_gainprm[:,0],kp)
        metrics=BatchedDesignMetrics(m,args.device);original_step=env.sim.step
        def step(*a,**kw):
            env.sim.data.ctrl[:]=target;result=original_step(*a,**kw)
            states.append(torch.cat([env.sim.data.qpos[:],env.sim.data.qvel[:]],dim=-1).cpu().numpy().copy())
            measures.append(metrics(env.sim.data).cpu().numpy().copy());forces.append(env.sim.data.actuator_force[:].cpu().numpy().copy())
            return result
        env.sim.step=step;env.termination_manager.compute=lambda:torch.zeros(n,dtype=torch.bool,device=args.device)
        for _ in range(round(args.seconds/env.step_dt)):env.step(torch.zeros((n,len(names)),device=args.device))
        a=np.asarray(states);x=np.asarray(measures);tau=np.asarray(forces);dt=float(m.opt.timestep);lo,hi=m.jnt_range[1:].T
        np.savez(args.out/'trajectory.npz',state=a,metrics=x,actuator_force=tau,servo_target=target.cpu().numpy(),fps=np.array([1/dt]))
        for i,case in enumerate(cases):
            rows=x[:,i];peak=int(np.argmax(rows[:,1]));prefix=a[:peak+1,i];p=rows[:peak+1]
            hard=float(np.maximum(np.maximum(lo-prefix[:,7:m.nq],prefix[:,7:m.nq]-hi),0).max())
            speed=float((abs(prefix[:,m.nq+6:])/limits).max());motor=float((abs(tau[:peak+1,i])/effort).max())
            right=float(p[:,8].min());nonfoot=float(p[:,9].min());left=float(p[:,7].min())
            sample=rows[3::4];valid=(sample[:,7]>.02)&(sample[:,8]>.02)&(sample[:,1]>0)&(sample[:,9]>.005)
            longest=run=0
            for value in valid:run=run+1 if value else 0;longest=max(longest,run)
            case['result']={'peak_vz':float(rows[peak,1]),'peak_time_s':(peak+1)*dt,'hy_at_peak':float(rows[peak,3]),
                'com_height_at_peak':float(rows[peak,0]),'minimum_vz_before_peak':float(p[:,1].min()),
                'prefix_hard_violation':hard,'prefix_speed_ratio':speed,'prefix_effort_ratio':motor,
                'prefix_minimum_left_height':left,'prefix_minimum_right_height':right,'prefix_minimum_nonfoot_height':nonfoot,
                'prefix_physical_limits_passed':bool(hard<1e-4 and speed<=1 and motor<=1.00001 and min(left,right)>-.005 and nonfoot>.005),
                'single_left_support_through_peak':bool(right>.04 and left<.01),
                'longest_ascending_air_s':longest*env.step_dt,'approved_for_training':False}
        (args.out/'result.json').write_text(json.dumps({'purpose':__doc__,'configuration':configuration,'cases':cases,
            'elapsed_s':time.monotonic()-started,'source_sha256':hashlib.sha256(source).hexdigest(),
            'equilibrium_sha256':hashlib.sha256(args.equilibrium.read_bytes()).hexdigest(),
            'state_edits_after_start':False,'physical_parameters_unchanged':True,'approved_for_training':False,
            'note':'Diagnostic initial-state grid with fixed original position servos. A rebound is not a Webster.'},indent=2))
        print(json.dumps({'cases':len(cases),'elapsed_s':time.monotonic()-started,
            'best':max(cases,key=lambda c:c['result']['peak_vz'])}),flush=True)
    finally:env.close()


if __name__=='__main__':main()
