"""Batch-screen calibrated poses in native physics before costly feedback probes."""
import argparse,json,hashlib,os
from pathlib import Path
import numpy as np
import torch


def summarize_cases(cases,a,values,nq,low,high,support_mode='left'):
    for i,item in enumerate(cases):
        vel=float(abs(a[:,i,nq:]).max());hard=float(np.maximum(np.maximum(low-a[:,i,7:nq],a[:,i,7:nq]-high),0).max())
        x=values[:,i]
        right_ok=x[:,8].min()>.04 if support_mode=='left' else (x[:,8].min()>-.005 and x[:,8].max()<.01)
        left_ok=x[:,7].min()>-.005 and (support_mode=='left' or x[:,7].max()<.01)
        item['screen']={'max_generalized_velocity':vel,'hard_violation':hard,
            'minimum_left_height':float(x[:,7].min()),'minimum_right_height':float(x[:,8].min()),'minimum_nonfoot_height':float(x[:,9].min()),
            'maximum_com_speed':float(x[:,5].max()),'screen_passed':bool(vel<.03 and hard<1e-4 and left_ok and right_ok and x[:,9].min()>.005)}
    cases.sort(key=lambda x:(not x['screen']['screen_passed'],x['maximum_motor_ratio']))
    return cases


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--seconds',type=float,default=.2)
    p.add_argument('--support-mode',choices=['left','both'],default='left')
    p.add_argument('--include-unbalanced',action='store_true',help='Diagnose force-unbalanced candidates without granting a static certificate')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    from supermimic.envs import SuperMimicEnvCfgArgs,make_env
    from supermimic.envs.state_io import StateIO
    from supermimic.data.design_metrics import BatchedDesignMetrics
    from supermimic.robots.dimensions import PROFILE
    torch.set_num_threads(1)
    program=Path(__file__).read_bytes();(args.out/'screen-source.py').write_bytes(program)
    report=json.loads((args.calibration/'result.json').read_text())
    cases=[x for x in report['trials'] if x['strict_geometry_static_feasible']]
    if args.include_unbalanced:
        if args.support_mode!='both':raise ValueError('Unbalanced diagnostic is only supported for two-foot poses')
        cases=[x for x in report['trials'] if x['maximum_motor_ratio']<=1 and x.get('hard_violation_rad',1)<1e-9
               and min(x['heights'][s] for s in ['left','right'])>-.005
               and max(x['heights'][s] for s in ['left','right'])<.005 and x['heights']['nonfoot']>.005]
    if not cases:raise ValueError('No eligible candidates to screen')
    trace_case_files=[x['state_file'] for x in cases]
    env=make_env(SuperMimicEnvCfgArgs(npz_files=[args.source],num_envs=len(cases),action_noise_std=0.,
        obs_noise=False,reset_noise=False,domain_randomization=False,sampling_mode='start',
        record_transitions=False,episode_length_s=1e5),args.device)
    try:
        env.command_manager.get_term('motion').forced_frame[:]=0;env.reset();io=StateIO(env);state=io.read()
        m=env.sim.mj_model;names=list(PROFILE['joint_names']);poses=[];torques=[]
        for item in cases:
            with np.load(args.calibration/item['state_file']) as z:
                assert z['joint_names'].tolist()==names
                poses.append(z['qpos']);torques.append(z['required_motor_torque'])
        pose=np.asarray(poses);state[:,:m.nq]=torch.tensor(pose,dtype=state.dtype,device=args.device)
        state[:,m.nq:]=0
        io.write(torch.arange(len(cases),device=args.device),state)
        torch.testing.assert_close(io.read()[:,:m.nq],torch.tensor(pose,dtype=state.dtype,device=args.device),rtol=0,atol=2e-5)
        kp=np.asarray([PROFILE['stiffness'][n] for n in names]);effort=np.asarray([PROFILE['effort_limit'][n] for n in names])
        target=torch.tensor(pose[:,7:]+np.asarray(torques)/kp,dtype=state.dtype,device=args.device)
        metrics=BatchedDesignMetrics(m,args.device);original_step=env.sim.step;traces=[];measurements=[]
        def step(*a,**kw):
            env.sim.data.ctrl[:]=target;result=original_step(*a,**kw)
            env.sim.forward()
            traces.append(torch.cat([env.sim.data.qpos[:],env.sim.data.qvel[:]],dim=-1).cpu().numpy().copy())
            measurements.append(metrics(env.sim.data).cpu().numpy().copy());return result
        env.sim.step=step;env.termination_manager.compute=lambda:torch.zeros(len(cases),dtype=torch.bool,device=args.device)
        for _ in range(round(args.seconds/env.step_dt)):env.step(torch.zeros((len(cases),len(names)),device=args.device))
        a=np.asarray(traces);values=np.asarray(measurements);low,high=m.jnt_range[1:].T
        cases=summarize_cases(cases,a,values,m.nq,low,high,args.support_mode)
        np.savez(args.out/'trajectory.npz',state=a,metrics=values,fps=np.array([200.]))
        output={'purpose':__doc__,'device':args.device,'pid':os.getpid(),'seconds':args.seconds,'support_mode':args.support_mode,
            'include_unbalanced':args.include_unbalanced,'cases':cases,'trajectory_case_files':trace_case_files,
            'state_edits_after_start':False,'approved_for_training':False,
            'source_sha256':hashlib.sha256(program).hexdigest(),'metric_state_alignment':'Post-integration derived fields refreshed before measurement',
            'note':'Constant native motor targets over a short interval; passing requires longer feedback validation.'}
        (args.out/'result.json').write_text(json.dumps(output,indent=2))
        print(json.dumps({'count':len(cases),'screen_passed':sum(x['screen']['screen_passed'] for x in cases),'best':cases[0]}),flush=True)
    finally:env.close()


if __name__=='__main__':main()
