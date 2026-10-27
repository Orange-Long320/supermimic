"""Search bounded native motor-command profiles; full Webster approval is separate."""
import argparse,hashlib,itertools,json,os,time
from pathlib import Path
import mujoco
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--equilibrium',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--population',type=int,default=512);p.add_argument('--iterations',type=int,default=0)
    p.add_argument('--max-seconds',type=float,default=180.);p.add_argument('--seconds',type=float,default=.6)
    p.add_argument('--vx',type=float,default=0.);p.add_argument('--vz',type=float,default=0.)
    p.add_argument('--hy',type=float,default=0.);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--warmstart',type=Path)
    p.add_argument('--template',type=Path,help='Refine an actual recorded native-control prefix')
    p.add_argument('--template-steps',type=int,help='Use only this many recorded physics steps as the motor prefix')
    p.add_argument('--swing-hip-span',type=float,default=.6)
    p.add_argument('--state-aligned-metrics',action='store_true',help='Refresh derived fields after integration before measuring')
    p.add_argument('--residual-std',type=float,default=.15)
    p.add_argument('--exploration-fraction',type=float,default=.15)
    p.add_argument('--bang-bang',action='store_true',help='Separate extension and rotation/unloading torque phases')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    if args.template and args.bang_bang:p.error('Recorded-target refinement and two-phase raw torque are distinct modes')
    program=Path(__file__).read_bytes();(args.out/'as-run.py').write_bytes(program)
    from supermimic.envs import SuperMimicEnvCfgArgs,make_env
    from supermimic.envs.state_io import StateIO
    from supermimic.data.entry_state import entry_velocity_state
    from supermimic.data.design_metrics import BatchedDesignMetrics
    from supermimic.robots.dimensions import PROFILE
    n=args.population;env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(args.source)],num_envs=n,
        action_noise_std=0.,obs_noise=False,reset_noise=False,domain_randomization=False,sampling_mode='start',
        record_transitions=False,episode_length_s=1e5),args.device)
    configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    configuration.update(pid=os.getpid(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        source_sha256=hashlib.sha256(program).hexdigest(),equilibrium_sha256=hashlib.sha256(args.equilibrium.read_bytes()).hexdigest())
    (args.out/'configuration.json').write_text(json.dumps(configuration,indent=2))
    try:
        cmd=env.command_manager.get_term('motion');cmd.forced_frame[:]=0;env.reset();m=env.sim.mj_model;io=StateIO(env)
        names=list(PROFILE['joint_names']);effort=np.array([PROFILE['effort_limit'][x] for x in names])
        assert [m.joint(i).name.rsplit('/',1)[-1] for i in range(1,m.njnt)]==names
        assert [m.joint(int(i)).name.rsplit('/',1)[-1] for i in m.actuator_trnid[:,0]]==names
        kp=np.array([PROFILE['stiffness'][x] for x in names]);kd=np.array([PROFILE['damping'][x] for x in names])
        limits=np.array([PROFILE['velocity_limit'][x] for x in names]);lo,hi=m.jnt_range[1:].T
        with np.load(args.equilibrium) as z:q0=z['qpos'].copy();tau0=z['required_motor_torque'].copy();assert z['joint_names'].tolist()==names
        incoming,entry=entry_velocity_state(m,q0,np.zeros(m.nv),args.vx,limits,
            com_velocity=[args.vx,0,args.vz],angular_momentum=[0,args.hy,0])
        template=None;template_state=None;template_duration=None
        if args.template:
            with np.load(args.template/'initial-state.npz') as z:template_initial=z['state'][0].copy()
            np.testing.assert_allclose(template_initial[:m.nq],q0,atol=1e-6)
            q0=template_initial[:m.nq];incoming=template_initial[m.nq:]
            with np.load(args.template/'trajectory.npz') as z:
                template=z['servo_target'].copy();template_state=z['state'].copy();template_dt=1/float(z['fps'][0])
            if args.template_steps is not None:
                if not 2<=args.template_steps<=len(template):raise ValueError('Invalid template prefix length')
                template=template[:args.template_steps];template_state=template_state[:args.template_steps]
            template_duration=len(template)*template_dt
            entry={'scope':'Original initial state of recorded prefix; no state edits after reset',
                'template':str(args.template),'initial_state_sha256':hashlib.sha256((args.template/'initial-state.npz').read_bytes()).hexdigest(),
                'trajectory_sha256':hashlib.sha256((args.template/'trajectory.npz').read_bytes()).hexdigest()}
        (args.out/'entry.json').write_text(json.dumps(entry,indent=2));mujoco.mj_saveModel(m,str(args.out/'native-model.mjb'))
        tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device=args.device)
        kp_t,kd_t,eff_t,lim_t,low_t,high_t=map(tensor,[kp,kd,effort,limits,lo,hi])
        assert np.allclose(m.actuator_gainprm[:,0],kp)
        assert np.allclose(m.actuator_biasprm[:,1],-kp) and np.allclose(m.actuator_biasprm[:,2],-kd)
        qdefault=tensor(q0[7:]+tau0/kp)
        sag=[names.index('left_'+s+'_joint') for s in ['hip_pitch','knee','ankle_pitch']]
        right=[names.index('right_'+s+'_joint') for s in ['hip_pitch','knee']]
        arms=[names.index(s+'_shoulder_pitch_joint') for s in ['left','right']]
        balance=[names.index(x) for x in ['left_hip_roll_joint','left_ankle_roll_joint','waist_roll_joint']]
        waist=names.index('waist_pitch_joint')
        tuck=q0[7:].copy()
        for side in ['left','right']:
            for suffix,value in [('hip_pitch',-2.4),('knee',2.35),('shoulder_pitch',-1.),('elbow',-1.4)]:
                tuck[names.index(side+'_'+suffix+'_joint')]=value
        tuck[waist]=.25;tuck_t=tensor(np.clip(tuck,lo+.025,hi-.025))
        metrics=BatchedDesignMetrics(m,args.device);initial=io.read();initial[:,:m.nq]=tensor(np.tile(q0,(n,1)))
        initial[:,m.nq:]=tensor(np.tile(incoming,(n,1)))
        original_step=env.sim.step;clock={'i':0,'parameters':None};state_trace=[];metric_trace=[];force_trace=[];control_trace=[]
        active=torch.ones(n,dtype=torch.bool,device=args.device);run=torch.zeros(n,dtype=torch.long,device=args.device)
        completed=torch.zeros(n,dtype=torch.bool,device=args.device);end_index=torch.full((n,),-1,dtype=torch.long,device=args.device)
        maxima=torch.zeros((n,3),device=args.device);minimum=torch.full((n,3),float('inf'),device=args.device)
        template_t=tensor(template) if template is not None else None
        def checked_step(*a,**kw):
            param=clock['parameters'];t=clock['i']*float(m.opt.timestep);q=env.sim.data.qpos[:,7:];v=env.sim.data.qvel[:,6:]
            end_time=param[:,0]*template_duration if template_t is not None else param[:,0]
            u=torch.clamp(t/end_time,0,1);smooth=u**3*(10-15*u+6*u*u)
            target=qdefault.expand(n,-1).clone();target[:,right]=qdefault[right][None]+smooth[:,None]*(param[:,7:9]-tensor(q0[7:][right]))
            target[:,arms]=qdefault[arms][None]+smooth[:,None]*(param[:,9:10]-tensor(q0[7:][arms]))
            target[:,waist]=qdefault[waist]+smooth*(param[:,10]-q0[7+waist])
            target[:,balance]+=param[:,11:14]*eff_t[balance]/kp_t[balance]
            force_fraction=param[:,1:4]*(1-u[:,None])+param[:,4:7]*u[:,None]
            if args.bang_bang:force_fraction=torch.where(u[:,None]<param[:,14:15],param[:,1:4],param[:,4:7])
            target[:,sag]=q[:,sag]+(force_fraction*eff_t[sag]+kd_t[sag]*v[:,sag])/kp_t[sag]
            if template_t is not None:
                position=torch.clamp(t/param[:,0]/template_dt,0,len(template)-1)
                first=position.long().clamp(max=len(template)-2);fraction=(position-first)[:,None]
                target=template_t[first]*(1-fraction)+template_t[first+1]*fraction
                target=torch.where((param[:,0]==1)[:,None],template_t[min(clock['i'],len(template)-1)][None],target)
                target[:,sag]+=force_fraction*eff_t[sag]/kp_t[sag]
                target[:,right]+=smooth[:,None]*param[:,7:9]
                target[:,arms]+=smooth[:,None]*param[:,9:10]
                target[:,waist]+=smooth*param[:,10]
                target[:,balance]+=param[:,11:14]*eff_t[balance]/kp_t[balance]
            target=torch.where((t>=end_time)[:,None],tuck_t[None],target)
            env.sim.data.ctrl[:]=target;result=original_step(*a,**kw)
            qnext=env.sim.data.qpos[:,7:];vnext=env.sim.data.qvel[:,6:];tau=env.sim.data.actuator_force[:].clone()
            if args.state_aligned_metrics:env.sim.forward()
            values=metrics(env.sim.data);hard=torch.maximum(low_t-qnext,qnext-high_t).clamp(min=0).amax(1)
            physical=torch.stack([hard,(abs(vnext)/lim_t).amax(1),(abs(tau)/eff_t).amax(1)],dim=1)
            maxima[:]=torch.where(active[:,None],torch.maximum(maxima,physical),maxima)
            heights=values[:,[7,8,9]];minimum[:]=torch.where(active[:,None],torch.minimum(minimum,heights),minimum)
            state_trace.append(torch.cat([env.sim.data.qpos[:],env.sim.data.qvel[:]],dim=1).cpu().numpy().copy())
            metric_trace.append(values.cpu().numpy().copy());force_trace.append(tau.cpu().numpy().copy());control_trace.append(target.cpu().numpy().copy())
            if clock['i']%4==3:
                positive=(values[:,7]>.02)&(values[:,8]>.02)&(values[:,9]>.005)&(values[:,1]>0)
                run[:]=torch.where(positive,run+1,torch.zeros_like(run))
                new=(run>=3)&~completed;end_index[new]=clock['i'];completed[:]|=new;active[:]&=~new
            clock['i']+=1;return result
        env.sim.step=checked_step;env.termination_manager.compute=lambda:torch.zeros(n,dtype=torch.bool,device=args.device)
        duration_lower=.025;duration_upper=.4
        lower=np.r_[duration_lower,np.full(6,-1.),-.4,1.6,-1.,-.3,np.full(3,-.3)]
        upper=np.r_[duration_upper,np.ones(6),1.8,2.35,1.2,.4,np.full(3,.3)]
        rng=np.random.default_rng(args.seed);mean=np.array([.1,.3,-1.,.6,.3,-1.,.6,.8,2.2,.6,.1,0,0,0])
        std=np.array([.05,.7,.3,.7,.7,.3,.7,.5,.2,.5,.2,.08,.08,.08]);best=None;history=[];started=time.monotonic()
        if args.bang_bang:
            lower=np.r_[lower,.15];upper=np.r_[upper,.85]
            mean=np.array([.18,.8,-1.,-.35,-.8,.3,1.,1.,2.2,.6,.1,0,0,0,.6])
            std=np.array([.06,.3,.1,.3,.3,.4,.3,.4,.1,.4,.1,.05,.05,.05,.15])
        if template_t is not None:
            lower=np.r_[.7,np.full(6,-.5),-.6,-.4,-.6,-.25,np.full(3,-.2)]
            upper=np.r_[1.5,np.full(6,.5),.6,.4,.6,.25,np.full(3,.2)]
            upper[7]=args.swing_hip_span
            mean=np.r_[1.,np.zeros(13)];std=np.r_[.15,np.full(6,.15),.2,.15,.2,.1,np.full(3,.05)]*args.residual_std/.15
            std[7]=max(std[7],args.swing_hip_span/3)
        if args.warmstart:mean=np.asarray(json.loads(args.warmstart.read_text())['parameters'])
        def evaluate(parameters):
            state_trace.clear();metric_trace.clear();force_trace.clear();control_trace.clear();clock.update(i=0,parameters=tensor(parameters))
            active[:]=True;completed[:]=False;run[:]=0;end_index[:]=-1;maxima[:]=0;minimum[:]=float('inf')
            env.reset()
            io.write(torch.arange(n,device=args.device),initial)
            torch.testing.assert_close(io.read()[:,:m.nq],tensor(np.tile(q0,(n,1))),rtol=0,atol=2e-5)
            for _ in range(round(args.seconds/env.step_dt)):env.step(torch.zeros((n,len(names)),device=args.device))
            states=np.asarray(state_trace);measure=np.asarray(metric_trace);forces_array=np.asarray(force_trace);ends=end_index.cpu().numpy()
            reports=[]
            for i in range(n):
                if ends[i]>=0:end=int(ends[i])
                else:
                    eligible=np.flatnonzero((measure[:,i,9]>.005)&(measure[:,i,0]>.55))
                    end=int(eligible[np.argmax(measure[eligible,i,1])]) if len(eligible) else 0
                rows=measure[3:end+1:4,i]
                if not len(rows):rows=measure[:end+1,i]
                first_air=len(rows)-3 if ends[i]>=0 else None
                window=rows[max(0,first_air-8):first_air] if first_air is not None else rows[:0]
                single=(len(window)>=4 and (window[:,8]>.04).all() and ((window[:,7]<.01)&(window[:,8]>.04)).sum()>=4)
                if single:
                    support=np.flatnonzero(np.min(window[:,7:9],axis=1)<.01)
                    single=bool(len(support) and window[support[-1],7]<.01 and window[support[-1],8]>.04)
                event=rows[-1] if ends[i]>=0 else measure[end,i]
                prefix=states[:end+1,i];prefix_metrics=measure[:end+1,i]
                hard=float(np.maximum(np.maximum(lo-prefix[:,7:m.nq],prefix[:,7:m.nq]-hi),0).max())
                speed=float((abs(prefix[:,m.nq+6:])/limits).max());motor=float((abs(forces_array[:end+1,i])/effort).max())
                left,right_height,nonfoot=map(float,prefix_metrics[:,[7,8,9]].min(0));penetration=max(-left,-right_height,0.)
                passed=bool(ends[i]>=0 and single and event[1]>1.5 and event[3]>20 and hard<1e-4 and speed<=1 and motor<=1.00001 and nonfoot>.005 and penetration<.005)
                cost=4*(float(event[1])-2.4)**2+float(np.sum((event[2:5]-[0,28,0])**2))/64
                cost+=1000*hard+10000*hard**2+1000*max(penetration-.005,0)+1000*max(.005-nonfoot,0)+20*max(speed-1,0)**2
                cost+=(0 if ends[i]>=0 else 30)+(0 if single else 30)
                cost+=60*float(np.maximum(.025-event[7:9],0).sum())
                reports.append({'cost':cost,'parameters':parameters[i].tolist(),'prefix_end_index':end,'takeoff_candidate_passed':passed,
                    'ascending_air':bool(ends[i]>=0),'objective_event_type':'ascending_air' if ends[i]>=0 else 'grounded_peak_proxy',
                    'single_left_support_at_launch':bool(single),'event_time_s':(end+1)*float(m.opt.timestep),'event_vz':float(event[1]),'event_momentum':event[2:5].tolist(),
                    'hard_violation':hard,'speed_ratio':speed,'effort_ratio':motor,'minimum_left_height':left,'minimum_right_height':right_height,'minimum_nonfoot_height':nonfoot,
                    'physical_candidate_passed':False,'approved_for_training':False})
            return reports,(states,measure,forces_array,np.asarray(control_trace))
        for iteration in range(args.iterations+1):
            parameters=np.clip(rng.normal(mean,std,(n,len(mean))),lower,upper);parameters[0]=np.clip(mean,lower,upper)
            explore=min(n-1,max(0,int(n*args.exploration_fraction)))
            if iteration and explore:parameters[-explore:]=rng.uniform(lower,upper,(explore,len(mean)))
            if iteration==0 and not args.warmstart and template_t is None:
                if args.bang_bang:
                    grid=list(itertools.product([.04,.06,.08,.10,.12,.14],[.03,.05,.07,.09],[.4,.7,1.],[-.4,-.7,-1.],[0.,.6]))
                    for i,(first,second,hip0,hip1,knee1) in enumerate(grid[:n]):
                        parameters[i]=mean;parameters[i,0]=first+second;parameters[i,1]=hip0;parameters[i,4]=hip1;parameters[i,5]=knee1;parameters[i,14]=first/(first+second)
                else:
                    durations=[.08,.14,.20,.26,.32] if q0[7+names.index('left_knee_joint')]>1 else [.04,.07,.10,.13,.16]
                    grid=list(itertools.product(durations,[-1.,-.5,0,.5,1.],[-.6,-.8,-1.],[-1.,-.5,0,.5,1.]))
                    for i,(duration,hip,knee,ankle) in enumerate(grid[:n]):parameters[i]=mean;parameters[i,:7]=[duration,hip,knee,ankle,hip,knee,ankle]
            reports,trace=evaluate(parameters);order=sorted(range(n),key=lambda i:(not reports[i]['takeoff_candidate_passed'],reports[i]['cost']));index=order[0]
            candidate=reports[index]
            if best is None or (not candidate['takeoff_candidate_passed'],candidate['cost'])<(not best['takeoff_candidate_passed'],best['cost']):
                best=candidate;origin=env.scene.env_origins[index].cpu().numpy().copy()
                best['environment_index']=index;best['environment_origin']=origin.tolist()
                canonical=trace[0][:,index].copy();canonical[:,:3]-=origin
                np.savez(args.out/'best-trajectory.npz',state=canonical,state_world=trace[0][:,index],env_origin=origin,
                    state_space=np.array('origin_free'),metrics=trace[1][:,index],actuator_force=trace[2][:,index],servo_target=trace[3][:,index],fps=np.array([200.]))
                (args.out/'best.json').write_text(json.dumps(best,indent=2))
            history.append({'iteration':iteration,'best_cost':best['cost'],'passes':sum(x['takeoff_candidate_passed'] for x in reports),'elapsed_s':time.monotonic()-started})
            if iteration==0:
                (args.out/'baseline-reports.json').write_text(json.dumps(reports,indent=2))
                if template_state is not None:
                    shared=min(len(template_state),len(trace[0]));canonical=trace[0][:shared,0].copy()
                    canonical[:,:3]-=env.scene.env_origins[0].cpu().numpy()
                    difference=canonical-template_state[:shared]
                    (args.out/'template-replay.json').write_text(json.dumps({'shared_steps':shared,
                        'maximum_qpos_difference':float(abs(difference[:,:m.nq]).max()),
                        'maximum_qvel_difference':float(abs(difference[:,m.nq:]).max())},indent=2))
            print(json.dumps(history[-1]),flush=True)
            if best['takeoff_candidate_passed'] or time.monotonic()-started>=args.max_seconds:break
            elite=parameters[order[:max(4,n//16)]];mean=.3*mean+.7*elite.mean(0);std=np.maximum(std*.05,.8*elite.std(0)+.2*std);mean=np.asarray(best['parameters'])
        (args.out/'completed.json').write_text(json.dumps({'best':best,'history':history,'elapsed_s':time.monotonic()-started,
            'not_a_trained_policy':True,'physical_parameters_unchanged':True,'state_edits_after_start':False,
            'resets_between_independent_rollouts':True,'approved_for_training':False},indent=2))
        print(json.dumps({'best':best,'elapsed_s':time.monotonic()-started}),flush=True)
    finally:env.close()


if __name__=='__main__':main()
