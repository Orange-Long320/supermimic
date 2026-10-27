"""Design a motion in the project's unmodified CPU/GPU physics using residual commands.

An immutable learned controller supplies feedback. Bounded, timed residuals and
preparation retiming are searched without changing masses, actuators or gravity.
Old-reference failure resets are suppressed only in this design process. Success
requires independent physical flip/landing gates, not those suppressed checks.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial import ConvexHull


def support_metrics(rows,first_air):
    """Require the final push, not merely an early pose, to use the left leg."""
    single=[i for i,r in enumerate(rows) if r["feet"]["left"]<.01 and r["feet"]["right"]>.04]
    window=list(range(max(0,first_air-8),first_air)) if first_air is not None else []
    last=next((i for i in reversed(window) if min(rows[i]["feet"].values())<.01),None)
    valid=(len(window)>=4 and last in single and sum(i in single for i in window)>=4
           and all(rows[i]["feet"]["right"]>.04 for i in window))
    prelaunch=rows[:first_air] if first_air is not None else rows
    deficit=float(np.mean([max(.04-r["feet"]["right"],0) for r in (prelaunch or rows[:1])]))
    return single,bool(valid),deficit


from supermimic.data.entry_state import entry_velocity_state


def compare_rollout_positions(first,second):
    """Compare the shared timeline without hiding an early-stop mismatch."""
    if first.ndim!=3 or second.ndim!=3 or first.shape[1:]!=second.shape[1:]:
        raise ValueError('Position batches must have matching environment and joint dimensions')
    common=min(len(first),len(second))
    if not common:raise ValueError('Empty physical rollout')
    per_step=np.max(np.abs(first[:common]-second[:common]),axis=(1,2))
    return float(per_step.max()),per_step,len(first)==len(second)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt",type=Path,required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--num-envs",type=int,default=4)
    p.add_argument("--population",type=int,default=24)
    p.add_argument("--iterations",type=int,default=30)
    p.add_argument("--max-seconds",type=float,default=1800)
    p.add_argument("--seconds",type=float,default=2.)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--phase",choices=("takeoff","full"),default="full")
    p.add_argument("--warmstart",type=Path,help="Parameter vector, or an (N,D) grid for zero-iteration timing ablations")
    p.add_argument("--initial-command-margin",type=float,default=0.,help="Move the first position command inward for joints starting at a soft boundary; no state or actuator edits")
    p.add_argument("--initial-knot",action="store_true",help="Search the initial residual command as well as the three preparation knots")
    p.add_argument("--tail-of",type=Path,help="Freeze a takeoff solution and search aerial/landing residuals instead")
    p.add_argument("--joint-phases",action="store_true",help="Use tail-of only as initialization; jointly search preparation and aerial commands")
    p.add_argument("--device",default="cpu")
    p.add_argument("--takeoff-frame",type=int)
    p.add_argument("--landing-frame",type=int)
    p.add_argument("--target-vz",type=float,default=2.4)
    p.add_argument("--target-hy",type=float,default=25.)
    p.add_argument("--entry-vx",type=float,help="Initial COM world-x speed; preserve centroidal H and zero planted-foot velocity")
    p.add_argument("--joint-stop-margin",type=float,help="Diagnostic predictive servo guard inside mechanical joint ranges; no state clipping")
    p.add_argument("--joint-stop-horizon",type=float,default=.1)
    p.add_argument("--baseline-air-scales",help="Comma-separated aerial timing ablation; requires --tail-of and zero search iterations")
    p.add_argument("--bounded-target-joints",help="Diagnostic servo-target bounds for named joints; does not clamp simulated state")
    p.add_argument("--bounded-target-margin",type=float,default=.04)
    p.add_argument("--record-physics",action="store_true",help="Save every 5 ms qpos/qvel/actuator force for independent audits")
    p.add_argument("--hard-penalty-linear",type=float,default=0.,help="Additional exact-penalty term to avoid trading small hard-limit violations for score")
    p.add_argument("--local-refinement",action="store_true",help="Refine a warm start with smaller steps, top 1/16 elites and a best-anchored mean")
    p.add_argument("--residual-sigma",type=float,help="Initial search deviation in normalized motor-effort fractions")
    p.add_argument("--clock-sigma",type=float,help="Initial search deviation for the preparation duration or air scale")
    p.add_argument("--search-groups",help="Comma-separated first joint names of residual groups to optimize; retain all other warm-start coefficients")
    p.add_argument("--floor-penalty-linear",type=float,default=0.,help="Penalize nonfoot clearance below 5 mm and foot compression beyond 5 mm")
    p.add_argument("--high-frequency-geometry",action="store_true",help="Audit post-integration foot/body geometry at every 5 ms physics step")
    p.add_argument("--air-rotation-weight",type=float,default=0.,help="Weight the rotation deficit in the first continuous airborne segment")
    p.add_argument("--air-rotation-target",type=float,default=3.8)
    p.add_argument("--landing-window-seconds",type=float,default=.2)
    p.add_argument("--checkpoint-grid",type=Path,nargs='+',help="Evaluate immutable checkpoints in the same environment; zero iterations only")
    p.add_argument("--landing-ckpt",type=Path,help="Diagnostic phase-specific frozen landing controller")
    p.add_argument("--landing-switch-times",type=float,nargs='+',default=[1.2])
    p.add_argument("--landing-policy-blend",type=float,default=.12)
    p.add_argument("--landing-guide-delay",type=float,help="Override absolute switch times: take over this many seconds after the retimed guide landing")
    p.add_argument("--consistent-retiming",action="store_true",help="Resample future poses and velocities with one continuous physical-time clock")
    p.add_argument("--physics-command-projection",action="store_true",help="Experimental 5 ms acceleration-based motor-command projection")
    p.add_argument("--projection-frequency",type=float,default=35.)
    p.add_argument("--projection-margin",type=float,default=.04)
    args=p.parse_args()
    if args.air_rotation_weight<0 or args.air_rotation_target<=0 or not 0<args.landing_window_seconds<=args.seconds:
        p.error('Invalid airborne-rotation objective or landing window')
    if any(x is not None and x<=0 for x in [args.residual_sigma,args.clock_sigma]):p.error('Search deviations must be positive')
    if args.joint_phases and (not args.tail_of or args.phase!='full' or args.baseline_air_scales):p.error('Joint phases require full phase, a preparation seed, and no air-scale-only grid')
    if args.landing_guide_delay is not None and (not args.landing_ckpt or args.landing_guide_delay<0 or len(args.landing_switch_times)!=1):p.error('Guide-relative landing needs one landing policy and a nonnegative delay')
    assert args.population%args.num_envs==0
    if args.checkpoint_grid and (args.iterations or any(c.name=='ckpt_latest.pt' for c in args.checkpoint_grid)):
        p.error('Checkpoint comparison needs immutable files and zero search iterations')
    if args.landing_ckpt:
        if args.checkpoint_grid or args.landing_ckpt.name=='ckpt_latest.pt':p.error('Use one immutable landing controller')
        if min(args.landing_switch_times)<=0 or args.landing_policy_blend<0 or args.num_envs%len(args.landing_switch_times):
            p.error('Landing switch grid needs positive times and equal replicas')
        if len(args.landing_switch_times)>1 and args.iterations:p.error('Landing timing grid requires zero iterations')
    air_grid=None
    if args.baseline_air_scales:
        air_grid=np.array([float(s) for s in args.baseline_air_scales.split(',')])
        if not args.tail_of or args.iterations or np.any(air_grid<=0) or args.num_envs%len(air_grid):
            p.error('Aerial grid needs a frozen preparation, zero iterations and equal replicas per positive scale')
    # Redesigned preparations have different contact frames. Never silently
    # replay them with the old 30/55-frame clock.
    metadata={}
    for filename in ("GUIDE_ONLY.json","contacts.json"):
        path=args.source.parent/filename
        if path.exists():metadata.update(json.loads(path.read_text()))
    takeoff=args.takeoff_frame
    landing=args.landing_frame
    if takeoff is None:takeoff=metadata.get("takeoff_reference_frame",metadata.get("takeoff_frame"))
    if landing is None:landing=metadata.get("landing_reference_frame",metadata.get("landing_frame"))
    if takeoff is None or landing is None:
        p.error("Contact timing is required: supply sibling contact metadata or both contact-frame arguments")
    with np.load(args.source) as source:
        fps=float(source["fps"].reshape(-1)[0])
        frames=len(source["joint_pos"])
    if abs(fps-50)>1e-6 or not 0<takeoff<landing<frames:
        p.error("Expected a 50 Hz guide with valid ordered takeoff and landing frames")
    args.takeoff_frame=int(takeoff);args.landing_frame=int(landing)
    reference_preparation=takeoff/fps
    reference_flight=(landing-takeoff)/fps
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1)
    from supermimic.robots.dimensions import PROFILE
    from supermimic.envs import SuperMimicEnvCfgArgs,make_env
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    device=torch.device(args.device)
    policy,cfg=load_policy_from_checkpoint(args.ckpt,str(device))
    landing_policy=None
    if args.landing_ckpt:
        landing_policy,landing_cfg=load_policy_from_checkpoint(args.landing_ckpt,str(device))
        if landing_cfg["env"].get("anchor_obs",False)!=cfg["env"].get("anchor_obs",False):
            raise ValueError('Landing controller observation settings differ')
    landing_times=torch.as_tensor(np.resize(args.landing_switch_times,args.num_envs),dtype=torch.float32,device=device)
    env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(args.source)],num_envs=args.num_envs,
        action_noise_std=0.,obs_noise=False,reset_noise=False,domain_randomization=False,
        sampling_mode="start",record_transitions=False,episode_length_s=1e5,
        anchor_obs=cfg["env"].get("anchor_obs",False)),str(device))
    cmd=env.command_manager.get_term("motion")
    original_compute=env.termination_manager.compute
    env.termination_manager.compute=lambda:torch.zeros_like(original_compute())
    original_update=cmd._update_command
    clock={"step":0,"preparation":torch.full((args.num_envs,),reference_preparation,device=device),"initial_frame":0,
           "air_scale":torch.ones(args.num_envs,device=device)}
    if args.consistent_retiming:
        from supermimic.data.reference_retiming import reference_clock,retimed_goal_window
        def sample_clock(times):
            return reference_clock(times,clock["preparation"][:,None],clock["air_scale"][:,None],
                reference_preparation,reference_flight,(frames-1)/fps)
        def goal_window(frame,k):
            # Native reset observations anticipate the next physical control
            # tick. Future samples must preserve that lead after retiming.
            times=(clock["step"]+1+torch.arange(k,device=device,dtype=torch.float32))[None,:]*env.step_dt
            reference_times,rates=sample_clock(times)
            return retimed_goal_window(cmd.library,reference_times*fps,rates)
        cmd.library.goal_window=goal_window
    def warped_update(env_ids=None):
        original_update(env_ids)
        if env_ids is None:
            clock["step"]+=1
            t=clock["step"]*.02
            prep=clock["preparation"]
            after=torch.clamp(t-prep,min=0)
            air_duration=reference_flight*clock["air_scale"]
            air_frame=torch.where(after<=air_duration,takeoff+after/clock["air_scale"]*fps,
                                  landing+(after-air_duration)*fps)
            frame=torch.where(t<=prep,t/prep*takeoff,air_frame).round().long()
            if args.consistent_retiming:
                reference_times,_=sample_clock(torch.full((1,1),t+env.step_dt,device=device))
                cmd.frame[:]=torch.minimum((reference_times[:,0]*fps).round().long(),cmd.library.clip_last[0])
            else:
                cmd.frame[:]=torch.minimum(frame+clock["initial_frame"],cmd.library.clip_last[0])
            cmd._refresh_ref()
            cmd.update_relative_body_poses()
    cmd._update_command=warped_update
    m=env.sim.mj_model
    mujoco.mj_saveModel(m,str(args.out/'native-model.mjb'))
    d=mujoco.MjData(m)
    root=int(m.jnt_bodyid[0])
    names=PROFILE["joint_names"]
    assert [m.joint(int(j)).name.rsplit("/",1)[-1] for j in m.actuator_trnid[:,0]]==list(names)
    kp=np.array([PROFILE["stiffness"][n] for n in names])
    kd=np.array([PROFILE["damping"][n] for n in names])
    effort=np.array([PROFILE["effort_limit"][n] for n in names])
    velocity_limit=np.array([PROFILE["velocity_limit"][n] for n in names])
    scale=PROFILE["action_scale"]
    action_scale=np.array([scale[n] for n in names]) if isinstance(scale,dict) else np.full(len(names),scale)
    groups=[[(n,1)] for n in ["left_hip_pitch_joint","left_knee_joint","left_ankle_pitch_joint",
        "right_hip_pitch_joint","right_knee_joint","right_ankle_pitch_joint","waist_pitch_joint","waist_roll_joint"]]
    groups += [[("left_shoulder_pitch_joint",1),("right_shoulder_pitch_joint",1)],
        [("left_elbow_joint",1),("right_elbow_joint",1)],
        [("left_hip_roll_joint",1),("right_hip_roll_joint",1)],
        [("left_ankle_roll_joint",1)],[("right_ankle_roll_joint",1)]]
    mapping=np.zeros((len(groups),len(names)))
    for k,group in enumerate(groups):
        for name,sign in group:
            j=names.index(name)
            mapping[k,j]=sign*effort[j]/kp[j]/action_scale[j]
    mapping=torch.as_tensor(mapping,dtype=torch.float32,device=device)
    centre=m.jnt_range[1:].mean(axis=1)
    half=.45*np.diff(m.jnt_range[1:],axis=1).ravel()
    lower,upper=centre-half,centre+half
    hard_lower,hard_upper=m.jnt_range[1:].T
    bounded_indices=[names.index(n) for n in args.bounded_target_joints.split(',')] if args.bounded_target_joints else []
    from supermimic.data.design_metrics import BatchedDesignMetrics
    metrics=BatchedDesignMetrics(m,device)
    origins=env.scene.env_origins.cpu().numpy()
    rng=np.random.default_rng(args.seed)
    base_dimension=1+3*len(groups)
    fixed_preparation=np.load(args.tail_of)["parameters"].copy() if args.tail_of else None
    if args.tail_of:
        assert args.phase=="full" and fixed_preparation.shape in [(base_dimension,),(base_dimension+len(groups),)]
    preparation_dimension=len(fixed_preparation) if args.tail_of else base_dimension+(len(groups) if args.initial_knot else 0)
    dimension=preparation_dimension+base_dimension if args.joint_phases else base_dimension if args.tail_of else preparation_dimension
    clock_indices=[0,preparation_dimension] if args.joint_phases else [0]
    blocks=[(1,3)]
    if (not args.tail_of or args.joint_phases) and preparation_dimension>base_dimension:blocks.append((base_dimension,1))
    if args.joint_phases:blocks.append((preparation_dimension+1,3))
    search_mask=np.ones(dimension,dtype=bool)
    if args.search_groups:
        labels=[g[0][0] for g in groups];selected=args.search_groups.split(',')
        if not selected or any(n not in labels for n in selected):p.error('Unknown residual group; use first joint names shown in configuration groups')
        search_mask[:]=False;search_mask[clock_indices]=True
        for name in selected:
            group=labels.index(name)
            for start_column,knots in blocks:
                for knot in range(knots):search_mask[start_column+knot*len(groups)+group]=True
    mean=np.zeros(dimension);mean[0]=1. if args.tail_of else reference_preparation
    if args.joint_phases:mean[:preparation_dimension]=fixed_preparation;mean[preparation_dimension]=1.
    parameter_grid=None
    if args.warmstart:
        initial=np.load(args.warmstart)["parameters"]
        if args.joint_phases and initial.shape[-1]==base_dimension:
            prefix=np.broadcast_to(fixed_preparation,(*initial.shape[:-1],preparation_dimension))
            initial=np.concatenate([prefix,initial],axis=-1)
        elif not args.tail_of and initial.shape[-1]==base_dimension and dimension>base_dimension:
            initial=np.pad(initial,[(0,0)]*(initial.ndim-1)+[(0,dimension-base_dimension)])
        if initial.ndim==2:
            if args.iterations or initial.shape[1]!=dimension or args.num_envs%len(initial):
                p.error('Parameter grid requires zero iterations, matching dimensions and equal replicas')
            parameter_grid=initial;mean[:]=initial[0]
        else:
            if initial.shape!=(dimension,):p.error('Warm-start parameter dimensions do not match the selected phase layout')
            mean[:]=initial
    clock_sigma=args.clock_sigma if args.clock_sigma is not None else (.04 if args.local_refinement else .1)
    residual_sigma=args.residual_sigma if args.residual_sigma is not None else (.08 if args.local_refinement else .25)
    sigma=np.full(dimension,residual_sigma);sigma[clock_indices]=clock_sigma
    covariance=np.diag(sigma**2)
    lo=np.r_[.6 if args.tail_of else .45,np.full(dimension-1,-1.5)]
    hi=np.r_[1.4 if args.tail_of else 1.2,np.full(dimension-1,1.5)]
    if args.joint_phases:lo[0]=.45;hi[0]=1.2;lo[preparation_dimension]=.6;hi[preparation_dimension]=1.4
    covariance_floor=np.full(dimension,.01);covariance_floor[clock_indices]=.003
    start=time.monotonic()
    evaluations=0
    best=None
    physics_limits=np.zeros((args.num_envs,4))
    physics_geometry=np.zeros((args.num_envs,3));physics_geometry[:,2]=1e3
    from supermimic.data.design_limits import joint_limit_statistics
    limit_constants=[torch.as_tensor(x,dtype=torch.float64,device=device) for x in
                     [lower,upper,hard_lower,hard_upper,velocity_limit,effort]]
    device_limits=torch.zeros((args.num_envs,4),dtype=torch.float64,device=device)
    device_geometry=torch.zeros((args.num_envs,3),dtype=torch.float64,device=device);device_geometry[:,2]=1e3
    device_mask=torch.ones(args.num_envs,dtype=torch.bool,device=device)
    physics_mask=np.ones(args.num_envs,dtype=bool)
    original_step=env.sim.step
    active_physics={"record":False}
    physics_trace=[]
    projection={"inertia":None,"interventions":np.zeros(len(names),dtype=np.int64),
                "infeasible":np.zeros(len(names),dtype=np.int64)}
    if args.physics_command_projection:
        from supermimic.policy.joint_projection import project_joint_commands
        assert np.allclose(m.actuator_gainprm[:,0],kp)
        assert np.allclose(m.actuator_biasprm[:,1],-kp) and np.allclose(m.actuator_biasprm[:,2],-kd)
        tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device=device)
        projection_constants=[tensor(x) for x in [kp,kd,effort,hard_lower,hard_upper]]
    def checked_step(*step_args,**step_kwargs):
        if args.physics_command_projection and active_physics["record"]:
            kp_t,kd_t,effort_t,low_t,high_t=projection_constants
            control,changed,infeasible=project_joint_commands(env.sim.data.ctrl[:],
                env.sim.data.qpos[:,7:],env.sim.data.qvel[:,6:],env.sim.data.qacc[:,6:],
                env.sim.data.actuator_force[:],kp_t,kd_t,effort_t,projection["inertia"],low_t,high_t,
                args.projection_margin,args.projection_frequency)
            env.sim.data.ctrl[:]=control
            projection["interventions"]+=changed.sum(0).cpu().numpy()
            projection["infeasible"]+=infeasible.sum(0).cpu().numpy()
        result=original_step(*step_args,**step_kwargs)
        if active_physics["record"]:
            q=env.sim.data.qpos;v=env.sim.data.qvel;f=env.sim.data.actuator_force
            if args.record_physics:physics_trace.append(tuple(x.cpu().numpy().copy() for x in [q,v,f]))
            limits=joint_limit_statistics(q,v,f,*limit_constants)
            device_limits.copy_(torch.where(device_mask[:,None],torch.maximum(device_limits,limits),device_limits))
            if args.high_frequency_geometry:
                env.sim.forward()
                measured=metrics(env.sim.data)
                bounds=torch.stack([(-measured[:,9]).clamp(min=0),(-measured[:,7:9].amin(dim=1)).clamp(min=0),measured[:,9]],dim=1).to(torch.float64)
                device_geometry[:,:2].copy_(torch.where(device_mask[:,None],torch.maximum(device_geometry[:,:2],bounds[:,:2]),device_geometry[:,:2]))
                device_geometry[:,2].copy_(torch.where(device_mask,torch.minimum(device_geometry[:,2],bounds[:,2]),device_geometry[:,2]))
        return result
    env.sim.step=checked_step
    (args.out/"configuration.json").write_text(json.dumps({
        "arguments":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        "checkpoint_sha256":hashlib.sha256(args.ckpt.read_bytes()).hexdigest(),
        "landing_checkpoint_sha256":hashlib.sha256(args.landing_ckpt.read_bytes()).hexdigest() if args.landing_ckpt else None,
        "source_sha256":hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "physics":f"Unmodified project make_env on {device}; only local design-process reset requests suppressed",
        "joint_acceptance":"Actual mechanical joint ranges at each physics step; soft limits retained as a separate regularizer",
        "metric_state_alignment":"Post-integration derived fields refreshed before each control-frame geometry measurement",
        "geometry_acceptance_frequency_hz":1/float(m.opt.timestep) if args.high_frequency_geometry else 50,
        "limit_statistics":"Float64 device-side reductions; transferred once per rollout, or trace samples when explicitly recording",
        "hard_joint_ranges_rad":m.jnt_range[1:].tolist(),
        "takeoff_objective":"Continuous ascending flight; grounded extension is a fallback proxy with an explicit flight deficit penalty",
        "groups":groups,"residual_units":"fraction of actuator effort, mapped to existing position actions",
        "fixed_preparation_sha256":hashlib.sha256(args.tail_of.read_bytes()).hexdigest() if args.tail_of else None,
        "optimization_variables":"preparation, aerial timing and all residual phases" if args.joint_phases else "aerial guide time scale and air/landing residuals" if args.tail_of else "preparation duration and support/swing residuals",
        "preparation_frozen":bool(args.tail_of and not args.joint_phases),
        "preparation_parameter_dimension":preparation_dimension,"clock_parameter_indices":clock_indices,
        "initial_knot_searched":bool((not args.tail_of or args.joint_phases) and preparation_dimension>base_dimension),
        "fixed_preparation_has_initial_knot":bool(fixed_preparation is not None and len(fixed_preparation)>base_dimension),
        "searched_parameter_indices":np.flatnonzero(search_mask).tolist(),
        "not_a_trained_policy":True},indent=2,default=str))

    @torch.no_grad()
    def rollout(parameters):
        nonlocal evaluations
        evaluations+=len(parameters)
        clock["step"]=0
        physics_limits[:]=0
        physics_geometry[:,:2]=0;physics_geometry[:,2]=1e3
        device_limits.zero_();device_geometry[:,:2].zero_();device_geometry[:,2]=1e3;device_mask.fill_(True)
        physics_mask[:]=True
        active_physics["record"]=False
        physics_trace.clear()
        clock["initial_frame"]=0
        prep_parameters=parameters[:,:preparation_dimension] if args.joint_phases else np.tile(fixed_preparation,(args.num_envs,1)) if args.tail_of else parameters
        air_parameters=parameters[:,preparation_dimension:] if args.joint_phases else parameters if args.tail_of else None
        clock["preparation"]=torch.tensor(prep_parameters[:,0],dtype=torch.float32,device=device)
        clock["air_scale"]=torch.tensor(air_parameters[:,0] if air_parameters is not None else np.ones(args.num_envs),dtype=torch.float32,device=device)
        if args.landing_guide_delay is not None:landing_times.copy_(clock['preparation']+reference_flight*clock['air_scale']+args.landing_guide_delay)
        cmd.forced_frame[:]=0
        obs,_=env.reset()
        ids=torch.arange(args.num_envs,device=device)
        if args.entry_vx is not None:
            from supermimic.envs.state_io import StateIO
            io=StateIO(env);state=io.read()
            velocity,entry=entry_velocity_state(m,state[0,:m.nq].cpu().numpy(),
                state[0,m.nq:].cpu().numpy(),args.entry_vx,velocity_limit)
            state[:,m.nq:]=torch.as_tensor(velocity,dtype=state.dtype,device=device)
            io.write(ids,state)
            (args.out/"entry-initial-condition.json").write_text(json.dumps(entry,indent=2))
        if args.consistent_retiming:
            reference_times,_=sample_clock(torch.full((1,1),env.step_dt,device=device))
            cmd.frame[:]=torch.minimum((reference_times[:,0]*fps).round().long(),cmd.library.clip_last[0])
            cmd._refresh_ref()
        if args.entry_vx is not None or args.consistent_retiming:
            cmd.update_relative_body_poses();env.sim.sense()
            env.observation_manager.reset(ids)
            obs=env.observation_manager.compute(update_history=True,env_ids=ids)
            env.obs_buf=obs
        clock["initial_frame"]=int(cmd.frame[0])
        clock["step"]=0
        if args.physics_command_projection:
            d.qpos[:]=env.sim.data.qpos[0].cpu().numpy();mujoco.mj_forward(m,d)
            mass=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,mass)
            projection["inertia"]=tensor(1/np.diag(np.linalg.inv(mass))[6:])
            projection["interventions"][:]=0;projection["infeasible"][:]=0
        active_physics["record"]=True
        offsets=torch.tensor(prep_parameters[:,1:base_dimension].reshape(args.num_envs,3,len(groups)),dtype=torch.float32,device=device)
        tail_offsets=torch.tensor(air_parameters[:,1:].reshape(args.num_envs,3,len(groups)),dtype=torch.float32,device=device) if air_parameters is not None else None
        initial_action=policy(obs["policy"],obs["goal"])
        initial_state={"qpos":env.sim.data.qpos.cpu().numpy().copy(),
                       "qvel":env.sim.data.qvel.cpu().numpy().copy(),
                       "first_action":initial_action.cpu().numpy().copy()}
        initial_delta=torch.zeros_like(initial_action)
        if args.initial_command_margin>0:
            q0=env.sim.data.qpos.cpu().numpy()[:,7:]
            nominal=np.array([PROFILE["default_joint_pos"][n] for n in names])+initial_action.cpu().numpy()*action_scale
            corrected=nominal.copy()
            near_lower=q0<lower+.005
            near_upper=q0>upper-.005
            corrected=np.where(near_lower,np.maximum(corrected,lower+args.initial_command_margin),corrected)
            corrected=np.where(near_upper,np.minimum(corrected,upper-args.initial_command_margin),corrected)
            initial_delta=torch.tensor((corrected-nominal)/action_scale,dtype=torch.float32,device=device)
        if prep_parameters.shape[1]>base_dimension:
            initial_delta+=torch.as_tensor(prep_parameters[:,base_dimension:],dtype=torch.float32,device=device)@mapping
        states=[];velocities=[];actions=[]
        records=[[] for _ in range(args.num_envs)]
        maxima=[{"soft":0.,"hard":0.,"speed":0.,"effort":0.,"nonfoot_penetration":0.,
                 "foot_penetration":0.,"minimum_nonfoot_height":1e3} for _ in range(args.num_envs)]
        completed_air=np.zeros(args.num_envs,dtype=bool)
        air_run=np.zeros(args.num_envs,dtype=int)
        for step in range(round(args.seconds*50)):
            progress=torch.clamp(step*.02/clock["preparation"],0,1)
            f=progress*3
            index=f.long().clamp(max=2)
            fraction=f-index
            knots=torch.cat([initial_delta[:,None,:],offsets@mapping],dim=1)
            ids=torch.arange(args.num_envs,device=device)
            residual=knots[ids,index]*(1-fraction[:,None])+knots[ids,index+1]*fraction[:,None]
            residual*=torch.clamp(1-(step*.02-clock["preparation"])/.12,0,1)[:,None]
            if args.tail_of:
                after=torch.clamp(step*.02-clock["preparation"],min=0)
                air_duration=reference_flight*clock["air_scale"]
                tail_f=torch.clamp(after/air_duration,0,1)*3
                tail_index=tail_f.long().clamp(max=2)
                tail_fraction=tail_f-tail_index
                tail_knots=torch.cat([torch.zeros_like(initial_delta[:,None,:]),tail_offsets@mapping],dim=1)
                tail=(tail_knots[ids,tail_index]*(1-tail_fraction[:,None])
                      +tail_knots[ids,tail_index+1]*tail_fraction[:,None])
                tail*=torch.clamp(1-(after-air_duration)/.4,0,1)[:,None]
                residual+=tail
            action=policy(obs["policy"],obs["goal"])+residual
            if landing_policy is not None:
                fraction=(torch.clamp((step*.02-landing_times)/args.landing_policy_blend,0,1)
                    if args.landing_policy_blend else (step*.02>=landing_times).to(torch.float32))
                landing_action=landing_policy(obs["policy"],obs["goal"])+residual
                action=torch.lerp(action,landing_action,fraction[:,None])
            if bounded_indices:
                indices=torch.tensor(bounded_indices,device=device)
                scale_t=torch.as_tensor(action_scale[bounded_indices],dtype=action.dtype,device=device)
                default=torch.as_tensor([PROFILE["default_joint_pos"][names[i]] for i in bounded_indices],dtype=action.dtype,device=device)
                target=default+action[:,indices]*scale_t
                target=torch.clamp(target,
                    min=torch.as_tensor(hard_lower[bounded_indices]+args.bounded_target_margin,dtype=action.dtype,device=device),
                    max=torch.as_tensor(hard_upper[bounded_indices]-args.bounded_target_margin,dtype=action.dtype,device=device))
                action[:,indices]=(target-default)/scale_t
            if args.joint_stop_margin is not None:
                pos=env.sim.data.qpos[:,7:];vel=env.sim.data.qvel[:,6:]
                f=lambda a:torch.as_tensor(a,dtype=pos.dtype,device=device)
                default=f(np.array([PROFILE["default_joint_pos"][n] for n in names]))
                scale_t=f(action_scale);ratio=f(kd/kp)
                target=default+action*scale_t
                predicted=pos+args.joint_stop_horizon*vel
                inner_lower=f(hard_lower+args.joint_stop_margin)
                inner_upper=f(hard_upper-args.joint_stop_margin)
                target=torch.where(predicted<inner_lower,
                    torch.maximum(target,inner_lower-ratio*torch.minimum(vel,torch.zeros_like(vel))),target)
                target=torch.where(predicted>inner_upper,
                    torch.minimum(target,inner_upper-ratio*torch.maximum(vel,torch.zeros_like(vel))),target)
                action=(target-default)/scale_t
            if args.phase=='takeoff':device_mask.copy_(torch.as_tensor(physics_mask,device=device))
            obs,_,_,_,_=env.step(action)
            q=env.sim.data.qpos.cpu().numpy().copy()
            v=env.sim.data.qvel.cpu().numpy().copy()
            force=env.sim.data.actuator_force.cpu().numpy()
            states.append(q);velocities.append(v);actions.append(action.cpu().numpy())
            # MuJoCo caches geometry before its final integration. Refresh
            # derived fields so contact heights and orientation match saved q.
            env.sim.forward()
            measurements=metrics(env.sim.data).cpu().numpy()
            for e in range(args.num_envs):
                if args.phase=="takeoff" and completed_air[e]:continue
                measure=measurements[e]
                feet={"left":float(measure[7]),"right":float(measure[8])}
                nonfoot=float(measure[9])
                maxa=maxima[e]
                maxa["soft"]=max(maxa["soft"],float(np.maximum(np.maximum(lower-q[e,7:],q[e,7:]-upper),0).max()))
                maxa["speed"]=max(maxa["speed"],float(np.max(np.abs(v[e,6:])/velocity_limit)))
                maxa["effort"]=max(maxa["effort"],float(np.max(np.abs(force[e])/effort)))
                maxa["nonfoot_penetration"]=max(maxa["nonfoot_penetration"],max(-nonfoot,0))
                maxa["minimum_nonfoot_height"]=min(maxa["minimum_nonfoot_height"],nonfoot)
                maxa["foot_penetration"]=max(maxa["foot_penetration"],max(-min(feet.values()),0))
                records[e].append({"time":(step+1)*.02,"com_z":float(measure[0]),
                    "vz":float(measure[1]),"momentum":measure[2:5].tolist(),
                    "com_speed":float(measure[5]),"root_angular_speed":float(measure[6]),
                    "feet":feet,"nonfoot_height":nonfoot,"pitch":float(measure[10]),"up_z":float(measure[11])})
                if min(feet.values())>.02 and measure[1]>0. and nonfoot>.005:
                    air_run[e]+=1
                else:air_run[e]=0
                if air_run[e]>=3:
                    completed_air[e]=True
                    if args.phase=="takeoff":physics_mask[e]=False
            if args.phase=="takeoff" and completed_air.all():break
        active_physics["record"]=False
        physics_limits[:]=device_limits.cpu().numpy();physics_geometry[:]=device_geometry.cpu().numpy()
        if args.physics_command_projection:
            (args.out/"command-projection.json").write_text(json.dumps({
                "joint_names":names,"initial_free_body_effective_inertia":projection["inertia"].cpu().tolist(),
                "intervention_counts":projection["interventions"].tolist(),
                "infeasible_acceleration_bound_counts":projection["infeasible"].tolist(),
                "approximation":"Fixed initial inverse-mass diagonal plus previous-step observed acceleration; no safety guarantee",
                "physical_model_unchanged":True},indent=2))
        reports=[]
        for e,rows in enumerate(records):
            pitch=np.unwrap([r["pitch"] for r in rows])
            air=[i for i,r in enumerate(rows) if min(r["feet"].values())>.02 and r["nonfoot_height"]>.005]
            first_air=next((i for i in range(len(rows)-2) if all(j in air for j in range(i,i+3))),None)
            single,single_at_launch,right_deficit=support_metrics(rows,first_air)
            positive_air=[i for i in air if rows[i]["vz"]>0.]
            longest_air=run=0
            for i in range(len(rows)):
                run=run+1 if i in positive_air else 0
                longest_air=max(longest_air,run)
            # A rising COM with both feet still planted is not a takeoff.
            # Prefer the measured ascending flight event and explicitly score
            # continuous air; do not optimize a grounded extension proxy.
            event_rows=[rows[i] for i in positive_air] or [r for r in rows if r["time"]>=.25]
            def event_error(r):
                return ((r["vz"]-args.target_vz)**2*4+np.sum((np.array(r["momentum"])-[0,args.target_hy,0])**2)/64
                    +sum(max(.025-h,0)*60 for h in r["feet"].values()))
            event=min(event_rows,key=event_error)
            event_cost=event_error(event)
            event_cost+=30*(1-min(longest_air,3)/3)**2
            physical=maxima[e]
            # For a completed takeoff, stats stop at that event. Full-clip
            # checks always use every 5 ms physics step, not only video frames.
            for k,key in enumerate(("soft","hard","speed","effort")):
                physical[key]=max(physical[key],float(physics_limits[e,k]))
            if args.high_frequency_geometry:
                physical['nonfoot_penetration']=max(physical['nonfoot_penetration'],float(physics_geometry[e,0]))
                physical['foot_penetration']=max(physical['foot_penetration'],float(physics_geometry[e,1]))
                physical['minimum_nonfoot_height']=min(physical['minimum_nonfoot_height'],float(physics_geometry[e,2]))
            cost=event_cost+physical["hard"]**2*10000+physical["soft"]**2*20+max(physical["speed"]-1,0)**2*20
            cost+=args.hard_penalty_linear*physical["hard"]
            cost+=args.floor_penalty_linear*(max(.005-physical["minimum_nonfoot_height"],0)
                +max(physical["foot_penetration"]-.005,0))
            cost+=physical["nonfoot_penetration"]**2*2000+physical["foot_penetration"]**2*2000
            if args.phase=="full":
                landing_window=rows[-max(1,round(args.landing_window_seconds*50)):]
                cost+=max(5.7-pitch.max()+pitch[0],0)**2*2
                cost+=max(.8-rows[-1]["up_z"],0)**2*10+max(.55-rows[-1]["com_z"],0)**2*50
                cost+=np.mean([r["com_speed"]**2*4+max(.85-r["up_z"],0)**2*10
                    +r["root_angular_speed"]**2*.5
                    +max(min(r["feet"].values())-.01,0)**2*200 for r in landing_window])
            # The free leg must stay up during the push. An early single-leg
            # pose followed by a two-leg jump is not a Webster takeoff.
            cost+=500*right_deficit+(0 if single_at_launch else 30)
            launch_row=rows[min(first_air+2,len(rows)-1)] if first_air is not None else None
            takeoff_passed=(completed_air[e] and single_at_launch and physical["hard"]<1e-4
                and launch_row["vz"]>1.5 and launch_row["momentum"][1]>20
                and physical["speed"]<=1 and physical["effort"]<=1.00001
                and physical["minimum_nonfoot_height"]>.005 and physical["foot_penetration"]<.005)
            landing_window=rows[-max(1,round(args.landing_window_seconds*50)):]
            landed=all(r["up_z"]>.85 and r["com_z"]>.55 and r["com_speed"]<.5 and r["root_angular_speed"]<1
                and min(r["feet"].values())<.01 for r in landing_window)
            air_end=first_air
            if first_air is not None:
                while air_end+1 in air:air_end+=1
            air_rotation=float(pitch[air_end]-pitch[first_air]) if first_air is not None else 0.
            if args.phase=='full':cost+=args.air_rotation_weight*max(args.air_rotation_target-air_rotation,0)**2
            passed=(args.phase=="full" and len(air)>=4 and single_at_launch and pitch.max()-pitch[0]>5.5
                and air_rotation>3.5
                and min(rows[i]["up_z"] for i in air)<-.7 and rows[-1]["up_z"]>.85
                and landed
                and physical["hard"]<1e-4 and physical["speed"]<=1 and physical["effort"]<=1.00001
                and physical["minimum_nonfoot_height"]>.005 and physical["foot_penetration"]<.005)
            reports.append({"cost":float(cost),"maxima":physical,"air_frames":len(air),
                "consecutive_ascending_air_frames":longest_air,"single_left_support_frames":len(single),
                "mean_prelaunch_right_foot_clearance_deficit_m":right_deficit,
                "objective_event":{"time":event["time"],"vz":event["vz"],"momentum":event["momentum"],"feet":event["feet"]},
                "maximum_forward_rotation_rad":float(pitch.max()-pitch[0]),"final":rows[-1],
                "airborne_forward_rotation_rad":air_rotation,
                "landing_window_seconds":args.landing_window_seconds,"landing_window_passed":bool(landed),
                "takeoff_candidate_passed":bool(takeoff_passed),"single_left_support_at_launch":bool(single_at_launch),
                "physical_candidate_passed":bool(passed),"approved_for_training":False})
            if landing_policy is not None:reports[-1]["landing_switch_time_s"]=float(landing_times[e])
        physical_trace=tuple(np.stack([r[k] for r in physics_trace]) for k in range(3)) if physics_trace else None
        return reports,(np.array(states),np.array(velocities),np.array(actions),records,initial_state,physical_trace)

    try:
        # Establish the exact-controller baseline before searching.
        baseline_parameters=np.tile(mean,(args.num_envs,1))
        if parameter_grid is not None:baseline_parameters=np.tile(parameter_grid,(args.num_envs//len(parameter_grid),1))
        if air_grid is not None:baseline_parameters[:,0]=np.resize(air_grid,args.num_envs)
        baseline,payload=rollout(baseline_parameters)
        (args.out/"baseline.json").write_text(json.dumps(baseline[0],indent=2))
        (args.out/"baseline-all.json").write_text(json.dumps(baseline))
        np.savez(args.out/"baseline-trajectory.npz",qpos=payload[0][:,0],qvel=payload[1][:,0],actions=payload[2][:,0],fps=np.array([50.]))
        if payload[5] is not None:
            positions=payload[5][0][:,0].copy();positions[:,:3]-=origins[0]
            np.savez(args.out/"baseline-physics.npz",qpos=positions,qvel=payload[5][1][:,0],
                actuator_force=payload[5][2][:,0],fps=np.array([1/m.opt.timestep]),
                initial_qpos=payload[4]["qpos"][0]-np.r_[origins[0],np.zeros(m.nq-3)],
                initial_qvel=payload[4]["qvel"][0])
        if parameter_grid is not None or air_grid is not None or (landing_policy is not None and len(args.landing_switch_times)>1):
            positions=payload[0].copy();positions[:,:,:3]-=origins[None,:,:]
            np.savez(args.out/"timing-grid-trajectories.npz",qpos=positions,qvel=payload[1],actions=payload[2],
                air_scales=baseline_parameters[:,preparation_dimension] if args.joint_phases else baseline_parameters[:,0] if args.tail_of else np.ones(args.num_envs),
                landing_switch_times=landing_times.cpu().numpy(),parameters=baseline_parameters,fps=np.array([50.]))
        print("BASELINE",json.dumps(baseline[0]),flush=True)
        repeat,repeat_payload=rollout(baseline_parameters)
        (args.out/"reset-repeat-all.json").write_text(json.dumps(repeat))
        error,per_step,equal_length=compare_rollout_positions(payload[0],repeat_payload[0])
        initial_errors={k:float(np.abs(payload[4][k]-repeat_payload[4][k]).max()) for k in payload[4]}
        initial_passed=max(initial_errors.values())<1e-5
        np.savez(args.out/"reset-repeat-trajectory.npz",qpos=repeat_payload[0],qvel=repeat_payload[1])
        (args.out/"reset-reproducibility.json").write_text(json.dumps({"maximum_qpos_error":error,
            "maximum_qpos_error_by_step":per_step.tolist(),
            "rollout_frame_counts":[len(payload[0]),len(repeat_payload[0])],"equal_length":equal_length,
            "initial_state_errors":initial_errors,"initial_state_check_passed":initial_passed,
            "candidate_validation_required":"Three independent successful full physical rollouts before export or training; GPU numerical trajectory differences are not treated as success.",
            "passed":equal_length and error<1e-3,"scope":"Same controller, goals, initial state and reset; common timeline compared, unequal durations explicitly fail reproducibility"},indent=2))
        assert initial_passed, "Initial state or first policy action differs across resets; design search rejected"
        if device.type=="cpu":
            assert equal_length and error<1e-3, "Repeated deterministic CPU resets disagree; design search rejected"
        if args.checkpoint_grid:
            anchor_obs=cfg["env"].get("anchor_obs",False)
            for checkpoint in args.checkpoint_grid:
                policy,candidate_cfg=load_policy_from_checkpoint(checkpoint,str(device))
                if candidate_cfg["env"].get("anchor_obs",False)!=anchor_obs:
                    raise ValueError('Checkpoint observation settings differ; comparison rejected')
                reports,recording=rollout(baseline_parameters)
                folder=args.out/"checkpoint-grid"/checkpoint.stem;folder.mkdir(parents=True,exist_ok=False)
                with checkpoint.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
                (folder/"reports.json").write_text(json.dumps({'checkpoint':str(checkpoint),
                    'checkpoint_sha256':digest,'reports':reports,'approved_for_training':False},indent=2))
                positions=recording[0].copy();positions[:,:,:3]-=origins[None,:,:]
                np.savez(folder/"trajectories.npz",qpos=positions,qvel=recording[1],
                    actions=recording[2],fps=np.array([50.]))
                print('CHECKPOINT',str(checkpoint),json.dumps(reports[0]),flush=True)
        for iteration in range(args.iterations):
            iteration_start=time.monotonic()
            population=np.clip(rng.multivariate_normal(mean,covariance,args.population),lo,hi)
            population[:,~search_mask]=mean[~search_mask]
            population[0]=mean
            if best is not None:population[1]=best[1]
            outcomes=[]
            for a in range(0,args.population,args.num_envs):
                reports,payload=rollout(population[a:a+args.num_envs])
                for e,report in enumerate(reports):
                    parameters=population[a+e].copy()
                    outcomes.append((report["cost"],parameters,report))
                    complete=report["physical_candidate_passed"] or (args.phase=="takeoff" and report["takeoff_candidate_passed"])
                    best_complete=best is not None and (best[2]["physical_candidate_passed"] or (args.phase=="takeoff" and best[2]["takeoff_candidate_passed"]))
                    if best is None or (complete and not best_complete) or (complete==best_complete and report["cost"]<best[0]):
                        best=(report["cost"],parameters,report)
                        canonical_q=payload[0][:len(payload[3][e]),e].copy()
                        canonical_q[:,:3]-=origins[e]
                        temporary=args.out/"best-trajectory.tmp.npz"
                        np.savez(temporary,parameters=parameters,qpos=canonical_q,qvel=payload[1][:len(canonical_q),e],
                            actions=payload[2][:len(canonical_q),e],fps=np.array([50.]))
                        temporary.replace(args.out/"best-trajectory.npz")
                        (args.out/"best-frames.json").write_text(json.dumps(payload[3][e]))
                if time.monotonic()-start>args.max_seconds:break
            outcomes.sort(key=lambda x:x[0])
            elite=np.array([r[1] for r in outcomes[:max(2,len(outcomes)//(16 if args.local_refinement else 4))]])
            mean=(.25*mean+.5*elite.mean(axis=0)+.25*best[1]) if args.local_refinement else .3*mean+.7*elite.mean(axis=0)
            covariance=.7*covariance+.3*np.cov(elite,rowvar=False)+np.diag(covariance_floor**2)
            record={"iteration":iteration,"evaluations":evaluations,"elapsed_s":time.monotonic()-start,**best[2]}
            record["iteration_elapsed_s"]=time.monotonic()-iteration_start
            record["device"]=str(device)
            if device.type=="cuda":
                record["torch_peak_allocated_mib"]=torch.cuda.max_memory_allocated(device)/2**20
                record["torch_peak_reserved_mib"]=torch.cuda.max_memory_reserved(device)/2**20
            (args.out/"latest.json").write_text(json.dumps(record,indent=2))
            with (args.out/"progress.jsonl").open("a") as f:f.write(json.dumps(record)+"\n")
            print(json.dumps(record),flush=True)
            if best[2]["physical_candidate_passed"] or (args.phase=="takeoff" and best[2]["takeoff_candidate_passed"]) or time.monotonic()-start>args.max_seconds:break
        if args.joint_phases and best is not None:
            np.savez(args.out/'best-preparation.npz',parameters=best[1][:preparation_dimension])
            np.savez(args.out/'best-air.npz',parameters=best[1][preparation_dimension:])
        (args.out/"completed.json").write_text(json.dumps({"elapsed_s":time.monotonic()-start,"evaluations":evaluations,
            "physical_candidate_passed":best[2]["physical_candidate_passed"] if best else False,
            "takeoff_candidate_passed":best[2]["takeoff_candidate_passed"] if best else False,"approved_for_training":False},indent=2))
    finally:
        env.close()


if __name__=="__main__":main()
