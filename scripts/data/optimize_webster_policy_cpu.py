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
    p.add_argument("--warmstart",type=Path)
    p.add_argument("--initial-command-margin",type=float,default=0.,help="Move the first position command inward for joints starting at a soft boundary; no state or actuator edits")
    p.add_argument("--tail-of",type=Path,help="Freeze a takeoff solution and search aerial/landing residuals instead")
    p.add_argument("--device",default="cpu")
    args=p.parse_args()
    assert args.population%args.num_envs==0
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    from supermimic.robots.dimensions import PROFILE
    from supermimic.envs import SuperMimicEnvCfgArgs,make_env
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    device=torch.device(args.device)
    policy,cfg=load_policy_from_checkpoint(args.ckpt,str(device))
    env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(args.source)],num_envs=args.num_envs,
        action_noise_std=0.,obs_noise=False,reset_noise=False,domain_randomization=False,
        sampling_mode="start",record_transitions=False,episode_length_s=1e5,
        anchor_obs=cfg["env"].get("anchor_obs",False)),str(device))
    cmd=env.command_manager.get_term("motion")
    original_compute=env.termination_manager.compute
    env.termination_manager.compute=lambda:torch.zeros_like(original_compute())
    original_update=cmd._update_command
    clock={"step":0,"preparation":torch.full((args.num_envs,),.6,device=device),"initial_frame":0,
           "air_scale":torch.ones(args.num_envs,device=device)}
    def warped_update(env_ids=None):
        original_update(env_ids)
        if env_ids is None:
            clock["step"]+=1
            t=clock["step"]*.02
            prep=clock["preparation"]
            after=torch.clamp(t-prep,min=0)
            air_duration=.5*clock["air_scale"]
            air_frame=torch.where(after<=air_duration,30+after/clock["air_scale"]*50,
                                  55+(after-air_duration)*50)
            frame=torch.where(t<=prep,t/prep*30,air_frame).round().long()
            cmd.frame[:]=torch.minimum(frame+clock["initial_frame"],cmd.library.clip_last[0])
            cmd._refresh_ref()
            cmd.update_relative_body_poses()
    cmd._update_command=warped_update
    m=env.sim.mj_model
    d=mujoco.MjData(m)
    root=int(m.jnt_bodyid[0])
    names=PROFILE["joint_names"]
    assert [m.joint(int(j)).name.rsplit("/",1)[-1] for j in m.actuator_trnid[:,0]]==list(names)
    kp=np.array([PROFILE["stiffness"][n] for n in names])
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
    from supermimic.data.design_metrics import BatchedDesignMetrics
    metrics=BatchedDesignMetrics(m,device)
    origins=env.scene.env_origins.cpu().numpy()
    rng=np.random.default_rng(args.seed)
    dimension=1+3*len(groups)
    fixed_preparation=np.load(args.tail_of)["parameters"].copy() if args.tail_of else None
    if args.tail_of:
        assert args.phase=="full" and fixed_preparation.shape==(dimension,)
    mean=np.zeros(dimension);mean[0]=1. if args.tail_of else .6
    if args.warmstart:
        mean[:]=np.load(args.warmstart)["parameters"]
    covariance=np.diag(np.r_[.1,np.full(dimension-1,.25)]**2)
    lo=np.r_[.6 if args.tail_of else .45,np.full(dimension-1,-1.5)]
    hi=np.r_[1.4 if args.tail_of else 1.2,np.full(dimension-1,1.5)]
    start=time.monotonic()
    evaluations=0
    best=None
    physics_limits=np.zeros((args.num_envs,3))
    physics_mask=np.ones(args.num_envs,dtype=bool)
    original_step=env.sim.step
    active_physics={"record":False}
    def checked_step(*step_args,**step_kwargs):
        result=original_step(*step_args,**step_kwargs)
        if active_physics["record"]:
            q=env.sim.data.qpos.cpu().numpy()
            v=env.sim.data.qvel.cpu().numpy()
            f=env.sim.data.actuator_force.cpu().numpy()
            limits=np.c_[np.maximum(np.maximum(lower-q[:,7:],q[:,7:]-upper),0).max(axis=1),
                (np.abs(v[:,6:])/velocity_limit).max(axis=1),(np.abs(f)/effort).max(axis=1)]
            physics_limits[physics_mask]=np.maximum(physics_limits[physics_mask],limits[physics_mask])
        return result
    env.sim.step=checked_step
    (args.out/"configuration.json").write_text(json.dumps({
        "arguments":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        "checkpoint_sha256":hashlib.sha256(args.ckpt.read_bytes()).hexdigest(),
        "source_sha256":hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "physics":f"Unmodified project make_env on {device}; only local design-process reset requests suppressed",
        "groups":groups,"residual_units":"fraction of actuator effort, mapped to existing position actions",
        "fixed_preparation_sha256":hashlib.sha256(args.tail_of.read_bytes()).hexdigest() if args.tail_of else None,
        "optimization_variables":"aerial guide time scale and air/landing residuals" if args.tail_of else "preparation duration and support/swing residuals",
        "not_a_trained_policy":True},indent=2))

    @torch.no_grad()
    def rollout(parameters):
        nonlocal evaluations
        evaluations+=len(parameters)
        clock["step"]=0
        physics_limits[:]=0
        physics_mask[:]=True
        active_physics["record"]=False
        clock["initial_frame"]=0
        prep_parameters=np.tile(fixed_preparation,(args.num_envs,1)) if args.tail_of else parameters
        clock["preparation"]=torch.tensor(prep_parameters[:,0],dtype=torch.float32,device=device)
        clock["air_scale"]=torch.tensor(parameters[:,0] if args.tail_of else np.ones(args.num_envs),dtype=torch.float32,device=device)
        cmd.forced_frame[:]=0
        obs,_=env.reset()
        clock["initial_frame"]=int(cmd.frame[0])
        clock["step"]=0
        active_physics["record"]=True
        offsets=torch.tensor(prep_parameters[:,1:].reshape(args.num_envs,3,len(groups)),dtype=torch.float32,device=device)
        tail_offsets=torch.tensor(parameters[:,1:].reshape(args.num_envs,3,len(groups)),dtype=torch.float32,device=device) if args.tail_of else None
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
        states=[];velocities=[];actions=[]
        records=[[] for _ in range(args.num_envs)]
        maxima=[{"soft":0.,"speed":0.,"effort":0.,"nonfoot_penetration":0.,
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
                air_duration=.5*clock["air_scale"]
                tail_f=torch.clamp(after/air_duration,0,1)*3
                tail_index=tail_f.long().clamp(max=2)
                tail_fraction=tail_f-tail_index
                tail_knots=torch.cat([torch.zeros_like(initial_delta[:,None,:]),tail_offsets@mapping],dim=1)
                tail=(tail_knots[ids,tail_index]*(1-tail_fraction[:,None])
                      +tail_knots[ids,tail_index+1]*tail_fraction[:,None])
                tail*=torch.clamp(1-(after-air_duration)/.4,0,1)[:,None]
                residual+=tail
            action=policy(obs["policy"],obs["goal"])+residual
            obs,_,_,_,_=env.step(action)
            q=env.sim.data.qpos.cpu().numpy().copy()
            v=env.sim.data.qvel.cpu().numpy().copy()
            force=env.sim.data.actuator_force.cpu().numpy()
            states.append(q);velocities.append(v);actions.append(action.cpu().numpy())
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
                if min(feet.values())>.02 and measure[1]>.8 and nonfoot>.005:
                    air_run[e]+=1
                else:air_run[e]=0
                if air_run[e]>=3:
                    completed_air[e]=True
                    if args.phase=="takeoff":physics_mask[e]=False
            if args.phase=="takeoff" and completed_air.all():break
        active_physics["record"]=False
        reports=[]
        for e,rows in enumerate(records):
            pitch=np.unwrap([r["pitch"] for r in rows])
            air=[i for i,r in enumerate(rows) if min(r["feet"].values())>.02 and r["nonfoot_height"]>.005]
            single=[i for i,r in enumerate(rows) if r["feet"]["left"]<.01 and r["feet"]["right"]>.04]
            event_cost=min((r["vz"]-2.4)**2*4+np.sum((np.array(r["momentum"])-[0,25,0])**2)/64
                +sum(max(.025-h,0)*60 for h in r["feet"].values()) for r in rows if .25<=r["time"]<=1.2)
            physical=maxima[e]
            # For a completed takeoff, stats stop at that event. Full-clip
            # checks always use every 5 ms physics step, not only video frames.
            for k,key in enumerate(("soft","speed","effort")):
                physical[key]=max(physical[key],float(physics_limits[e,k]))
            cost=event_cost+physical["soft"]**2*10000+max(physical["speed"]-1,0)**2*20
            cost+=physical["nonfoot_penetration"]**2*2000+physical["foot_penetration"]**2*2000
            if args.phase=="full":
                cost+=max(5.7-pitch.max()+pitch[0],0)**2*2
                cost+=max(.8-rows[-1]["up_z"],0)**2*10+max(.55-rows[-1]["com_z"],0)**2*50
                cost+=np.mean([r["com_speed"]**2*4+max(.85-r["up_z"],0)**2*10
                    +r["root_angular_speed"]**2*.5
                    +max(min(r["feet"].values())-.01,0)**2*200 for r in rows[-10:]])
            cost+=10 if len(single)<4 else 0
            first_air=next((i for i in range(len(rows)-2) if all(j in air for j in range(i,i+3))),None)
            single_at_launch=first_air is not None and sum(i in single for i in range(max(0,first_air-8),first_air))>=4
            launch_row=rows[min(first_air+2,len(rows)-1)] if first_air is not None else None
            takeoff_passed=(completed_air[e] and single_at_launch and physical["soft"]<1e-4
                and launch_row["vz"]>1.5 and launch_row["momentum"][1]>20
                and physical["speed"]<=1 and physical["effort"]<=1.00001
                and physical["minimum_nonfoot_height"]>.005 and physical["foot_penetration"]<.005)
            landed=all(r["up_z"]>.85 and r["com_z"]>.55 and r["com_speed"]<.5 and r["root_angular_speed"]<1
                and min(r["feet"].values())<.01 for r in rows[-10:])
            air_end=first_air
            if first_air is not None:
                while air_end+1 in air:air_end+=1
            air_rotation=float(pitch[air_end]-pitch[first_air]) if first_air is not None else 0.
            passed=(args.phase=="full" and len(air)>=4 and single_at_launch and pitch.max()-pitch[0]>5.5
                and air_rotation>3.5
                and min(rows[i]["up_z"] for i in air)<-.7 and rows[-1]["up_z"]>.85
                and landed
                and physical["soft"]<1e-4 and physical["speed"]<=1 and physical["effort"]<=1.00001
                and physical["minimum_nonfoot_height"]>.005 and physical["foot_penetration"]<.005)
            reports.append({"cost":float(cost),"maxima":physical,"air_frames":len(air),"single_left_support_frames":len(single),
                "maximum_forward_rotation_rad":float(pitch.max()-pitch[0]),"final":rows[-1],
                "airborne_forward_rotation_rad":air_rotation,
                "takeoff_candidate_passed":bool(takeoff_passed),"single_left_support_at_launch":bool(single_at_launch),
                "physical_candidate_passed":bool(passed),"approved_for_training":False})
        return reports,(np.array(states),np.array(velocities),np.array(actions),records,initial_state)

    try:
        # Establish the exact-controller baseline before searching.
        baseline_parameters=np.tile(mean,(args.num_envs,1))
        baseline,payload=rollout(baseline_parameters)
        (args.out/"baseline.json").write_text(json.dumps(baseline[0],indent=2))
        np.savez(args.out/"baseline-trajectory.npz",qpos=payload[0][:,0],qvel=payload[1][:,0],actions=payload[2][:,0],fps=np.array([50.]))
        print("BASELINE",json.dumps(baseline[0]),flush=True)
        repeat,repeat_payload=rollout(baseline_parameters)
        error=float(np.max(np.abs(payload[0]-repeat_payload[0])))
        per_step=np.max(np.abs(payload[0]-repeat_payload[0]),axis=(1,2))
        initial_errors={k:float(np.abs(payload[4][k]-repeat_payload[4][k]).max()) for k in payload[4]}
        initial_passed=max(initial_errors.values())<1e-5
        np.savez(args.out/"reset-repeat-trajectory.npz",qpos=repeat_payload[0],qvel=repeat_payload[1])
        (args.out/"reset-reproducibility.json").write_text(json.dumps({"maximum_qpos_error":error,
            "maximum_qpos_error_by_step":per_step.tolist(),
            "initial_state_errors":initial_errors,"initial_state_check_passed":initial_passed,
            "candidate_validation_required":"Three independent successful full physical rollouts before export or training; GPU numerical trajectory differences are not treated as success.",
            "passed":error<1e-3,"scope":"Same controller, goals, initial state and reset; whole rollout compared"},indent=2))
        assert initial_passed, "Initial state or first policy action differs across resets; design search rejected"
        if device.type=="cpu":
            assert error<1e-3, "Repeated deterministic CPU resets disagree; design search rejected"
        for iteration in range(args.iterations):
            iteration_start=time.monotonic()
            population=np.clip(rng.multivariate_normal(mean,covariance,args.population),lo,hi)
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
            elite=np.array([r[1] for r in outcomes[:max(2,len(outcomes)//4)]])
            mean=.3*mean+.7*elite.mean(axis=0)
            covariance=.7*covariance+.3*np.cov(elite,rowvar=False)+np.diag(np.r_[.003,np.full(dimension-1,.01)]**2)
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
        (args.out/"completed.json").write_text(json.dumps({"elapsed_s":time.monotonic()-start,"evaluations":evaluations,
            "physical_candidate_passed":best[2]["physical_candidate_passed"] if best else False,
            "takeoff_candidate_passed":best[2]["takeoff_candidate_passed"] if best else False,"approved_for_training":False},indent=2))
    finally:
        env.close()


if __name__=="__main__":main()
