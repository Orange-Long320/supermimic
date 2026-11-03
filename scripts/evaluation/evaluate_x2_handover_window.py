"""CPU free-base preparation/handover diagnostic for the frozen 11500 policy.

Ground supports the feet; no root constraint, cable support, joint-limit grace,
observation padding, target clipping or policy updates are used. Snapshot root
orientation uses the recorded base_link IMU as an explicit uncalibrated model
assumption; height is derived from model foot geometry, not measured position.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml
from scipy.optimize import lsq_linear

from supermimic.deploy.early_handover import EarlyHandover, Phase
from supermimic.deploy.skill_goal import SkillGoal
from supermimic.deploy.torch_policy import TorchScriptPolicy


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def static_support_effort(model, data, foot_geoms, motor_dofs):
    """Model-only vertical support equilibrium, without moving the free root.

    Only sphere bottoms near the lowest foot are eligible. A nonnegative
    normal-force solution must balance all six unactuated coordinates. This
    is a diagnostic inverse-statics assumption, not measured contact forces.
    """
    state=mujoco.MjData(model);state.qpos[:]=data.qpos
    mujoco.mj_forward(model,state)  # zero velocity: gravity, no velocity bias
    bottoms=[float(state.geom_xpos[g,2]-model.geom_size[g,0]) for gs in foot_geoms for g in gs]
    lowest=min(bottoms);jacobians=[]
    for gs in foot_geoms:
        for g in gs:
            if state.geom_xpos[g,2]-model.geom_size[g,0]>lowest+.0005:continue
            point=state.geom_xpos[g].copy();point[2]-=model.geom_size[g,0]
            jac=np.zeros((3,model.nv));mujoco.mj_jac(model,state,jac,None,point,int(model.geom_bodyid[g]))
            jacobians.append(jac[2].copy())
    columns=np.asarray(jacobians).T
    result=lsq_linear(columns[:6],state.qfrc_bias[:6],bounds=(0.,np.inf),tol=1e-12,max_iter=200)
    residual=float(np.max(np.abs(columns[:6]@result.x-state.qfrc_bias[:6])))
    if not result.success or residual>1e-5:
        raise ValueError('Model vertical support equilibrium infeasible: '+str(residual))
    return (state.qfrc_bias-columns@result.x)[motor_dofs],dict(
        equilibrium_residual=residual,total_normal_n=float(result.x.sum()),points=len(jacobians))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('scene','profile','dataset','policy','candidate','snapshot','prep-gains','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--start',choices=('reference','snapshot'),required=True)
    p.add_argument('--prep-gain-source',choices=('sdk','policy'),required=True)
    p.add_argument('--blend-seconds',type=float,required=True)
    p.add_argument('--history-source', choices=('executed', 'policy'), default='policy',
                   help='Simulation-only comparison of feedback semantics')
    p.add_argument('--active-seconds',type=float,default=5.)
    p.add_argument('--stand-elbow-margin',type=float,default=0.,
                   help='Model-only standing-goal elbow margin; does not alter source references or weights')
    p.add_argument('--preparation-elbow-margin',type=float,default=0.,
                   help='Model-only preparation target margin, separate from the policy goal margin')
    p.add_argument('--waist-pitch-sdk-gains',action='store_true',
                   help='Model-only active waist-pitch gain comparison; never a hardware gain approval')
    p.add_argument('--preparation-gravity-compensation',action='store_true',
                   help='Model-only inverse-statics support torque, faded out during handover')
    p.add_argument('--preparation-only-seconds',type=float,default=0.,
                   help='Diagnostic preparation window without requesting policy handover (0 uses normal test)')
    p.add_argument('--suspended-root-model',action='store_true',
                   help='Preparation-only counterfactual: rigid world-to-pelvis support, feet clear of floor; not an actual sling model')
    a=p.parse_args()
    if (a.out.exists() or not .1<=a.blend_seconds<=2 or not 5<=a.active_seconds<=30
            or not 0<=a.stand_elbow_margin<=.25 or not 0<=a.preparation_elbow_margin<=.25
            or not (a.preparation_only_seconds==0 or .2<=a.preparation_only_seconds<=20.)
            or (a.suspended_root_model and not a.preparation_only_seconds)):
        raise ValueError('New output and bounded diagnostic duration required')
    a.out.mkdir(parents=True)
    if sha(a.policy)!='4424f231fc1896da098b2fca366e8522df6791596638b2b9257ba682520bbe4c':
        raise ValueError('Wrong policy')
    torch.set_num_threads(1)
    spec=json.loads(a.profile.read_text());c=json.loads(a.candidate.read_text())
    snapshot=json.loads(a.snapshot.read_text());names=spec['joint_names'];n=len(names)
    if n!=31 or c['joint_names']!=names or snapshot['names']!=names:
        raise ValueError('Joint order mismatch')
    if a.suspended_root_model:
        model_spec=mujoco.MjSpec.from_file(str(a.scene));support=model_spec.add_equality()
        support.name='diagnostic_rigid_pelvis_support';support.type=mujoco.mjtEq.mjEQ_WELD
        support.objtype=mujoco.mjtObj.mjOBJ_BODY;support.name1='world';support.name2='pelvis'
        m=model_spec.compile()
    else:m=mujoco.MjModel.from_xml_path(str(a.scene))
    d=mujoco.MjData(m)
    ids=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,x) for x in names])
    if min(ids)<0 or m.nu!=n:raise ValueError('Unexpected joints or actuators')
    qa=m.jnt_qposadr[ids];va=m.jnt_dofadr[ids]
    by_joint={int(j):i for i,j in enumerate(m.actuator_trnid[:,0])}
    motors=np.array([by_joint[int(j)] for j in ids]);limits=m.actuator_ctrlrange[motors]
    pelvis=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'pelvis')
    floor=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'floor')
    feet=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,s+'_ankle_roll_link') for s in ('left','right')]
    if min(pelvis,floor,*feet)<0:raise ValueError('Required scene geometry missing')
    foot_geoms=[[i for i in range(m.ngeom) if int(m.geom_bodyid[i])==b
                 and (m.geom_contype[i] or m.geom_conaffinity[i])] for b in feet]
    if any(not gs for gs in foot_geoms) or any(m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_SPHERE for gs in foot_geoms for g in gs):
        raise ValueError('This diagnostic requires the verified official foot spheres')
    target=np.array(c['startup']['position']);lower=np.array(c['measured_lower_candidate']);upper=np.array(c['measured_upper_candidate'])
    preparation_override={}
    if a.preparation_elbow_margin:
        for name in ('left_elbow_joint','right_elbow_joint'):
            j=names.index(name);desired=lower[j]+a.preparation_elbow_margin
            if not lower[j]<desired<upper[j]:raise ValueError('Invalid model preparation margin')
            preparation_override[name]=dict(original=float(target[j]),candidate=float(desired));target[j]=desired
    start=target.copy() if a.start=='reference' else np.array(snapshot['q'])
    quat=np.array(c['startup']['root_quat_wxyz'] if a.start=='reference' else snapshot['imu']['orientation_wxyz'])
    if abs(np.linalg.norm(quat)-1)>1e-3:raise ValueError('Invalid initial IMU/reference orientation')
    quat/=np.linalg.norm(quat)
    d.qpos[:7]=np.r_[[0.,0.,1.],quat];d.qpos[qa]=start
    d.qvel[va]=0. if a.start=='reference' else snapshot['qd']
    mujoco.mj_forward(m,d)
    bottoms=[min(float(d.geom_xpos[g,2]-m.geom_size[g,0]) for g in gs) for gs in foot_geoms]
    d.qpos[2]-=min(bottoms)+.0002
    if a.suspended_root_model:
        d.qpos[2]+=.1
        eid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_EQUALITY,'diagnostic_rigid_pelvis_support')
        m.eq_data[eid,3:10]=d.qpos[:7]
    mujoco.mj_forward(m,d)
    initial_root=d.qpos[:7].copy()
    initial_bottoms=[min(float(d.geom_xpos[g,2]-m.geom_size[g,0]) for g in gs) for gs in foot_geoms]
    default=np.array(c['action_default']);scale=np.array(c['action_scale'])
    kp=np.array([spec['stiffness'][x] for x in names]);kd=np.array([spec['damping'][x] for x in names])
    gains=yaml.safe_load(a.prep_gains.read_text());special={x['name']:x for x in gains['default_joints']['joints']}
    if a.prep_gain_source=='policy':pk,pd=kp.copy(),kd.copy()
    else:
        pk=np.array([special.get(x,{}).get('kp',gains['default_kp']) for x in names])
        pd=np.array([special.get(x,{}).get('kd',gains['default_kd']) for x in names])
    gain_override={}
    if a.waist_pitch_sdk_gains:
        name='waist_pitch_joint';j=names.index(name)
        gain_override[name]=dict(original_kp=float(kp[j]),original_kd=float(kd[j]),
            candidate_kp=float(special[name]['kp']),candidate_kd=float(special[name]['kd']))
        kp[j]=special[name]['kp'];kd[j]=special[name]['kd']
    policy=TorchScriptPolicy(a.policy)
    tracker=SkillGoal(a.dataset,a.profile,d.qpos[:3].copy(),d.qpos[3:7].copy())
    goal_override={}
    if a.stand_elbow_margin:
        for name in ('left_elbow_joint','right_elbow_joint'):
            j=names.index(name);desired=lower[j]+a.stand_elbow_margin
            if not lower[j]<desired<upper[j]:raise ValueError('Invalid model standing margin')
            goal_override[name]=dict(original=float(tracker.scheduler.stand[j]+default[j]),candidate=float(desired))
            tracker.scheduler.stand[j]=desired-default[j]
        # Before any goal has been exposed, seed the committed future window
        # from this explicit candidate. No received observation is rewritten.
        tracker.scheduler.queue=deque(tracker.scheduler.stand.copy() for _ in range(tracker.scheduler.horizon))
    controller=EarlyHandover(start,target,default,scale,pk,pd,kp,kd,lower,upper,lower,upper,np.full(n,.3),
        max_velocity=np.full(n,.35),max_acceleration=np.full(n,.4),minimum_duration=8.,
        blend_seconds=a.blend_seconds,max_tick_gap=.031,max_tilt=.35,max_angular_speed=2.,
        policy=policy,tracker_factory=lambda position,orientation:tracker,history_source=a.history_source)
    substeps=round(.02/m.opt.timestep)
    if abs(substeps*m.opt.timestep-.02)>1e-9:raise ValueError('Unexpected simulation timestep')
    rows=[];failure=None;active_start=None;clipped=0;total=0;peak_other=0.;support_diagnostic=None
    max_sim_seconds=a.preparation_only_seconds or (8.+a.blend_seconds+a.active_seconds)
    for i in range(round(max_sim_seconds/.02)):
        now=i*.02
        q=d.qpos[qa].copy();qd=d.qvel[va].copy();quat=d.xquat[pelvis].copy();w,x,y,z=quat
        gravity=np.array([2*(w*y-x*z),-2*(w*x+y*z),-1+2*(x*x+y*y)])
        velocity=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_XBODY,pelvis,velocity,1)
        try:
            if not a.preparation_only_seconds and controller.phase is Phase.PREPARE and len(controller.history)==10:
                controller.request_handover()
            out=controller.step(now,d.qpos[:3].copy(),quat,q,qd,velocity[:3],gravity)
            feedforward=np.zeros(n)
            if a.preparation_gravity_compensation and controller.phase is not Phase.ACTIVE:
                if a.suspended_root_model:
                    static=mujoco.MjData(m);static.qpos[:]=d.qpos;mujoco.mj_forward(m,static)
                    feedforward=static.qfrc_bias[va].copy()
                    support_diagnostic=dict(model_only='Gravity with rigid pelvis support, no foot reactions')
                else:feedforward,support_diagnostic=static_support_effort(m,d,foot_geoms,va)
                if controller.phase is Phase.HANDOVER:
                    s=float(np.clip((now-controller.blend_started)/controller.blend_seconds,0.,1.))
                    feedforward*=1-(10*s**3-15*s**4+6*s**5)
        except ValueError as exc:
            context=controller.fault_context
            if context is not None and 'indices' in context:
                context=dict(context,joint_names=[names[j] for j in context['indices']])
            failure=dict(time_s=now,reason=str(exc),phase=controller.phase.value,context=context);break
        if controller.phase is Phase.ACTIVE and active_start is None:active_start=now
        normal=np.zeros(2);other=0.;effort_ratio=0.
        for _ in range(substeps):
            effort=out['stiffness']*(out['position']-d.qpos[qa])-out['damping']*d.qvel[va]+feedforward
            clipped+=int(np.any((effort<limits[:,0]) | (effort>limits[:,1])));total+=1
            d.ctrl[motors]=np.clip(effort,limits[:,0],limits[:,1])
            effort_ratio=max(effort_ratio,float(np.max(np.abs(effort)/np.max(np.abs(limits),axis=1))))
            mujoco.mj_step(m,d)
            normal[:]=0.
            for j in range(d.ncon):
                contact=d.contact[j];g=[int(contact.geom[0]),int(contact.geom[1])];force=np.zeros(6)
                mujoco.mj_contactForce(m,d,j,force)
                if floor in g:
                    body=int(m.geom_bodyid[g[1] if g[0]==floor else g[0]])
                    if body in feet:normal[feet.index(body)]+=max(0.,force[0])
                    else:other=max(other,float(force[0]))
                else:other=max(other,float(force[0]))
        peak_other=max(peak_other,other)
        rows.append(dict(time_s=now,phase=controller.phase.value,root=d.qpos[:7].tolist(),
            q=q.tolist(),qd=qd.tolist(),target=out['position'].tolist(),normal_n=normal.tolist(),
            feedforward_model_only=feedforward.tolist(),support_diagnostic=support_diagnostic,
            unclipped_effort_ratio=effort_ratio,nonfoot_contact_n=other))
        if d.qpos[2]<.4:
            failure=dict(time_s=now,reason='Root height below .4 m',phase=controller.phase.value);break
        if active_start is not None and now-active_start>=a.active_seconds:break
    active_s=0. if active_start is None or not rows else rows[-1]['time_s']-active_start
    active_rows=[r for r in rows if active_start is not None and r['time_s']>=active_start+1.]
    loaded=sum(min(r['normal_n'])>5. for r in active_rows)/len(active_rows) if active_rows else 0.
    fraction=clipped/total if total else 0.
    result=dict(scope=__doc__,simulation_only=True,test_only=True,mujoco_version=mujoco.__version__,
        source_sha256={k:sha(getattr(a,k)) for k in ('scene','profile','dataset','policy','candidate','snapshot','prep_gains') if k!='dataset'},
        start=a.start,initial_root_model_assumption=initial_root.tolist(),initial_foot_bottoms_m=initial_bottoms,
        calibration_validated=False,payload_model='No additional payload; not a measured load model',
        support_model=('Counterfactual rigid world-to-pelvis weld; not measured sling geometry or compliance'
                       if a.suspended_root_model else 'Ground contacts only; fall-arrest harness supplies no balance force'),
        root_fixed=a.suspended_root_model,
        prep_gain_source=a.prep_gain_source,blend_seconds=a.blend_seconds,history_source=a.history_source,
        stand_goal_override_model_only=goal_override,
        preparation_override_model_only=preparation_override,
        policy_gain_override_model_only=gain_override,
        preparation_gravity_compensation_model_only=a.preparation_gravity_compensation,
        preparation_only_seconds=a.preparation_only_seconds,
        final_preparation_error_rad=None if not rows else float(np.max(np.abs(np.asarray(rows[-1]['q'])-target))),
        final_joint_speed_rad_s=None if not rows else float(np.max(np.abs(rows[-1]['qd']))),
        max_root_translation_from_initial_m=max((float(np.linalg.norm(np.array(r['root'][:3])-initial_root[:3])) for r in rows),default=0.),
        simulation_only_guard=dict(target_step_rad=.3,max_tilt_rad=.35,max_angular_speed_rad_s=2.),
        failure=failure,frames=len(rows),active_seconds=active_s,
        saturated_substep_fraction=fraction,maximum_nonfoot_contact_n=peak_other,
        both_feet_loaded_fraction_after_active_1s=loaded,
        passed_model_window=bool(not a.preparation_only_seconds and failure is None and active_s>=a.active_seconds-.001 and fraction<.01 and peak_other<=10. and loaded>=.95),
        motor_commands_sent=0,ready_for_hardware_trial=False)
    (a.out/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
