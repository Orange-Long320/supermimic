"""CPU diagnostic of measured-pose preparation and supported policy handover.

Harness geometry/stiffness here are explicit scenario assumptions, not measured
properties of the real hoist. Unilateral cable forces act on the torso; the root
is never fixed. A successful scenario alone does not release hardware control.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

from supermimic.deploy.entry import EntryController, EntryPhase, EntryThresholds, QuinticPose, PreparationLoadCompensator
from supermimic.deploy.independent_policy import IndependentPolicyCoordinator
from supermimic.deploy.skill_goal import SkillGoal
from supermimic.deploy.torch_policy import TorchScriptPolicy


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


class OfficialLinearPose:
    """Offline approximation of the SDK's 1.5 s JOINT_DEFAULT interpolation."""
    def __init__(self, start, target):
        self.start = np.asarray(start, dtype=float).copy()
        self.target = np.asarray(target, dtype=float).copy()
        self.duration = 1.5

    def sample(self, elapsed):
        progress = float(np.clip(elapsed / self.duration, 0., 1.))
        # The SDK sends zero velocity and effort even during interpolation.
        zero = np.zeros_like(self.start)
        return self.start + progress * (self.target - self.start), zero, zero


def diagnostic_urdf_pelvis_inertia(model, urdf):
    """Override only a private simulation model, preserving full inertia tensor."""
    link = next((e for e in ET.parse(urdf).getroot().findall('link') if e.get('name') == 'pelvis'), None)
    if link is None or link.find('inertial') is None: raise ValueError('Missing pelvis inertia')
    inertial = link.find('inertial'); origin = inertial.find('origin'); mass = float(inertial.find('mass').get('value'))
    pos = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
    rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
    it = inertial.find('inertia'); v = {k: float(it.get(k)) for k in ('ixx','iyy','izz','ixy','ixz','iyz')}
    tensor=np.array([[v['ixx'],v['ixy'],v['ixz']],[v['ixy'],v['iyy'],v['iyz']],[v['ixz'],v['iyz'],v['izz']]])
    rotation=Rotation.from_euler('xyz',rpy).as_matrix(); tensor=rotation@tensor@rotation.T
    principal, axes=np.linalg.eigh(tensor)
    if mass <= 0 or np.min(principal) <= 0 or pos.shape != (3,) or not np.isfinite(tensor).all():
        raise ValueError('Invalid pelvis inertia')
    if np.linalg.det(axes)<0: axes[:,0]*=-1
    body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'pelvis')
    if body<0: raise ValueError('Missing model pelvis')
    old={'mass_kg':float(model.body_mass[body]),'ipos_m':model.body_ipos[body].tolist()}
    model.body_mass[body]=mass; model.body_ipos[body]=pos; model.body_inertia[body]=principal
    model.body_iquat[body]=Rotation.from_matrix(axes).as_quat()[[3,0,1,2]]
    return {'original':old,'urdf_mass_kg':mass,'urdf_ipos_m':pos.tolist(),'urdf_sha256':sha(urdf)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('scene','profile','dataset','policy','snapshot','candidate','prep-gains','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--support-fraction',type=float,required=True)
    p.add_argument('--attachment-x',type=float,default=0.)
    p.add_argument('--spring',type=float,default=2000.)
    p.add_argument('--unload',action='store_true')
    p.add_argument('--unload-before-handover',action='store_true',help='Transfer single-point support before requiring level settled stance')
    p.add_argument('--active-seconds',type=float,default=15.)
    p.add_argument('--load-compensation', action='store_true')
    p.add_argument('--orientation-assist', action='store_true', help='Simulation candidate ankle tilt feedback')
    p.add_argument('--diagnostic-handover-tilt',type=float,
                   help='Counterfactual capture test only; override the .15 rad simulation entry tilt gate')
    p.add_argument('--unload-at-handover',action='store_true',
                   help='Diagnostic simultaneous policy blend and cable unloading')
    p.add_argument('--handover-seconds',type=float,default=1.)
    p.add_argument('--unload-seconds',type=float,default=5.)
    p.add_argument('--prep-gain-source',choices=('sdk','policy'),default='sdk',
                   help='Diagnostic comparison of preparation and trained actuator gains')
    p.add_argument('--waist-orientation-assist',action='store_true',
                   help='Simulation candidate: use waist posture to balance a single-point suspension')
    p.add_argument('--diagnostic-initial-limit-grace',action='store_true',
                   help='Counterfactual only: disable initially violated model limits until each joint reenters its original range')
    p.add_argument('--official-preparation', action='store_true',
                   help='Simulation only: 1.5 s linear interpolation to SDK default_joints with SDK gains')
    p.add_argument('--diagnostic-pelvis-urdf', type=Path,
                   help='Simulation-only counterfactual: use pelvis mass and inertia from this URDF')
    p.add_argument('--independent-coordinator', action='store_true',
                   help='Use actuator-free independent coordinator; automatically request handover when ready in this simulation')
    p.add_argument('--diagnostic-reference-prepare', action='store_true',
                   help='Counterfactual: start exactly at reference target with short settle; not an actual preparation')
    p.add_argument('--diagnostic-early-blend',type=float,
                   help='Simulation only: after 10 actual physics frames, blend immediately to policy over this many seconds')
    p.add_argument('--diagnostic-policy-trace',action='store_true',
                   help='Retain policy input and action per simulated frame for offline causality diagnostics')
    p.add_argument('--diagnostic-previous-action-scale',type=float,default=1.,
                   help='Simulation only: ablate previous-action input; changes policy observation semantics')
    p.add_argument('--diagnostic-profile-armature',action='store_true',
                   help='Simulation-only: replace official XML armature with training profile values')
    a=p.parse_args()
    if (not 0 <= a.support_fraction <= 1. or not abs(a.attachment_x)<=.1
            or not 500<=a.spring<=10000 or not 5<=a.active_seconds<=120
            or not 1<=a.handover_seconds<=10 or not 1<=a.unload_seconds<=10
            or (a.unload_at_handover and (not a.unload or a.unload_before_handover))
            or (a.orientation_assist and a.waist_orientation_assist)
            or (a.diagnostic_early_blend is not None and (not .02<=a.diagnostic_early_blend<=2.
                    or a.independent_coordinator or a.unload or a.load_compensation or a.orientation_assist
                    or a.waist_orientation_assist))
            or not 0.<=a.diagnostic_previous_action_scale<=1.
            or (a.diagnostic_previous_action_scale!=1. and a.diagnostic_early_blend is None)
            or (a.diagnostic_handover_tilt is not None and not .15<a.diagnostic_handover_tilt<=.35)):
        raise ValueError('Invalid scenario')
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    spec=json.loads(a.profile.read_text());names=spec['joint_names']; n=len(names)
    snapshot=json.loads(a.snapshot.read_text());candidate=json.loads(a.candidate.read_text())
    if snapshot['joint_names']!=names or candidate['joint_names']!=names or n!=31: raise ValueError('Joint order mismatch')
    if sha(a.policy)!='4424f231fc1896da098b2fca366e8522df6791596638b2b9257ba682520bbe4c':raise ValueError('Wrong policy')
    m=mujoco.MjModel.from_xml_path(str(a.scene));d=mujoco.MjData(m)
    inertia_diagnostic=None
    if a.diagnostic_pelvis_urdf:
        inertia_diagnostic=diagnostic_urdf_pelvis_inertia(m,a.diagnostic_pelvis_urdf)
        mujoco.mj_setConst(m,d)
    if abs(round(.02/m.opt.timestep)*m.opt.timestep-.02)>1e-9:raise ValueError('Timestep mismatch')
    joint=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name) for name in names]
    if min(joint)<0:raise ValueError('Missing joint')
    qa=np.asarray([m.jnt_qposadr[j] for j in joint]);va=np.asarray([m.jnt_dofadr[j] for j in joint])
    armature_before=m.dof_armature[va].copy()
    if a.diagnostic_profile_armature:
        armature=np.asarray([spec['armature'][name] for name in names],dtype=float)
        if not np.isfinite(armature).all() or np.any(armature<0):raise ValueError('Invalid profile armature')
        m.dof_armature[va]=armature;mujoco.mj_setConst(m,d)
    lookup={int(j):i for i,j in enumerate(m.actuator_trnid[:,0])};motors=np.asarray([lookup[j] for j in joint])
    limits=m.actuator_ctrlrange[motors]
    torso=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'torso_link')
    pelvis=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'pelvis')
    floor=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'floor')
    feet=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,s+'_ankle_roll_link') for s in ('left','right')]
    raw=np.asarray(snapshot['joint_position']);ref=np.asarray(candidate['startup']['position'])
    original_joint_limited=m.jnt_limited.copy();original_joint_ranges=m.jnt_range.copy()
    grace_joints={int(j) for j,q in zip(joint,raw) if m.jnt_limited[j]
                  and not m.jnt_range[j,0]<=q<=m.jnt_range[j,1]}
    initial_grace_names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j) for j in sorted(grace_joints)]
    if a.diagnostic_initial_limit_grace:
        for j in grace_joints:m.jnt_limited[j]=False
    else:grace_joints=set()
    grace_events=[]
    root=np.asarray(snapshot['position']).copy();root[:2]=0.
    d.qpos[:7]=np.r_[root,snapshot['quat_wxyz']];d.qpos[qa]=raw
    d.qvel[va]=snapshot['joint_velocity'];mujoco.mj_forward(m,d)
    default=np.asarray(candidate['action_default']);scale=np.asarray(candidate['action_scale'])
    kp=np.asarray([spec['stiffness'][name] for name in names]);kd=np.asarray([spec['damping'][name] for name in names])
    gains=yaml.safe_load(a.prep_gains.read_text());special={j['name']:j for j in gains['default_joints']['joints']}
    prep_kp=np.asarray([special.get(name,{}).get('kp',gains['default_kp']) for name in names])
    prep_kd=np.asarray([special.get(name,{}).get('kd',gains['default_kd']) for name in names])
    if a.prep_gain_source=='policy':prep_kp,prep_kd=kp.copy(),kd.copy()
    if a.official_preparation:
        if a.prep_gain_source != 'sdk': raise ValueError('Official preparation requires SDK gains')
        ref=np.asarray([special.get(name,{}).get('position',0.) for name in names],dtype=float)
        path=OfficialLinearPose(raw,ref)
    else:
        if a.diagnostic_reference_prepare and np.max(np.abs(raw-ref))>1e-8:
            raise ValueError('Reference diagnostic requires the exact target pose')
        path=QuinticPose(raw,ref,np.full(n,.35),np.full(n,.4),minimum_duration=.1 if a.diagnostic_reference_prepare else 8.)
    thresholds=EntryThresholds(.08,.12,.20,a.diagnostic_handover_tilt or .15,
                               .1 if a.diagnostic_reference_prepare else .5,12.,.031,10,a.handover_seconds)
    entry=EntryController(path,thresholds,prep_kp,prep_kd,kp,kd)
    compensator=PreparationLoadCompensator(prep_kp,prep_kd,np.max(np.abs(limits),axis=1),
                    integral_gain=.3,max_bias_speed=.08,effort_fraction=.5) if a.load_compensation else None
    policy=TorchScriptPolicy(a.policy);tracker=None;history=deque(maxlen=10)
    coordinator=(IndependentPolicyCoordinator(entry,default,scale,policy,
                 lambda position,quat: SkillGoal(a.dataset,a.profile,position,quat))
                 if a.independent_coordinator else None)
    previous=(raw-default)/scale;previous_target=raw.copy();current_target=raw.copy()
    local_attach=np.array([a.attachment_x,0.,.15]);point=d.xpos[torso]+d.xmat[torso].reshape(3,3)@local_attach
    anchor=point+np.array([0.,0.,2.]);mass=float(m.body_mass.sum());weight=mass*abs(m.opt.gravity[2])
    rest=2.-a.support_fraction*weight/a.spring;damping=100.
    active_start=None;failure=None;events=[];rows=[];last_phase=entry.phase;unload_start=None
    support_force=np.zeros(3)
    early_started=None;early_start_target=None
    balance_bias=np.zeros(n)
    waist=names.index('waist_pitch_joint');waist_pendulum_jacobian=None
    if a.waist_orientation_assist:
        # Derive the hanging equilibrium response from this model's COM and
        # attachment, rather than assuming a foot is fixed while suspended.
        scratch=mujoco.MjData(m);scratch.qpos[:7]=[0.,0.,.65,1.,0.,0.,0.];scratch.qpos[qa]=ref
        angles=[];epsilon=1e-4
        for delta in (-epsilon,epsilon):
            scratch.qpos[qa[waist]]=ref[waist]+delta;mujoco.mj_forward(m,scratch)
            attach=scratch.xpos[torso]+scratch.xmat[torso].reshape(3,3)@local_attach
            vector=scratch.subtree_com[pelvis]-attach
            angles.append(float(np.arctan2(vector[0],-vector[2])))
        waist_pendulum_jacobian=(angles[1]-angles[0])/(2*epsilon)
        if not np.isfinite(waist_pendulum_jacobian) or abs(waist_pendulum_jacobian)<.05:
            raise ValueError('Suspension waist response cannot be resolved')
    initial_outside=[name for name,q,lo,hi in zip(names,raw,candidate['measured_lower_candidate'],candidate['measured_upper_candidate']) if not lo<=q<=hi]
    substeps=round(.02/m.opt.timestep)
    for step in range(round((path.duration+thresholds.settle_timeout+2.+a.active_seconds)/.02)):
        now=step*.02;q=d.qpos[qa].copy();qd=d.qvel[va].copy();quat=d.xquat[pelvis].copy();w,x,y,z=quat
        gravity=np.array([2*(w*y-x*z),-2*(w*x+y*z),-1+2*(x*x+y*y)])
        body_velocity=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_XBODY,pelvis,body_velocity,1)
        omega=body_velocity[:3].copy();proposed=None;trace=None
        try:
            if (a.unload_before_handover and unload_start is None and now>=path.duration
                    and np.max(np.abs(q-ref-(balance_bias if a.waist_orientation_assist else 0.)))<thresholds.position_error
                    and np.max(np.abs(qd))<thresholds.joint_speed and np.linalg.norm(omega)<thresholds.angular_speed
                    and rows and min(rows[-1]['normal_n'])>10.):
                unload_start=now;events.append({'time_s':now,'phase':'support_transfer_started'})
            if a.diagnostic_early_blend is not None:
                history.append((q-default,qd,omega,gravity,previous.copy()*a.diagnostic_previous_action_scale))
                if tracker is None:tracker=SkillGoal(a.dataset,a.profile,d.qpos[:3].copy(),d.qpos[3:7].copy())
            elif coordinator is None and entry.phase in (EntryPhase.HISTORY,EntryPhase.HANDOVER,EntryPhase.ACTIVE):
                history.append((q-default,qd,omega,gravity,previous.copy()))
                if tracker is None:tracker=SkillGoal(a.dataset,a.profile,d.qpos[:3].copy(),d.qpos[3:7].copy())
            if a.diagnostic_early_blend is not None and len(history)==10:
                goal=tracker.goal(d.qpos[:3].copy(),d.qpos[3:7].copy(),q,'stand')
                obs=tuple(float(v) for term in range(5) for h in history for v in h[term])
                action=np.asarray(policy.infer(obs,tuple(float(v) for v in goal)))
                if action.shape!=(n,) or not np.isfinite(action).all():raise ValueError('Invalid policy action')
                if a.diagnostic_policy_trace:trace={'observation':list(obs),'goal':np.asarray(goal).tolist(),'action':action.tolist()}
                proposed=default+scale*action;tracker.advance()
            elif coordinator is None and entry.phase in (EntryPhase.HANDOVER,EntryPhase.ACTIVE):
                if len(history)!=10:raise ValueError('Incomplete actual preparation command history')
                goal=tracker.goal(d.qpos[:3].copy(),d.qpos[3:7].copy(),q,'stand')
                obs=tuple(float(v) for term in range(5) for h in history for v in h[term])
                action=np.asarray(policy.infer(obs,tuple(float(v) for v in goal)))
                proposed=default+scale*action;tracker.advance()
            bias=None
            if entry.phase in (EntryPhase.PREPARE,EntryPhase.SETTLE,EntryPhase.HISTORY):
                desired=path.sample(now)[0]
                if a.waist_orientation_assist:
                    pitch=float(np.arcsin(np.clip(gravity[0],-1.,1.)))
                    requested=np.zeros(n)
                    requested[waist]=np.clip(-(pitch+.2*omega[1])/waist_pendulum_jacobian,
                        max(-.3,candidate['measured_lower_candidate'][waist]-ref[waist]),
                        min(.3,candidate['measured_upper_candidate'][waist]-ref[waist]))
                    balance_bias+=np.clip(requested-balance_bias,-.04*.02,.04*.02)
                    desired=desired+balance_bias
                elif a.orientation_assist and (not a.unload_before_handover or unload_start is not None):
                    roll=float(np.arctan2(-gravity[1],-gravity[2]));pitch=float(np.arcsin(np.clip(gravity[0],-1.,1.)))
                    requested=np.zeros(n);ramp=float(np.clip(now/2.,0.,1.));ramp=10*ramp**3-15*ramp**4+6*ramp**5
                    for side,foot in zip(('left','right'),feet):
                        indices=[names.index(side+'_ankle_pitch_joint'),names.index(side+'_ankle_roll_joint')]
                        jac=np.zeros((3,m.nv));mujoco.mj_jacBody(m,d,None,jac,foot)
                        axes=d.xmat[pelvis].reshape(3,3).T@jac[:,va[indices]]
                        # With a foot on the ground, root rotation changes
                        # opposite to ankle-relative rotation. Derive the ankle
                        # signs from model axes instead of joint name guesses.
                        error=np.array([roll+.1*omega[0],pitch+.1*omega[1],0.])
                        correction=np.linalg.lstsq(axes,error,rcond=None)[0]
                        requested[indices]=ramp*np.clip(correction,[-.25,-.15],[.25,.15])
                    balance_bias+=np.clip(requested-balance_bias,-.08*.02,.08*.02)
                    desired=desired+balance_bias
                if compensator is not None: bias=compensator.update(desired,q,qd,.02)+balance_bias
                elif a.orientation_assist or a.waist_orientation_assist: bias=balance_bias.copy()
            support_ready=(not a.unload_before_handover or (unload_start is not None and now-unload_start>=5.
                           and np.linalg.norm(support_force)<.05*weight and rows and min(rows[-1]['normal_n'])>10.))
            if a.diagnostic_early_blend is not None:
                if proposed is None:
                    current_target,target_v,_=path.sample(now);kp_now,kd_now=prep_kp,prep_kd
                else:
                    if early_started is None:
                        early_started=now;early_start_target=previous_target.copy()
                        entry.phase=EntryPhase.HANDOVER;entry.handover_started=now
                    s=float(np.clip((now-early_started)/a.diagnostic_early_blend,0.,1.))
                    blend=10*s**3-15*s**4+6*s**5
                    current_target=(1-blend)*early_start_target+blend*proposed
                    target_v=np.zeros(n);kp_now=(1-blend)*prep_kp+blend*kp;kd_now=(1-blend)*prep_kd+blend*kd
                    if s>=1.:entry.phase=EntryPhase.ACTIVE
            elif coordinator is not None:
                if bias is not None: raise ValueError('Coordinator simulation does not implement preparation bias')
                if entry.phase is EntryPhase.HISTORY and len(coordinator.history)==thresholds.history_frames:
                    coordinator.request_handover()
                current_target,target_v,kp_now,kd_now=coordinator.step(now,d.qpos[:3],d.qpos[3:7],q,qd,omega,gravity,
                                                                      support_ready=bool(support_ready))
            else:
                current_target,target_v,kp_now,kd_now=entry.step(now,q,qd,omega,gravity,policy_target=proposed,
                                                             preparation_bias=bias,support_ready=bool(support_ready))
            if entry.phase!=last_phase:
                events.append({'time_s':now,'phase':entry.phase.value})
                if entry.phase in (EntryPhase.HISTORY,EntryPhase.SETTLE):history.clear()
                if entry.phase==EntryPhase.ACTIVE:active_start=now
                last_phase=entry.phase
            if d.qpos[2]<.4 or gravity[2]>-.7:raise ValueError('Height or tilt gate')
            previous=(current_target-default)/scale
            maximum_tension=0.;support_force=np.zeros(3)
            for _ in range(substeps):
                for j in tuple(grace_joints):
                    value=d.qpos[m.jnt_qposadr[j]]
                    if original_joint_ranges[j,0]<=value<=original_joint_ranges[j,1]:
                        m.jnt_limited[j]=original_joint_limited[j];grace_joints.remove(j)
                        grace_events.append(dict(time_s=float(d.time),joint=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j)))
                d.qfrc_applied[:]=0.
                if a.support_fraction:
                    point=d.xpos[torso]+d.xmat[torso].reshape(3,3)@local_attach
                    cable=anchor-point;length=float(np.linalg.norm(cable));direction=cable/length
                    jac=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jac,None,point,torso)
                    beginning=(unload_start if a.unload_before_handover else entry.handover_started if a.unload_at_handover
                               else None if active_start is None else active_start+3.)
                    unload=0. if not a.unload or beginning is None else float(np.clip((now-beginning)/a.unload_seconds,0.,1.))
                    blend=10*unload**3-15*unload**4+6*unload**5
                    tension=max(0.,a.spring*(length-rest-.5*blend)-damping*float((jac@d.qvel)@direction))
                    support_force=tension*direction;maximum_tension=max(maximum_tension,tension)
                    mujoco.mj_applyFT(m,d,support_force,np.zeros(3),point,torso,d.qfrc_applied)
                torque=kp_now*(current_target-d.qpos[qa])-kd_now*d.qvel[va]
                d.ctrl[motors]=np.clip(torque,limits[:,0],limits[:,1]);mujoco.mj_step(m,d)
            mujoco.mj_forward(m,d)
            normal=np.zeros(2);slip_force=np.zeros(2);other_contacts=[];contact_details=[]
            for ci in range(d.ncon):
                c=d.contact[ci];geoms=list(map(int,c.geom));wrench=np.zeros(6);mujoco.mj_contactForce(m,d,ci,wrench)
                if floor in geoms:
                    other=geoms[1] if geoms[0]==floor else geoms[0];body=int(m.geom_bodyid[other])
                    if body in feet:
                        side=feet.index(body);load=max(0.,float(wrench[0]));jac=np.zeros((3,m.nv))
                        mujoco.mj_jac(m,d,jac,None,c.pos,body);normal[side]+=load
                        slip_force[side]+=load*float(np.linalg.norm((jac@d.qvel)[:2]))
                elif wrench[0]>1.:
                    other_contacts.append(float(wrench[0]))
                    contact_details.append({'geoms':[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) for g in geoms],
                        'bodies':[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])) for g in geoms],
                        'normal_n':float(wrench[0]),'distance_m':float(c.dist)})
            rows.append({'time_s':now,'phase':entry.phase.value,'root':d.qpos[:3].tolist(),
                'root_quat_after_wxyz':d.qpos[3:7].tolist(),'joint_position_after':d.qpos[qa].tolist(),
                'joint_position':q.tolist(),'target':current_target.tolist(),'kp':kp_now.tolist(),'kd':kd_now.tolist(),
                'max_target_step_rad':float(np.max(np.abs(current_target-previous_target))),
                'max_joint_error_rad':float(np.max(np.abs(q-ref))), 'max_joint_speed_rad_s':float(np.max(np.abs(qd))),
                'preparation_bias':entry.preparation_bias.tolist(),'balance_bias':balance_bias.tolist(),
                'support_ready':bool(support_ready),
                'angular_speed_rad_s':float(np.linalg.norm(omega)), 'gravity_body':gravity.tolist(), 'normal_n':normal.tolist(),
                'loaded_slip_m_s':np.divide(slip_force,normal,out=np.zeros(2),where=normal>1e-6).tolist(),
                'max_tension_n':maximum_tension,'last_support_force':support_force.tolist(),
                'actuator_effort_ratio':float(np.max(np.abs(d.actuator_force[motors])/np.max(np.abs(limits),axis=1))),
                'non_floor_contact_forces_n':other_contacts,'non_floor_contacts':contact_details})
            if trace is not None: rows[-1]['policy_trace']=trace
            previous_target=current_target.copy()
            if active_start is not None and now-active_start>=a.active_seconds:break
        except ValueError as exc:failure={'time_s':now,'phase':entry.phase.value,'reason':str(exc)};break
    (a.out/'frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    tail=[row for row in rows if active_start is not None and row['time_s']>=active_start+max(5.,a.active_seconds-5.)]
    result={'scope':__doc__,'mujoco_version':mujoco.__version__,'sha256':{k:sha(getattr(a,k)) for k in ('scene','profile','policy','snapshot','candidate','prep_gains')},
            'support':{'fraction_preload':a.support_fraction,'attachment_body':'torso_link','attachment_local':local_attach.tolist(),
                'spring_n_m':a.spring,'damping_ns_m':damping,'anchor_world':anchor.tolist(),'initial_rest_length_m':rest,'unload':a.unload,
                'unload_model':('after supported joint settling' if a.unload_before_handover else
                                'at handover start' if a.unload_at_handover else 'after active+3s')+
                               f', lengthen cable .5m over {a.unload_seconds}s; fixed ceiling anchor, unilateral tension only',
                'hardware_geometry_or_stiffness_validated':False},
            'initial_outside_file_envelope':initial_outside,'prep_duration_s':path.duration,'thresholds_sim_only':vars(thresholds),
            'official_preparation_simulation':a.official_preparation,
            'independent_coordinator_simulation':a.independent_coordinator,
            'diagnostic_reference_prepare':a.diagnostic_reference_prepare,
            'diagnostic_early_blend_s':a.diagnostic_early_blend,
            'diagnostic_previous_action_scale':a.diagnostic_previous_action_scale,
            'diagnostic_profile_armature':a.diagnostic_profile_armature,
            'original_armature_range': [float(armature_before.min()),float(armature_before.max())],
            'used_armature_range': [float(m.dof_armature[va].min()),float(m.dof_armature[va].max())],
            'diagnostic_early_scope':'Bypasses settle gate in simulation, history contains actual simulated frames; not authorized for hardware',
            'pelvis_inertia_counterfactual':inertia_diagnostic,
            'official_preparation_scope':'20 ms sampled approximation of SDK 1.5 s interpolation, followed by project handover gates; not an exact SDK controller replay',
            'prep_gain_source':('device SDK example' if a.prep_gain_source=='sdk' else 'selected policy profile')+
                               ', used only as a simulation candidate',
            'load_compensation_sim_only':a.load_compensation,
            'orientation_assist_sim_only':a.orientation_assist,
            'waist_orientation_assist_sim_only':a.waist_orientation_assist,
            'waist_pendulum_jacobian_rad_per_rad':waist_pendulum_jacobian,
            'orientation_assist_axes':'body-frame foot rotational Jacobian; no hardcoded ankle sign',
            'support_transfer_before_handover':a.unload_before_handover,'support_transfer_start_s':unload_start,
            'unload_at_handover':a.unload_at_handover,
            'counterfactual_capture_test':a.diagnostic_handover_tilt is not None,
            'counterfactual_scope':'Simulation entry gate experiment only; does not change hardware thresholds or approval',
            'counterfactual_initial_limit_grace':a.diagnostic_initial_limit_grace,
            'initial_limit_grace_scope':'Diagnostic in-memory model change only; not encoder correction, measured limit validation or hardware permission',
            'initial_limit_grace_joints':initial_grace_names if a.diagnostic_initial_limit_grace else [],
            'initial_limit_restored_events':grace_events,
            'initial_limit_grace_unrestored':[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j) for j in sorted(grace_joints)],
            'previous_action_history':'actual commanded PD target, normalized by selected action default/scale; no invented history',
            'events':events,'failure':failure,'frames':len(rows),'active_start_s':active_start,
            'active_seconds':0. if active_start is None or not rows else rows[-1]['time_s']-active_start,
            'mass_kg':mass,'root_fixed':False,'motor_output_enabled':False,'ready_for_hardware_trial':False}
    if rows:result['maximum_command_step_rad']=max(row['max_target_step_rad'] for row in rows)
    result['whole_run_phase_metrics']={}
    for phase in sorted(set(row['phase'] for row in rows)):
        phase_rows=[row for row in rows if row['phase']==phase]
        result['whole_run_phase_metrics'][phase]={
            'frames':len(phase_rows),
            'maximum_tilt_rad':float(max(np.arccos(np.clip(-row['gravity_body'][2],-1.,1.)) for row in phase_rows)),
            'maximum_angular_speed_rad_s':max(row['angular_speed_rad_s'] for row in phase_rows),
            'maximum_effort_ratio':max(row['actuator_effort_ratio'] for row in phase_rows),
            'maximum_raw_pd_target_step_rad':max(row['max_target_step_rad'] for row in phase_rows),
            'maximum_non_floor_contact_n':max([0.]+[f for row in phase_rows for f in row['non_floor_contact_forces_n']])}
    if tail:
        pos=np.asarray([row['root'] for row in tail]);normal=np.asarray([row['normal_n'] for row in tail]);slip=np.asarray([row['loaded_slip_m_s'] for row in tail])
        result['final_window']={'seconds':len(tail)*.02,'max_xy_change_m':float(np.linalg.norm(pos[:,:2]-pos[0,:2],axis=1).max()),
            'height_minmax_m':[float(pos[:,2].min()),float(pos[:,2].max())], 'normal_p05_n':np.percentile(normal,5,axis=0).tolist(),
            'both_loaded_over10n_fraction':float(np.mean(np.all(normal>10.,axis=1))),
            'slip_p95_m_s':np.percentile(slip,95,axis=0).tolist(),'angular_speed_p95_rad_s':float(np.percentile([row['angular_speed_rad_s'] for row in tail],95)),
            'max_tension_n':max(row['max_tension_n'] for row in tail)}
    (a.out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
