"""Compare standing entries with native MuJoCo contact forces, simulation only.

This independently drives the official XML model using profile PD gains and the
frozen policy. It is not the official ROS simulator and is not hardware proof.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('scene', 'profile', 'dataset', 'policy', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--hold-seconds', type=float, default=30.)
    a = p.parse_args()
    if a.out.exists() or a.hold_seconds < 5:
        raise ValueError('New output and at least 5 seconds of hold required')
    from supermimic.deploy.skill_goal import SkillGoal
    from supermimic.deploy.torch_policy import TorchScriptPolicy
    torch.set_num_threads(1)
    spec = json.loads(a.profile.read_text()); names = spec['joint_names']; n = len(names)
    if spec['robot'] != 'x2_ultra' or n != 31:
        raise ValueError('Expected X2 Ultra')
    m = mujoco.MjModel.from_xml_path(str(a.scene))
    if m.nu != n or abs(round(.02/m.opt.timestep)*m.opt.timestep-.02) > 1e-9:
        raise ValueError('Unexpected actuator count or simulation timestep')
    ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
    if min(ids) < 0:
        raise ValueError('Missing joint')
    qadr = np.array([m.jnt_qposadr[x] for x in ids]); vadr = np.array([m.jnt_dofadr[x] for x in ids])
    motor_by_joint = {int(j): i for i, j in enumerate(m.actuator_trnid[:, 0])}
    if set(motor_by_joint) != set(ids):
        raise ValueError('Motor and joint sets do not match')
    motors = np.array([motor_by_joint[j] for j in ids])
    default = np.array([spec['default_joint_pos'][x] for x in names])
    scale = np.array([spec['action_scale'][x] for x in names])
    kp = np.array([spec['stiffness'][x] for x in names]); kd = np.array([spec['damping'][x] for x in names])
    motor_limits = m.actuator_ctrlrange[motors]
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'torso_link')
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, side+'_ankle_roll_link') for side in ('left','right')]
    if min(torso, floor, *feet) < 0:
        raise ValueError('Missing torso, floor, or foot body')
    with np.load(a.dataset/'walk_stop.npz', allow_pickle=False) as z:
        if list(z['joint_names']) != names:
            raise ValueError('Reference joint order mismatch')
        root = list(z['body_names']).index('pelvis')
        initial_q = z['joint_pos'][-1].copy(); initial_pos = z['body_pos_w'][-1, root].copy()
        quat = z['body_quat_w'][-1, root][[1,2,3,0]]
        roll,pitch,_ = Rotation.from_quat(quat).as_euler('xyz')
        initial_rot = Rotation.from_euler('xyz',[roll,pitch,0]).as_quat()[[3,0,1,2]]
    initial_pos[:2] = 0.
    policy = TorchScriptPolicy(a.policy)
    a.out.mkdir(parents=True)
    outputs = {}
    substeps = round(.02/m.opt.timestep)
    for scenario in ('direct', 'post_jog'):
        d = mujoco.MjData(m); d.qpos[:7] = np.r_[initial_pos, initial_rot]; d.qpos[qadr] = initial_q
        mujoco.mj_forward(m,d)
        goal = SkillGoal(a.dataset, a.profile, initial_pos, initial_rot)
        history = deque(maxlen=10); previous = np.zeros(n); target = initial_q.copy()
        records = []; failure = None; start_stand = 0. if scenario == 'direct' else 6.
        for step in range(round((start_stand+a.hold_seconds)/.02)):
            now = step*.02; mode = 'jog' if scenario == 'post_jog' and 3 <= now < 6 else 'stand'
            q = d.qpos[qadr].copy(); v = d.qvel[vadr].copy()
            imu = d.xquat[torso].copy(); w,x,y,z = imu
            gravity = np.array([2*(w*y-x*z), -2*(w*x+y*z), -1+2*(x*x+y*y)])
            vel = np.zeros(6); mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,torso,vel,1)
            history.append((q-default,v,vel[:3].copy(),gravity,previous.copy()))
            if len(history)==10:
                features = goal.goal(d.qpos[:3].copy(),d.qpos[3:7].copy(),q,mode)
                if abs(float(features[2*n+9])-d.qpos[2]) > .25 or abs(float(features[2*n+2])-gravity[2]) > .8:
                    failure = {'kind':'root height or tilt termination', 'time_s':now, 'root_height':float(d.qpos[2])}
                    break
                obs = tuple(x for term in range(5) for frame in history for x in frame[term])
                action = np.asarray(policy.infer(obs, features), dtype=float)
                if action.shape != (n,) or not np.isfinite(action).all():
                    raise ValueError('Invalid policy output')
                target = default + scale*action; previous = action
                goal.advance()
            for _ in range(substeps):
                torque = kp*(target-d.qpos[qadr])-kd*d.qvel[vadr]
                d.ctrl[motors] = np.clip(torque, motor_limits[:,0], motor_limits[:,1])
                mujoco.mj_step(m,d)
            # Contacts and solved forces are from the last pre-integration solve.
            loads = np.zeros(2); slip_weighted = np.zeros(2); max_slip = np.zeros(2)
            for ci in range(d.ncon):
                contact = d.contact[ci]; geom = [int(contact.geom[0]),int(contact.geom[1])]
                if floor not in geom:
                    continue
                other = geom[1] if geom[0] == floor else geom[0]
                body = int(m.geom_bodyid[other])
                if body not in feet:
                    continue
                side = feet.index(body); wrench = np.zeros(6); mujoco.mj_contactForce(m,d,ci,wrench)
                normal = max(0.,float(wrench[0])); jac = np.zeros((3,m.nv))
                mujoco.mj_jac(m,d,jac,None,contact.pos,body)
                speed = float(np.linalg.norm((jac@d.qvel)[:2])); loads[side] += normal
                slip_weighted[side] += normal*speed; max_slip[side] = max(max_slip[side],speed)
            slip = np.divide(slip_weighted,loads,out=np.zeros(2),where=loads>1e-6)
            forces = d.actuator_force[motors].copy()
            limits = np.max(np.abs(motor_limits),axis=1)
            records.append({'time_s':now,'mode':mode,'root':d.qpos[:3].tolist(),
                            'body_angular_velocity':vel[:3].tolist(), 'joint_effort':forces.tolist(),
                            'peak_effort_ratio':float(np.max(np.abs(forces)/limits)),
                            'left_right_normal_n':loads.tolist(),'loaded_slip_m_s':slip.tolist(),
                            'max_contact_speed_m_s':max_slip.tolist()})
        path = a.out/(scenario+'.jsonl')
        path.write_text(''.join(json.dumps(r)+'\n' for r in records))
        hold = [r for r in records if r['time_s'] >= start_stand+5.]
        summary = {'failure':failure, 'frames':len(records),'resets_after_initialization':0,
                   'recorded_hold_seconds_after_5s':len(hold)*.02}
        if hold:
            positions=np.array([r['root'] for r in hold]);displacement=np.linalg.norm(positions[:,:2]-positions[0,:2],axis=1)
            normal=np.array([r['left_right_normal_n'] for r in hold]);slip=np.array([r['loaded_slip_m_s'] for r in hold])
            omega=np.linalg.norm(np.array([r['body_angular_velocity'] for r in hold]),axis=1)
            summary.update(net_drift_m=float(displacement[-1]), max_drift_m=float(displacement.max()),
                           normal_p05_n=np.percentile(normal,5,axis=0).tolist(),
                           both_normal_over_10n_fraction=float(np.mean(np.all(normal>10,axis=1))),
                           loaded_slip_p95_m_s=np.percentile(slip,95,axis=0).tolist(),
                           torso_angular_speed_p95_rad_s=float(np.percentile(omega,95)),
                           peak_effort_ratio=float(max(r['peak_effort_ratio'] for r in hold)))
        outputs[scenario] = summary
    result={'scenarios':outputs,'model_sha256':sha(a.scene),'policy_sha256':sha(a.policy),
            'dataset_manifest_sha256':sha(a.dataset/'manifest.json'),
            'scope':'native MuJoCo CPU, same XML and independent PD implementation; actual solver contact forces, not official ROS simulator or hardware',
            'force_time_alignment':'last pre-integration solve, with post-integration joint and root states',
            'hold_load_threshold_n':10.}
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
