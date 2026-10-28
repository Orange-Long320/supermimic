"""Independently replay recorded native commands and audit actual geometry.

This verifies execution evidence; it does not certify a complete climb or a
learned policy. Only the initial state is assigned during command replay.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('Independent output must be new')
    import mujoco
    import numpy as np
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra',check_assets=True)
    from supermimic.robots.dimensions import PROFILE
    result=json.loads((a.trace/'result.json').read_text())
    model_path=Path(result['configuration']['model'])
    if hashlib.sha256(model_path.read_bytes()).hexdigest()!=result['model_sha256']:
        raise ValueError('Native model changed after execution')
    m=mujoco.MjModel.from_binary_path(str(model_path));d=mujoco.MjData(m);record=mujoco.MjData(m)
    with np.load(a.trace/'trajectory.npz',allow_pickle=False) as z:
        state=z['state'].copy();controls=z['controls'].copy();dt=float(z['dt'])
    if (state.shape!=(len(controls)+1,m.nq+m.nv) or controls.shape[1:]!=(m.nu,)
        or not np.isfinite(state).all() or not np.isfinite(controls).all() or dt!=float(m.opt.timestep)):
        raise ValueError('Invalid state, command or timestep layout')
    names=[m.joint(i).name.removeprefix('robot/') for i in range(1,m.njnt)]
    if names!=PROFILE['joint_names']:raise ValueError('Unexpected joint order')
    effort=np.array([PROFILE['effort_limit'][n] for n in names])
    speed=np.array([PROFILE['velocity_limit'][n] for n in names])
    root=int(m.jnt_bodyid[0]);robot=[g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]]==root and (m.geom_contype[g] or m.geom_conaffinity[g])]
    terrain=[g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]]!=root and (m.geom_contype[g] or m.geom_conaffinity[g])]
    columns=m.jnt_qposadr[m.actuator_trnid[:,0]];vcols=m.jnt_dofadr[m.actuator_trnid[:,0]]
    d.qpos[:]=state[0,:m.nq];d.qvel[:]=state[0,m.nq:]
    d.ctrl[:]=d.qpos[columns]-m.actuator_biasprm[:,2]*d.qvel[vcols]/m.actuator_gainprm[:,0]
    mujoco.mj_forward(m,d)
    replay_q=0.;replay_v=0.;max_motor=0.;max_hard=0.;max_speed=0.;max_pen=0.;max_self=0.
    external_contact_frames={};rows=[];jac=np.zeros((3,m.nv));witness=np.zeros(6)
    for i,s in enumerate(state):
        record.qpos[:]=s[:m.nq];record.qvel[:]=s[m.nq:];mujoco.mj_forward(m,record)
        max_hard=max(max_hard,float(np.maximum(np.maximum(m.jnt_range[1:,0]-s[7:m.nq],s[7:m.nq]-m.jnt_range[1:,1]),0.).max()))
        max_speed=max(max_speed,float((abs(s[m.nq+6:])/speed).max()))
        contacts=set()
        for g in robot:
            for o in terrain:
                max_pen=max(max_pen,-float(mujoco.mj_geomDistance(m,record,g,o,.02,witness)))
        for c in record.contact:
            if c.dist>0:continue
            if int(c.geom1) in robot and int(c.geom2) in robot:max_self=max(max_self,-float(c.dist))
            bodies=[int(m.geom_bodyid[g]) for g in c.geom]
            if any(m.body_rootid[b]!=root for b in bodies):
                contacts.update(m.body(b).name for b in bodies if m.body_rootid[b]==root)
        for body in contacts:external_contact_frames[body]=external_contact_frames.get(body,0)+1
        mujoco.mj_jacSubtreeCom(m,record,jac,root)
        rows.append({'t':i*dt,'root_position':s[:3].tolist(),'root_up_z':float(record.xmat[root].reshape(3,3)[2,2]),
            'com_position':record.subtree_com[root].tolist(),'com_speed':float(np.linalg.norm(jac@record.qvel)),
            'external_contact_bodies':sorted(contacts)})
        if i==len(controls):break
        # Match the original driver's forward/step ordering; no later resets.
        mujoco.mj_forward(m,d);d.ctrl[:]=controls[i];mujoco.mj_forward(m,d)
        max_motor=max(max_motor,float((abs(d.actuator_force)/effort).max()))
        mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
        max_motor=max(max_motor,float((abs(d.actuator_force)/effort).max()))
        replay_q=max(replay_q,float(abs(d.qpos-state[i+1,:m.nq]).max()))
        replay_v=max(replay_v,float(abs(d.qvel-state[i+1,m.nq:]).max()))
    terminal=rows[max(0,len(rows)-round(.5/dt)-1):]
    positions=np.array([r['root_position'] for r in terminal])
    report={'recorded_status':result['status'],'seconds_executed':len(controls)*dt,'state_samples':len(state),
        'maximum_replayed_position_error':replay_q,'maximum_replayed_velocity_error':replay_v,
        'deterministic_replay_passed':bool(replay_q<=1e-6 and replay_v<=1e-5),
        'maximum_motor_ratio':max_motor,'hard_joint_violation_rad':max_hard,'joint_speed_ratio':max_speed,
        'maximum_environment_penetration_m':max_pen,'maximum_enabled_self_penetration_m':max_self,
        'geometry_1mm_passed':bool(max_pen<=.001 and max_self<=.001),
        'actuation_and_limits_passed':bool(max_motor<=1.00001 and max_hard<=1e-6 and max_speed<=1.),
        'terminal_half_second_max_com_speed':max(r['com_speed'] for r in terminal),
        'terminal_half_second_root_drift_m':float(np.linalg.norm(positions-positions[0],axis=1).max()),
        'minimum_root_up_z':min(r['root_up_z'] for r in rows),'external_contact_frame_counts':external_contact_frames,
        'external_applied_force_max':float(max(abs(d.qfrc_applied).max(),abs(d.xfrc_applied).max())),
        'model_sha256':result['model_sha256'],'trajectory_sha256':hashlib.sha256((a.trace/'trajectory.npz').read_bytes()).hexdigest(),
        'complete_climb_verified':False,'training_approved':False,'scope':__doc__,'rows':rows}
    a.out.mkdir(parents=True);(a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
