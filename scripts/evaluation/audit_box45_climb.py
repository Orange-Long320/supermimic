"""Audit a continuous native climb trace and its final five-second standing hold.

Run on the simulation server. This certifies only the recorded controller trial,
never a learned policy or an unexecuted reference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--box',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--standing-auditor',type=Path,default=Path(__file__).with_name('audit_standing_tracker.py'))
    a=p.parse_args()
    if a.out.exists():p.error('output exists')
    import mujoco
    import numpy as np
    from supermimic.robots.dimensions import PROFILE
    from supermimic.envs.box_scene import FixedBox
    report=json.loads((a.trace/'result.json').read_text());cfg=report['configuration']
    box=FixedBox(**json.loads(a.box.read_text())['box'])
    model_path=Path(cfg['model']);m=mujoco.MjModel.from_binary_path(str(model_path));d=mujoco.MjData(m)
    z=np.load(a.trace/'trajectory.npz');states=z['state'];controls=z['controls'];dt=float(z['dt'])
    if len(states)!=len(controls)+1 or states.shape[1]!=m.nq+m.nv:raise ValueError('Trace dimensions differ')
    root=int(m.jnt_bodyid[0]);feet={m.body('robot/'+s+'_ankle_roll_link').id for s in ('left','right')}
    robot=[g for g in range(m.ngeom) if m.body_rootid[int(m.geom_bodyid[g])]==root and (m.geom_contype[g] or m.geom_conaffinity[g])]
    obstacles=[m.geom('box45_collision').id]+[g for g in range(m.ngeom) if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE]
    effort=np.array([PROFILE['effort_limit'][n] for n in PROFILE['joint_names']]);speed=np.array([PROFILE['velocity_limit'][n] for n in PROFILE['joint_names']])
    lo,hi=m.jnt_range[1:].T;witness=np.zeros(6)
    maximum={'hard_violation_rad':0.,'joint_speed_ratio':0.,'motor_ratio':0.,'foot_penetration_m':0.,'nonfoot_penetration_m':0.}
    entry=[]
    for i,state in enumerate(states):
        d.qpos[:]=state[:m.nq];d.qvel[:]=state[m.nq:];d.ctrl[:]=controls[max(0,i-1)] if len(controls) else 0;mujoco.mj_forward(m,d)
        maximum['hard_violation_rad']=max(maximum['hard_violation_rad'],float(np.maximum(np.maximum(lo-d.qpos[7:],d.qpos[7:]-hi),0).max()))
        maximum['joint_speed_ratio']=max(maximum['joint_speed_ratio'],float((abs(d.qvel[6:])/speed).max()))
        maximum['motor_ratio']=max(maximum['motor_ratio'],float((abs(d.actuator_force)/effort).max()))
        for g in robot:
            foot=int(m.geom_bodyid[g]) in feet
            key='foot_penetration_m' if foot else 'nonfoot_penetration_m'
            for obstacle in obstacles:
                maximum[key]=max(maximum[key],-float(mujoco.mj_geomDistance(m,d,g,obstacle,.02,witness)))
            if i==0 and foot:
                mesh=int(m.geom_dataid[g])
                if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX:
                    local=np.array([(x,y,z) for x in (-1,1) for y in (-1,1) for z in (-1,1)])*m.geom_size[g]
                else:
                    adr,count=m.mesh_vertadr[mesh],m.mesh_vertnum[mesh];local=m.mesh_vert[adr:adr+count]
                world=local@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
                entry.append({'body':int(m.geom_bodyid[g]),'minimum_z':float(world[:,2].min()),'toe_x':float(world[:,0].max())})
        if i%500==0:print(f'audited={i}/{len(states)}',flush=True)
    a.out.mkdir(parents=True)
    tail_count=round(5/dt);tail=a.out/'tail-trace';tail.mkdir()
    np.savez_compressed(tail/'trajectory.npz',state=states[-tail_count-1:],controls=controls[-tail_count:],dt=dt)
    (tail/'result.json').write_text(json.dumps(report))
    # Reuse the established standing gates; their trace is an explicitly saved
    # suffix of this execution, not an independently initialized simulation.
    subprocess.run([sys.executable,str(a.standing_auditor),'--trace',str(tail),'--support-height',str(box.height),'--box',str(a.box),'--out',str(a.out/'tail-audit')],check=True)
    steady=json.loads((a.out/'tail-audit/result.json').read_text())
    entered_on_floor=all(any(x['body']==b and abs(x['minimum_z'])<.005 and x['toe_x']<=box.front_x for x in entry) for b in feet)
    physical=maximum['hard_violation_rad']<1e-4 and maximum['joint_speed_ratio']<=1 and maximum['motor_ratio']<=1.00001 and maximum['foot_penetration_m']<=.005 and maximum['nonfoot_penetration_m']<=.001
    reference=np.load(cfg['reference']);fps=float(reference['fps'].reshape(-1)[0]);duration=(len(reference['qpos'])-1)/fps
    complete=report['status']=='completed' and len(controls)*dt>=duration+2-1e-6 and int(cfg['start_frame'])==0 and report['state_resets_after_entry']==0
    result={'continuous_climb_passed':bool(entered_on_floor and physical and complete and steady['standing_hold_passed']),
        'entry_on_floor_before_box':entered_on_floor,'complete_reference_and_extra_hold':complete,'physical_passed':bool(physical),'maximum':maximum,
        'final_five_seconds_standing_passed':steady['standing_hold_passed'],'seconds':len(controls)*dt,
        'trace':str(a.trace),'model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'trajectory_sha256':hashlib.sha256((a.trace/'trajectory.npz').read_bytes()).hexdigest(),
        'policy_evaluation':False,'approved_for_training':False,'scope':'Recorded native controller execution only; learned policy remains unverified.'}
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');(a.out/'as-run.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(result),flush=True)


if __name__=='__main__':main()
