"""Independently audit native tracker states, foot loads and steady standing."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--support-height',type=float,default=0.,help='Horizontal support plane height for relative clearances')
    p.add_argument('--box',type=Path,help='Box fit/placement JSON for full foot-hull containment')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    trace_report=json.loads((args.trace/'result.json').read_text())
    model_path=Path(trace_report['configuration']['model'])
    m=mujoco.MjModel.from_binary_path(str(model_path));d=mujoco.MjData(m)
    z=np.load(args.trace/'trajectory.npz');states=z['state'];controls=z['controls'];dt=float(z['dt'])
    if len(states)!=len(controls)+1 or states.shape[1]!=m.nq+m.nv:raise ValueError('Invalid native tracker state trace')
    from supermimic.robots.dimensions import PROFILE
    effort=np.array([PROFILE['effort_limit'][n] for n in PROFILE['joint_names']])
    speed=np.array([PROFILE['velocity_limit'][n] for n in PROFILE['joint_names']])
    root=int(m.jnt_bodyid[0]);feet={s:m.body('robot/'+s+'_ankle_roll_link').id for s in ['left','right']}
    box=None
    if args.box:
        from supermimic.envs.box_scene import FixedBox
        box=FixedBox(**json.loads(args.box.read_text())['box'])
    hulls_inside=True
    meshes=[]
    for g in range(m.ngeom):
        if m.body_rootid[int(m.geom_bodyid[g])]!=root or not(m.geom_contype[g] or m.geom_conaffinity[g]):continue
        if int(m.geom_type[g])==int(mujoco.mjtGeom.mjGEOM_BOX):
            v=np.array([[x,y,z] for x in [-1.,1.] for y in [-1.,1.] for z in [-1.,1.]])*m.geom_size[g]
        else:
            k=m.geom_dataid[g]
            if k<0:raise ValueError('Unsupported robot collision geometry')
            a,n=m.mesh_vertadr[k],m.mesh_vertnum[k];v=m.mesh_vert[a:a+n].astype(float)
        group=next((s for s,b in feet.items() if m.geom_bodyid[g]==b),'nonfoot')
        meshes.append((g,v,group))
    jac=np.zeros((3,m.nv));rows=[];lo,hi=m.jnt_range[1:].T
    for i,state in enumerate(states[1:]):
        d.qpos[:]=state[:m.nq];d.qvel[:]=state[m.nq:];d.ctrl[:]=controls[i];mujoco.mj_forward(m,d)
        mujoco.mj_jacSubtreeCom(m,d,jac,root)
        heights={s:float('inf') for s in ['left','right','nonfoot']};sole_tilt={s:[] for s in feet}
        for g,v,group in meshes:
            w=v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g];heights[group]=min(heights[group],float(w[:,2].min()-args.support_height))
            if group in feet and box is not None:
                hulls_inside &= all(box.top_contains(float(x),float(y)) for x,y in w[:,:2])
            if group in feet and int(m.geom_type[g])==int(mujoco.mjtGeom.mjGEOM_BOX):
                sole_tilt[group].append(float(np.arccos(np.clip(d.geom_xmat[g].reshape(3,3)[2,2],-1,1))))
        loads={s:0. for s in feet};points={s:[] for s in feet}
        for c_id,c in enumerate(d.contact[:d.ncon]):
            bodies=[int(m.geom_bodyid[int(g)]) for g in c.geom]
            if all(m.body_rootid[b]==root for b in bodies):continue
            force=np.zeros(6);mujoco.mj_contactForce(m,d,c_id,force)
            for side,body in feet.items():
                if body in bodies:
                    loads[side]+=float(max(force[0],0))
                    if force[0]>1.:points[side].append(c.pos[:2].copy())
        areas={}
        for side,pts in points.items():
            try:areas[side]=float(ConvexHull(np.asarray(pts)).volume) if len(pts)>=3 else 0.
            except Exception:areas[side]=0.
        rows.append({'time_s':(i+1)*dt,'com_z':float(d.subtree_com[root,2]),
            'com_speed':float(np.linalg.norm(jac@d.qvel)),'root_angular_speed':float(np.linalg.norm(d.qvel[3:6])),
            'root_up_z':float(d.xmat[root].reshape(3,3)[2,2]),
            'hard_violation':float(np.maximum(np.maximum(lo-d.qpos[7:],d.qpos[7:]-hi),0).max()),
            'joint_speed_ratio':float((abs(d.qvel[6:])/speed).max()),
            'motor_ratio':float((abs(d.actuator_force)/effort).max()),'heights':heights,'foot_loads_n':loads,
            'loaded_contact_area_m2':areas,'sole_tilt_rad':{s:max(v) if v else None for s,v in sole_tilt.items()}})
    tail=rows[-round(1/dt):]
    physical={'maximum_hard_violation_rad':max((x['hard_violation'] for x in rows),default=0),
        'maximum_joint_speed_ratio':max((x['joint_speed_ratio'] for x in rows),default=0),
        'maximum_motor_ratio':max((x['motor_ratio'] for x in rows),default=0),
        'minimum_foot_height_m':min((min(x['heights'][s] for s in feet) for x in rows),default=0),
        'minimum_nonfoot_height_m':min((x['heights']['nonfoot'] for x in rows),default=0)}
    steady={'max_com_speed_m_s':max((x['com_speed'] for x in tail),default=float('inf')),
        'max_root_angular_speed_rad_s':max((x['root_angular_speed'] for x in tail),default=float('inf')),
        'min_root_up_z':min((x['root_up_z'] for x in tail),default=0),
        'min_com_height_m':min((x['com_z']-args.support_height for x in tail),default=0),
        'minimum_bilateral_load_n':min((min(x['foot_loads_n'].values()) for x in tail),default=0),
        'minimum_loaded_contact_area_m2':min((min(x['loaded_contact_area_m2'].values()) for x in tail),default=0),
        'maximum_sole_tilt_rad':max((max(x['sole_tilt_rad'].values()) for x in tail),default=None)
            if tail and all(all(v is not None for v in x['sole_tilt_rad'].values()) for x in tail) else None}
    passed_physical=bool(physical['maximum_hard_violation_rad']<1e-4 and physical['maximum_joint_speed_ratio']<=1
        and physical['maximum_motor_ratio']<=1.00001 and physical['minimum_foot_height_m']>-.005 and physical['minimum_nonfoot_height_m']>.005 and hulls_inside)
    passed_steady=bool(len(tail)==round(1/dt) and steady['max_com_speed_m_s']<.02 and
        steady['max_root_angular_speed_rad_s']<.05 and steady['min_root_up_z']>.98 and steady['min_com_height_m']>.55
        and steady['minimum_bilateral_load_n']>10 and steady['minimum_loaded_contact_area_m2']>1e-4
        and steady['maximum_sole_tilt_rad'] is not None and steady['maximum_sole_tilt_rad']<np.radians(2))
    result={'trace':str(args.trace),'model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'trajectory_sha256':hashlib.sha256((args.trace/'trajectory.npz').read_bytes()).hexdigest(),
        'seconds':len(rows)*dt,'controller_status':trace_report['status'],'physical':physical,'steady_last_second':steady,
        'physical_passed':passed_physical,'steady_passed':passed_steady,
        'support_height_m':args.support_height,'foot_hulls_inside_box':bool(hulls_inside) if box is not None else None,
        'standing_hold_passed':bool(trace_report['status']=='completed' and len(rows)*dt>=5 and passed_physical and passed_steady),
        'policy_evaluation':False,'full_webster_verified':False,'approved_for_training':False,
        'scope':'Audit of the given initialized native tracker segment; does not certify the unexecuted prefix or policy learning.'}
    (args.out/'result.json').write_text(json.dumps(result,indent=2));(args.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
