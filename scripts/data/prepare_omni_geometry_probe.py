"""Collision-constrained dense resampling; still not a dynamics certificate.

Retain source-confirmed same-surface contact proximity while correcting joint
interpolation. Source files and native model are immutable. A short window is
a diagnostic only, and full output still requires independent physical audits.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--helper',type=Path,required=True)
    p.add_argument('--start-frame',type=int,default=0,help='200 Hz diagnostic window start')
    p.add_argument('--frames',type=int,default=0,help='0 for complete clip')
    p.add_argument('--brake-frame',type=int,help='Optional input-reference frame where ascent braking starts (requires a complete 200 Hz parent)')
    p.add_argument('--stop-frame',type=int,help='Input-reference frame held after smooth braking; never counts as a different source family')
    p.add_argument('--hold-seconds',type=float,default=2.)
    p.add_argument('--task-kind',choices=['ascent_then_hold','traverse_then_hold','return_then_hold'],default='ascent_then_hold')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists() or min(a.start_frame,a.frames)<0:p.error('New output and nonnegative window required')
    import mujoco
    import numpy as np
    from scipy.interpolate import PchipInterpolator
    from scipy.spatial.transform import Rotation,Slerp
    spec=importlib.util.spec_from_file_location('omni_resample_restore_snapshot',a.helper)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    m=mujoco.MjModel.from_binary_path(str(a.model));d=mujoco.MjData(m)
    if (m.nq,m.nv,m.nu)!=(38,37,31):raise ValueError('Expected original native X2 scene')
    with np.load(a.candidate/'x2-reference.npz',allow_pickle=False) as z:
        q=z['qpos'].copy();fps=float(np.asarray(z['fps']).reshape(-1)[0])
    meta=json.loads((a.candidate/'result.json').read_text())
    warp=a.brake_frame is not None or a.stop_frame is not None
    if warp and (fps!=200. or not meta.get('complete_source_clip') or a.brake_frame is None or a.stop_frame is None
        or not 0<=a.brake_frame<a.stop_frame<len(q) or a.hold_seconds<2 or a.start_frame or a.frames):
        raise ValueError('Braking requires a complete 200 Hz parent, ordered input frames and at least two seconds holding')
    if a.task_kind in ['traverse_then_hold','return_then_hold'] and (not warp or a.stop_frame!=len(q)-1):
        raise ValueError('Traverse/landing must retain the complete source interval before holding')
    duration=(len(q)-1)/fps;steps=int(np.ceil(duration*50))*4
    times=np.arange(len(q))/fps;query=np.linspace(0,duration,steps+1)
    time_map=None
    if warp:
        begin=a.brake_frame/fps;stop=a.stop_frame/fps;braking=2*(stop-begin)
        # Pad the stationary tail to a control tick. This preserves the ascent
        # clock when exporting 50 Hz goals instead of stretching its prefix.
        clock=np.arange(int(np.ceil((begin+braking+a.hold_seconds)*50))*4+1)/200.
        u=np.clip((clock-begin)/braking,0.,1.)
        slowed=begin+braking*(u-2.5*u**4+3*u**5-u**6)
        query=np.where(clock<=begin,clock,np.where(clock>=begin+braking,stop,slowed))
        if np.any(np.diff(query)<-1e-12):raise ValueError('Source-time mapping must remain monotone')
        time_map={'kind':a.task_kind,'input_fps':fps,'brake_frame':a.brake_frame,'stop_frame':a.stop_frame,
            'braking_duration_s':braking,'hold_seconds':float(clock[-1]-begin-braking),'requested_hold_seconds':a.hold_seconds,
            'description':'Source-time velocity smoothly falls from 1 to 0 with zero acceleration at both ends; original ascent prefix retained'}
    cols=np.r_[np.arange(3),np.arange(7,38)];dense=np.zeros((len(query),38))
    dense[:,cols]=PchipInterpolator(times,q[:,cols],axis=0)(query)
    dense[:,3:7]=Slerp(times,Rotation.from_quat(q[:,[4,5,6,3]]))(query).as_quat()[:,[3,0,1,2]]
    end=min(len(dense),a.start_frame+a.frames) if a.frames else len(dense)
    if a.start_frame>=end:raise ValueError('Empty source window')
    mid=m.jnt_range[1:].mean(1);half=np.diff(m.jnt_range[1:],axis=1).ravel()*meta['joint_limit_factor']/2
    lower=mid-half;upper=mid+half
    if meta['wrist_reference_limit_deg']:
        for j in range(31):
            if 'wrist' in m.joint(j+1).name:
                # Stay inside the already validated wrist task envelope.
                lower[j]=max(lower[j],q[:,j+7].min()-1e-9)
                upper[j]=min(upper[j],q[:,j+7].max()+1e-9)
    root=int(m.jnt_bodyid[0])
    robot=[g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]]==root and (m.geom_contype[g] or m.geom_conaffinity[g])]
    terrain=[g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]]!=root and (m.geom_contype[g] or m.geom_conaffinity[g])]
    pairs=[(g,o) for g in robot for o in terrain];witness=np.zeros(6)
    source_contacts=[]
    for pose in q:
        d.qpos[:]=pose;mujoco.mj_forward(m,d);contacts={}
        for g,o in pairs:
            gap=float(mujoco.mj_geomDistance(m,d,g,o,.01,witness))
            if abs(gap)<=.001 and abs(gap)>1e-10:
                contacts[(g,o)]=(witness[:3]-witness[3:]).copy()/gap
        source_contacts.append(contacts)
    a.out.mkdir(parents=True)
    (a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    (a.out/'restore-as-run.py').write_bytes(a.helper.read_bytes())
    (a.out/'original-adapter-result.json').write_bytes((a.candidate/'result.json').read_bytes())
    if time_map is not None:
        np.savez_compressed(a.out/'source-time-map.npz',source_time_s=query,fps=np.array([200.]))
        (a.out/'task-definition.json').write_text(json.dumps(time_map,indent=2))
    rows=[];accepted=[]
    for index in range(a.start_frame,end):
        pose=dense[index].copy();source_index=min(int(np.searchsorted(times,query[index],side='right')-1),len(q)-2)
        before=source_contacts[source_index];after=source_contacts[source_index+1]
        contact_pairs=[pair for pair in before.keys()&after.keys() if before[pair]@after[pair]>.99]
        d.qpos[:]=pose;mujoco.mj_forward(m,d)
        distances=np.array([mujoco.mj_geomDistance(m,d,g,o,.02,witness) for g,o in pairs])
        gaps=np.array([mujoco.mj_geomDistance(m,d,g,o,10.,witness) for g,o in contact_pairs])
        info=None
        if distances.min()<-.00085 or (len(gaps) and gaps.max()>.001):
            pose,info=helper.restore_pose(m,pose,{}, {},lower,upper,surface_pairs=contact_pairs,
                robot_geometry=robot,terrain_geometry=terrain,translation_domain=.02,rotation_domain=.05,max_nfev=60,
                surface_gap_upper=.000999)
        d.qpos[:]=pose;mujoco.mj_forward(m,d)
        penetration=max(0.,-min(float(mujoco.mj_geomDistance(m,d,g,o,.02,witness)) for g,o in pairs))
        contact_gap=max([0.]+[float(mujoco.mj_geomDistance(m,d,g,o,10.,witness)) for g,o in contact_pairs])
        soft=float(np.maximum(np.maximum(lower-pose[7:],pose[7:]-upper),0).max())
        # The optimizer aims for 0.8 mm. Its internal target is distinct from
        # the unchanged independently checked 1 mm acceptance threshold.
        if penetration>.001 or contact_gap>.001 or soft>1e-8:
            (a.out/'failure.json').write_text(json.dumps({'index':index,'restoration':info,
                'penetration_m':penetration,'contact_gap_m':contact_gap,'task_joint_violation_rad':soft},indent=2))
            np.savez_compressed(a.out/'failed-pose.npz',qpos=pose,original=dense[index])
            raise RuntimeError(f'Independent resampling constraints rejected frame {index}')
        accepted.append(pose);rows.append({'index':index,'penetration_m':penetration,'contact_pairs':len(contact_pairs),
            'source_contact_gap_m':contact_gap,'task_joint_violation_rad':soft,
            'maximum_joint_correction_rad':float(abs(pose[7:]-dense[index,7:]).max()),'restoration':info})
        if index%100==0:print(json.dumps({k:v for k,v in rows[-1].items() if k!='restoration'}),flush=True)
    accepted=np.asarray(accepted)
    np.savez_compressed(a.out/'x2-reference.npz',qpos=accepted,fps=np.array([200.]),dense_window_start_frame=a.start_frame)
    complete=a.start_frame==0 and end==len(dense)
    result={**{k:meta[k] for k in ['source_motion','source_start_frame','scene_boxes','joint_limit_factor','wrist_reference_limit_deg']},
        'frames':len(accepted),'fps':200.,'dense_window_start_frame':a.start_frame,'dense_window_start_time_s':a.start_frame/200.,
        'source_reference_sha256':hashlib.sha256((a.candidate/'x2-reference.npz').read_bytes()).hexdigest(),
        'source_reference_path':str((a.candidate/'x2-reference.npz').resolve()),
        'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),'restoration_helper_sha256':hashlib.sha256(a.helper.read_bytes()).hexdigest(),
        'uniform_time_scale':None if warp else (steps/200.)/duration,'complete_source_clip':complete and not warp,
        'task_variant':time_map,'complete_source_prefix':bool(complete and warp),
        'max_environment_penetration_m':max(row['penetration_m'] for row in rows),
        'maximum_joint_correction_rad':max(row['maximum_joint_correction_rad'] for row in rows),
        'corrected_frames':sum(row['restoration'] is not None for row in rows),'rows':rows,
        'policy_evaluation':False,'training_approved':False,'scope':__doc__}
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
