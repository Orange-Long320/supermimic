"""Independent state-aligned physical audit of a recorded native-control trace."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    from supermimic.data.design_metrics import BatchedDesignMetrics
    from supermimic.robots.dimensions import PROFILE
    m=mujoco.MjModel.from_binary_path(str(args.trace/'native-model.mjb'));d=mujoco.MjData(m)
    with np.load(args.trace/'trajectory.npz') as z:
        states=z['state'].copy();controls=z['servo_target'].copy();dt=1/float(z['fps'][0])
        recorded=z['metrics'].copy()
        accelerations=z['predicted_and_actual_acceleration'].copy() if 'predicted_and_actual_acceleration' in z else None
    with np.load(args.trace/'initial-state.npz') as z:initial=z['state'][0].copy()
    if states.ndim!=2 or states.shape[1]!=m.nq+m.nv:raise ValueError('Expected a single canonical-state trace')
    names=list(PROFILE['joint_names'])
    assert [m.joint(i).name.rsplit('/',1)[-1] for i in range(1,m.njnt)]==names
    assert [m.joint(int(i)).name.rsplit('/',1)[-1] for i in m.actuator_trnid[:,0]]==names
    kp=np.array([PROFILE['stiffness'][n] for n in names]);kd=np.array([PROFILE['damping'][n] for n in names])
    effort=np.array([PROFILE['effort_limit'][n] for n in names]);limits=np.array([PROFILE['velocity_limit'][n] for n in names])
    np.testing.assert_allclose(m.actuator_gainprm[:,0],kp)
    np.testing.assert_allclose(m.actuator_biasprm[:,1],-kp);np.testing.assert_allclose(m.actuator_biasprm[:,2],-kd)
    assert np.all(m.actuator_forcelimited) and np.allclose(m.actuator_forcerange,np.c_[-effort,effort])
    measure=BatchedDesignMetrics(m,'cpu');rows=[];rotations=[];angle=0.
    previous=Rotation.from_quat(initial[3:7][[1,2,3,0]])
    for state in states:
        d.qpos[:]=state[:m.nq];d.qvel[:]=state[m.nq:];mujoco.mj_forward(m,d)
        tensors={key:torch.as_tensor(np.array(getattr(d,key)),dtype=torch.float32)[None]
                 for key in ['xpos','xquat','cvel','geom_xpos','geom_xmat','qvel']}
        rows.append(measure(SimpleNamespace(**tensors))[0].numpy())
        current=Rotation.from_quat(state[3:7][[1,2,3,0]])
        angle+=(current*previous.inv()).as_rotvec()[1];rotations.append(angle);previous=current
    rows=np.asarray(rows);rotations=np.asarray(rotations);lo,hi=m.jnt_range[1:].T
    pre=np.r_[initial[None],states[:-1]]
    integration_diagnostic=None
    if accelerations is not None:
        from supermimic.policy.native_inverse import implicitfast_acceleration_map
        observed=(states[:,m.nq:]-pre[:,m.nq:])/dt;estimated=[];reported=[]
        for i,state in enumerate(pre):
            d.qpos[:]=state[:m.nq];d.qvel[:]=state[m.nq:];mujoco.mj_forward(m,d)
            transform=implicitfast_acceleration_map(m,d)
            estimated.append(np.linalg.solve(transform,accelerations[i,0]))
            reported.append(np.linalg.solve(transform,accelerations[i,1]))
        summarize=lambda e:{'maximum_absolute_error':float(np.nanmax(abs(e))),'rms_error':float(np.sqrt(np.nanmean(e*e)))}
        integration_diagnostic={'instantaneous_qacc_vs_observed_dv_dt':summarize(accelerations[:,1]-observed),
            'integrated_actual_qacc_vs_observed_dv_dt':summarize(np.asarray(reported)-observed),
            'integrated_inverse_prediction_vs_observed_dv_dt':summarize(np.asarray(estimated)-observed)}
    forces=np.clip(kp*(controls-pre[:,7:m.nq])-kd*pre[:,m.nq+6:],-effort,effort)
    hard=np.maximum(np.maximum(lo-states[:,7:m.nq],states[:,7:m.nq]-hi),0).max(axis=1)
    speed=(abs(states[:,m.nq+6:])/limits).max(axis=1);motor=(abs(forces)/effort).max(axis=1)
    def physical(end):
        sl=slice(0,end+1);feet=rows[sl,7:9].min(axis=0);nonfoot=float(rows[sl,9].min())
        values={'hard_violation_rad':float(hard[sl].max()),'joint_speed_ratio':float(speed[sl].max()),
                'motor_ratio':float(motor[sl].max()),'minimum_foot_heights_m':feet.tolist(),'minimum_nonfoot_height_m':nonfoot}
        values['passed']=bool(values['hard_violation_rad']<1e-4 and values['joint_speed_ratio']<=1 and
                              values['motor_ratio']<=1.00001 and feet.min()>-.005 and nonfoot>.005)
        return values
    stride=round(.02/dt)
    if abs(stride*dt-.02)>1e-8:raise ValueError('Trace rate must divide the canonical 20 ms clock')
    indices=np.arange(stride-1,len(rows),stride);sampled=rows[indices];takeoff=None
    positive=(sampled[:,7]>.02)&(sampled[:,8]>.02)&(sampled[:,9]>.005)&(sampled[:,1]>0)
    for j in range(2,len(sampled)):
        if not positive[j-2:j+1].all():continue
        first=j-2;window=sampled[max(0,first-8):first];support=np.flatnonzero(window[:,7:9].min(axis=1)<.01)
        single=bool(len(window)>=4 and (window[:,8]>.04).all() and ((window[:,7]<.01)&(window[:,8]>.04)).sum()>=4 and
                    len(support) and window[support[-1],7]<.01 and window[support[-1],8]>.04)
        end=int(indices[j]);event=sampled[j];bounds=physical(end)
        takeoff={'event_index':end,'time_s':(end+1)*dt,'vz_m_s':float(event[1]),'momentum':event[2:5].tolist(),
                 'single_left_support':single,'physical_prefix':bounds,
                 'passed':bool(single and bounds['passed'] and event[1]>1.5 and event[3]>20)}
        break
    airborne=(rows[:,7:9].min(axis=1)>.02);air_ids=np.flatnonzero(airborne);first_air_rotation=0.
    if len(air_ids):
        start=air_ids[0];end=start
        while end+1<len(rows) and airborne[end+1]:end+=1
        first_air_rotation=float(rotations[end]-rotations[start])
    tail=rows[-max(1,round(.2/dt)):]
    landing=bool((tail[:,11]>.85).all() and (tail[:,0]>.55).all() and (tail[:,5]<.5).all() and
                 (tail[:,6]<1).all() and (tail[:,7:9].min(axis=1)<.01).all())
    full=physical(len(rows)-1)
    complete=bool(takeoff and takeoff['passed'] and full['passed'] and rotations[-1]>5.5 and
                  first_air_rotation>3.5 and rows[:,11].min()<-.7 and landing)
    peak=int(rows[:,1].argmax())
    result={'integration_diagnostic':integration_diagnostic,'trace':str(args.trace),'metric_state_alignment':'Independent mj_forward from every recorded qpos/qvel',
            'model_sha256':hashlib.sha256((args.trace/'native-model.mjb').read_bytes()).hexdigest(),
            'trajectory_sha256':hashlib.sha256((args.trace/'trajectory.npz').read_bytes()).hexdigest(),
            'physical_steps':len(rows),'seconds':len(rows)*dt,'takeoff':takeoff,'full_physical':full,
            'total_forward_rotation_rad':float(rotations[-1]),'first_air_rotation_rad':first_air_rotation,
            'landing_passed':landing,'full_motion_passed':complete,'approved_for_training':False,
            'peak_vz':{'time_s':(peak+1)*dt,'vz_m_s':float(rows[peak,1]),'momentum':rows[peak,2:5].tolist()},
            'maximum_recorded_metric_difference':abs(rows-recorded[:,:12]).max(axis=0).tolist()}
    np.savez(args.out/'state-aligned-metrics.npz',metrics=rows,rotation=rotations,hard_violation=hard,joint_speed_ratio=speed,computed_motor_force=forces)
    (args.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    (args.out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
