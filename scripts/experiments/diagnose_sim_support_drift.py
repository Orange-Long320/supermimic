"""Diagnose simulated support geometry and velocity against odometry (scoring only)."""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def stamps(data):
    return np.array([r['stamp'][0] + r['stamp'][1]*1e-9 for r in data])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--src', type=Path, required=True)
    p.add_argument('--xml', type=Path, required=True)
    p.add_argument('--scene', type=Path, help='independent simulated contact reconstruction for diagnosis only')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); sys.path.insert(0, str(a.src))
    from supermimic.deploy.sim_foot_support import SimFootSupport
    if a.out.exists():
        raise ValueError('output exists')
    imu = rows(a.run/'trial-record/imu.jsonl')
    joints = [r for r in rows(a.run/'trial-record/joints.jsonl') if r['group'] == 'leg']
    odom = rows(a.run/'trial-record/odom.jsonl')
    it, jt, tt = map(stamps, (imu, joints, odom))
    truth = np.array([x['position'] for x in odom])
    support = SimFootSupport(a.xml); obs=[]; prev=-np.inf
    contact_counts = {'accepted': 0, 'selected_in_contact': 0, 'any_foot_contact': 0, 'no_foot_contact': 0}
    if a.scene:
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(a.scene)); data = mujoco.MjData(model)
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        foot_bodies = {model.body(side + '_ankle_roll_link').id for side in ('left','right')}

    for sample, t in zip(imu, it):
        if t < tt[0]+.1 or t > tt[-1]-.1 or t-prev < .02:
            continue
        prev=t; idx=np.searchsorted(jt,t,side='right')-1
        if idx < 0 or t-jt[idx] > .01:
            continue
        q={n:v+(t-jt[idx])*d for n,v,d,_ in joints[idx]['joints']}
        qd={n:d for n,v,d,_ in joints[idx]['joints']}
        r=Rotation.from_quat(sample['q_xyzw']).as_matrix()
        candidates=support.candidates(r,sample['w'],q,qd)
        side=min(candidates,key=lambda s:candidates[s][0])
        v,reason=support.observe(t,r,sample['w'],q,qd)
        if v is None:
            continue
        vtrue=np.array([(np.interp(t+.05,tt,truth[:,k])-np.interp(t-.05,tt,truth[:,k]))/.1 for k in range(3)])
        ptruth=np.array([np.interp(t,tt,truth[:,k]) for k in range(3)])
        height=-candidates[side][0]
        obs.append([t,*v,*vtrue,height,*ptruth])
        if a.scene:
            # Reconstruct collisions from recorded odom pose and encoder samples;
            # this is an oracle diagnostic, never input to the estimator.
            data.qpos[:] = model.qpos0
            data.qpos[:7] = np.r_[ptruth, np.asarray(sample['q_xyzw'])[[3,0,1,2]]]
            for name, value in q.items():
                jid = model.joint(name).id
                data.qpos[model.jnt_qposadr[jid]] = value
            mujoco.mj_forward(model, data)
            foot_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] in foot_bodies
                          and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE and model.geom_contype[g]]
            selected = min(foot_geoms, key=lambda g: data.geom_xpos[g,2]-model.geom_size[g,0])
            contacts = set()
            for i in range(data.ncon):
                c = data.contact[i]
                if c.dist <= 0. and (c.geom1 == floor or c.geom2 == floor):
                    contacts.add(c.geom2 if c.geom1 == floor else c.geom1)
            contact_counts['accepted'] += 1
            contact_counts['selected_in_contact'] += int(selected in contacts)
            any_foot = bool(set(foot_geoms) & contacts)
            contact_counts['any_foot_contact'] += int(any_foot)
            contact_counts['no_foot_contact'] += int(not any_foot)
    x=np.array(obs); t=x[:,0]; ve=x[:,1:4]-x[:,4:7]; height=x[:,7]; truthz=x[:,10]
    hz=(height-height[0])-(truthz-truthz[0]); rawgap=truthz-height
    result={'scope':'odometry is scoring only; known flat plane and lowest sphere assumed',
       'observations':len(x),'duration_s':float(t[-1]-t[0]),
       'support_velocity_error_mean_m_s':np.mean(ve,axis=0).tolist(),
       'support_velocity_error_median_m_s':np.median(ve,axis=0).tolist(),
       'support_velocity_error_rmse_m_s':np.sqrt(np.mean(ve*ve,axis=0)).tolist(),
       'mean_velocity_error_times_duration_m':(np.mean(ve,axis=0)*(t[-1]-t[0])).tolist(),
       'flat_plane_relative_z_rmse_m':float(np.sqrt(np.mean(hz**2))),
       'flat_plane_relative_z_final_error_m':float(hz[-1]),
       'flat_plane_relative_z_max_abs_error_m':float(np.max(np.abs(hz))),
       'true_lowest_sphere_gap_m':{'median':float(np.median(rawgap)),'p01':float(np.percentile(rawgap,1)),
          'p99':float(np.percentile(rawgap,99)),'max_abs':float(np.max(np.abs(rawgap)))},
       'contact_reconstruction':contact_counts if a.scene else None,
       'caveat':'geometry proximity is not a measured contact force; timing and truth differencing remain uncertain'}
    a.out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
    np.savez_compressed(a.out.with_suffix('.npz'), samples=x)


if __name__=='__main__': main()
