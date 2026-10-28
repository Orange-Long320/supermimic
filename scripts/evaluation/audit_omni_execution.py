"""Audit an actual uninterrupted Omni policy capture in its original terrain.

Report strict geometry and complete ascent/terminal support independently of
training return. A failed capture remains a failed result, even with exit zero.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--clip',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('Independent audit output must be new')
    import mujoco
    import numpy as np
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra',check_assets=True)
    from supermimic.robots.dimensions import PROFILE
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    dataset=json.loads((a.dataset/'manifest.json').read_text())
    if sha(dataset['terrain'])!=dataset['terrain_sha256']:raise ValueError('Terrain manifest changed')
    clip=next(c for c in dataset['clips'] if c['name']==a.clip)
    result=json.loads((a.trace/'result.json').read_text())
    if result['source_sha256']!=clip['motion_sha256']:raise ValueError('Capture uses a different reference')
    terrain=json.loads(Path(dataset['terrain']).read_text())['boxes']
    m=mujoco.MjModel.from_binary_path(str(a.trace/'native-model.mjb'));d=mujoco.MjData(m)
    if (m.nq,m.nv,m.nu,m.neq)!=(38,37,31,0):raise ValueError('Unexpected robot layout or equality support')
    np.testing.assert_allclose(m.opt.gravity,[0,0,-9.81],rtol=0,atol=1e-12)
    if not np.array_equal(m.jnt_dofadr[m.actuator_trnid[:,0]],np.arange(6,37)):raise ValueError('Root must be unactuated')
    for i,box in enumerate(terrain):
        g=m.geom(f'omni_training_box_{i}').id
        np.testing.assert_allclose(m.geom_pos[g],box['center'],atol=1e-10)
        np.testing.assert_allclose(m.geom_size[g],box['half_size'],atol=1e-10)
        matrix=np.zeros(9);mujoco.mju_quat2Mat(matrix,m.geom_quat[g])
        np.testing.assert_allclose(matrix.reshape(3,3),box['rotation_world'],atol=1e-9)
    with np.load(a.trace/'baseline-physics.npz') as z:
        q=z['qpos'].copy();v=z['qvel'].copy();force=z['actuator_force'].copy();constraint=z['constraint_generalized_force'].copy()
        initial=z['initial_qpos'].copy();initial_v=z['initial_qvel'].copy();fps=float(z['fps'][0])
    if q.shape!=(len(v),38) or v.shape!=(len(q),37) or force.shape!=(len(q),31) or constraint.shape!=v.shape or len(q)==0:raise ValueError('Invalid actual capture layout')
    if not all(np.isfinite(x).all() for x in [q,v,force,constraint]):raise ValueError('Nonfinite actual capture')
    effort=np.array([PROFILE['effort_limit'][n] for n in PROFILE['joint_names']]);speed=np.array([PROFILE['velocity_limit'][n] for n in PROFILE['joint_names']])
    np.testing.assert_allclose(m.actuator_forcerange,np.c_[-effort,effort],atol=1e-9)
    target_index=max(clip['box_indices'],key=lambda i:terrain[i]['center'][2]+terrain[i]['half_size'][2])
    target=terrain[target_index];center=np.array(target['center']);half=np.array(target['half_size']);rotation=np.array(target['rotation_world'])
    np.testing.assert_allclose(rotation[2],[0,0,1],atol=1e-9)
    top=center[2]+half[2];root=int(m.jnt_bodyid[0])
    robot=[g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]]==root and (m.geom_contype[g] or m.geom_conaffinity[g])]
    obstacles=[g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]]!=root and (m.geom_contype[g] or m.geom_conaffinity[g])]
    feet={side:[] for side in ['left','right']}
    for side in feet:
        body=m.body(f'robot/{side}_ankle_roll_link').id
        for g in robot:
            if m.geom_bodyid[g]!=body:continue
            mesh=int(m.geom_dataid[g])
            if m.geom_type[g]!=int(mujoco.mjtGeom.mjGEOM_MESH):raise ValueError('Expected original unaugmented foot meshes')
            first,count=m.mesh_vertadr[mesh],m.mesh_vertnum[mesh];feet[side].append((g,m.mesh_vert[first:first+count].copy()))
        if not feet[side]:raise ValueError('Missing complete physical foot geometry')
    witness=np.zeros(6);jac=np.zeros((3,m.nv));peak=0.;self_peak=0.;rows=[]
    for i,(pose,velocity) in enumerate(zip(np.r_[initial[None],q],np.r_[initial_v[None],v])):
        d.qpos[:]=pose;d.qvel[:]=velocity;mujoco.mj_forward(m,d)
        for g in robot:
            for other in obstacles:peak=max(peak,-float(mujoco.mj_geomDistance(m,d,g,other,.02,witness)))
        for c in d.contact:
            if c.geom1 in robot and c.geom2 in robot:self_peak=max(self_peak,-float(c.dist))
        planted=[];heights=[]
        for side,parts in feet.items():
            points=np.concatenate([vertices@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g] for g,vertices in parts])
            local=(points-center)@rotation;gap=float(local[:,2].min()-half[2]);heights.append(float(points[:,2].min()))
            planted.append(bool((abs(local[:,:2])<=half[:2]+1e-6).all() and -.001<=gap<=.002))
        mujoco.mj_jacSubtreeCom(m,d,jac,root)
        rows.append({'t':i/fps,'both_feet_on_target':all(planted),'foot_min_z':heights,'root_position':pose[:3].tolist(),
            'root_up_z':float(d.xmat[root].reshape(3,3)[2,2]),'com_speed':float(np.linalg.norm(jac@velocity))})
    all_q=np.r_[initial[None],q]
    hard=float(np.maximum(np.maximum(m.jnt_range[1:,0]-all_q[:,7:],all_q[:,7:]-m.jnt_range[1:,1]),0).max())
    speed_ratio=float((abs(v[:,6:])/speed).max());effort_ratio=float((abs(force)/effort).max())
    hold_steps=round(2*fps);terminal=rows[-hold_steps:];terminal_positions=np.array([r['root_position'] for r in terminal])
    drift=float(np.linalg.norm(terminal_positions-terminal_positions[0],axis=1).max())
    support_ratio=float(np.median(constraint[-hold_steps:,2])/(m.body_mass.sum()*9.81))
    entry_lower=bool(max(rows[0]['foot_min_z'])<top-.05)
    terminal_standing=bool(len(q)>=hold_steps and all(r['both_feet_on_target'] and r['root_up_z']>=.9 and r['com_speed']<=.1 for r in terminal) and drift<=.02 and .8<=support_ratio<=1.2)
    physics=bool(peak<=.001 and self_peak<=.001 and hard<=1e-6 and speed_ratio<=1. and effort_ratio<=1.00001 and result.get('external_applied_force_max')==0. and result.get('state_resets_after_entry')==0)
    report={'clip':a.clip,'target_box_index':target_index,'target_top_m':top,'seconds_executed':len(q)/fps,
        'maximum_environment_penetration_m':peak,'maximum_self_penetration_m':self_peak,'hard_joint_violation_rad':hard,
        'maximum_speed_ratio':speed_ratio,'maximum_effort_ratio':effort_ratio,'physical_checks_passed':physics,
        'entry_on_lower_surface':entry_lower,'ever_both_feet_on_target':any(r['both_feet_on_target'] for r in rows),
        'terminal_two_seconds_standing':terminal_standing,'terminal_root_drift_m':drift,'terminal_median_contact_force_over_weight':support_ratio,
        'complete_climb_verified':bool(physics and entry_lower and terminal_standing and result['policy_alone'] and result['start_reference_frame']==0),
        'external_applied_force_max':result.get('external_applied_force_max'),'checkpoint_sha256':result['checkpoint_sha256'],
        'model_sha256':sha(a.trace/'native-model.mjb'),'trajectory_sha256':sha(a.trace/'baseline-physics.npz'),
        'scope':__doc__,'rows':rows}
    a.out.mkdir(parents=True);(a.out/'report.json').write_text(json.dumps(report,indent=2));(a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
