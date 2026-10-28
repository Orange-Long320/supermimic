"""Task-specific acceptance of uninterrupted actual box/landing trajectories.

Keep the original strict physical audit; add measured load-bearing support and
reference-defined task geometry. A pose-only reference is never a policy pass.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def longest(mask):
    best = run = 0
    for value in mask:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['trace', 'dataset', 'out']:
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--clip', required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error('Independent output must be new')
    import mujoco
    import numpy as np
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    manifest = json.loads((a.dataset / 'manifest.json').read_text())
    clip = next(c for c in manifest['clips'] if c['name'] == a.clip)
    task = clip['task_type']
    if task not in ['box_stand', 'vault_land', 'step_up_down_land']:
        raise ValueError('Undeclared task type')
    result = json.loads((a.trace / 'result.json').read_text())
    source = Path(clip['source']) / 'reference-qpos.npz'
    if sha(source) != clip['reference_sha256']:
        raise ValueError('Source reference changed')
    with np.load(source) as z:
        ref = z['qpos'].copy(); ref_v = z['qvel'].copy(); ref_fps = float(z['fps'][0])
    ref[:, :3] += clip['offset']
    with np.load(a.dataset / (a.clip + '-qpos.npz')) as z:
        np.testing.assert_array_equal(ref, z['qpos'])
        np.testing.assert_array_equal(ref_v, z['qvel'])
    with np.load(a.trace / 'baseline-physics.npz') as z:
        q = z['qpos'].copy(); v = z['qvel'].copy(); initial = z['initial_qpos'].copy()
        initial_v = z['initial_qvel'].copy(); constraint = z['constraint_generalized_force'].copy()
        fps = float(z['fps'][0])
    # Reject a different initialization rather than inferring it from root height.
    initial_error = float(np.max(abs(initial - ref[0])))
    initial_v_error = float(np.max(abs(initial_v - ref_v[0])))
    contact_path = a.trace / 'contacts-actual.npz'
    if sha(contact_path) != result.get('contact_trace_sha256'):
        raise ValueError('Missing or changed actual contact trace')
    with np.load(contact_path) as z:
        offsets = z['offsets'].copy(); geom = z['geom'].copy()
        wrench = z['force_torque_on_geom2_world'].copy()
        point = z['position_world'].copy(); normal = z['normal_geom1_to_geom2_world'].copy()
        if float(z['fps'][0]) != fps:
            raise ValueError('Contact timing differs')
    if (offsets.shape != (len(q)+1,) or offsets[0] != 0 or offsets[-1] != len(geom)
        or not np.issubdtype(offsets.dtype, np.integer) or np.any(np.diff(offsets)<0)
        or geom.shape != (len(wrench), 2) or wrench.shape != (len(geom), 6)
        or point.shape != normal.shape or point.shape != (len(geom), 3)
        or not all(np.isfinite(x).all() for x in [wrench, point, normal])):
        raise ValueError('Malformed force trace')
    a.out.mkdir(parents=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name('audit_omni_execution.py')),
        '--trace', str(a.trace), '--dataset', str(a.dataset), '--clip', a.clip,
        '--out', str(a.out / 'strict-physics')], check=True, stdout=subprocess.DEVNULL)
    strict = json.loads((a.out / 'strict-physics/report.json').read_text())
    m = mujoco.MjModel.from_binary_path(str(a.trace / 'native-model.mjb')); d = mujoco.MjData(m)
    root = int(m.jnt_bodyid[0]); weight = float(m.body_mass.sum()*9.81)
    robot = np.array([m.body_rootid[b] == root for b in m.geom_bodyid])
    if np.any(geom < 0) or np.any(geom >= m.ngeom):
        raise ValueError('Invalid contact geometry id')
    boxes = json.loads(Path(manifest['terrain']).read_text())['boxes']
    target = strict['target_box_index']; box = boxes[target]
    center = np.array(box['center']); half = np.array(box['half_size']); rot = np.array(box['rotation_world'])
    target_geom = m.geom(f'omni_training_box_{target}').id
    foot_geoms = {}; hand_geoms = set()
    for side in ['left', 'right']:
        foot_geoms[side] = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(f'robot/{side}_ankle_roll_link').id and robot[g] and (m.geom_contype[g] or m.geom_conaffinity[g])]
        hand_geoms.update(g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(f'robot/{side}_wrist_roll_link').id and robot[g])
    surfaces = [('ground', None, None, None, 0.)]
    for index in clip['box_indices']:
        b = boxes[index]; top = b['center'][2]+b['half_size'][2]
        if top <= strict['target_top_m']-.05:
            surfaces.append((str(index), np.array(b['center']), np.array(b['half_size']), np.array(b['rotation_world']), top))
    rows = []
    for pose in np.concatenate([initial[None], q]):
        d.qpos[:] = pose; mujoco.mj_forward(m, d)
        feet = []
        for side, ids in foot_geoms.items():
            parts = []
            for g in ids:
                if m.geom_type[g] != int(mujoco.mjtGeom.mjGEOM_MESH):
                    raise ValueError('Expected original physical foot meshes')
                mesh = m.geom_dataid[g]; start = m.mesh_vertadr[mesh]; count = m.mesh_vertnum[mesh]
                parts.append(m.mesh_vert[start:start+count] @ d.geom_xmat[g].reshape(3,3).T + d.geom_xpos[g])
            feet.append(np.concatenate(parts))
        on_lower = []
        for name, c, h, r, top in surfaces:
            def planted(points):
                inside = True if c is None else bool((abs(((points-c)@r)[:,:2]) <= h[:2]+1e-6).all())
                return inside and -.001 <= float(points[:,2].min()-top) <= .002
            if all(planted(points) for points in feet): on_lower.append(name)
        rows.append(on_lower)
    foot_load = np.zeros((len(q),2)); hand_load = np.zeros(len(q)); balance = []
    for i in range(len(q)):
        total = np.zeros(3)
        for k in range(int(offsets[i]), int(offsets[i+1])):
            g1, g2 = geom[k]
            if robot[g1] == robot[g2]: continue
            g, other, sign = (g2,g1,1.) if robot[g2] else (g1,g2,-1.)
            force = sign*wrench[k,:3]; total += force
            for j, ids in enumerate(foot_geoms.values()):
                if g in ids: foot_load[i,j] += force[2]
            local = (point[k]-center)@rot
            if (g in hand_geoms and other == target_geom and sign*normal[k,2]>.9
                and abs(local[2]-half[2])<=.002 and (abs(local[:2])<=half[:2]+.002).all()):
                hand_load[i] += max(0.,float(force[2]))
        error = float(np.max(abs(total-constraint[i,:3])))
        relative = error/max(1.,float(np.max(abs(constraint[i,:3]))))
        balance.append(error<=.001 or relative<=1e-5)
    n = round(2*fps)
    terminal = strict['rows'][-n:]
    # Loads are preintegration; use the corresponding preintegration surfaces.
    common_surface = set.intersection(*(set(x) for x in rows[len(q)-n:len(q)])) if len(q)>=n else set()
    stable = bool(len(q)>=n and common_surface and strict['terminal_root_drift_m']<=.02
        and all(r['root_up_z']>=.9 and r['com_speed']<=.1 for r in terminal)
        and .8<=strict['terminal_median_contact_force_over_weight']<=1.2
        and np.median(foot_load[-n:].sum(axis=1))/weight>=.8
        and (np.median(foot_load[-n:],axis=0)/weight>=.05).all())
    # Task direction and exit boundary are fixed by the reference, never the rollout.
    local_ref = (ref[:,:3]-center)@rot
    axis = int(np.argmax(abs(local_ref[-1,:2]-local_ref[0,:2])/half[:2]))
    direction = float(np.sign(local_ref[-1,axis]-local_ref[0,axis]))
    reference_crossing = bool(direction != 0 and direction*local_ref[0,axis]<-half[axis]
        and direction*local_ref[-1,axis]>half[axis]
        and ((abs(local_ref[:,:2])<=half[:2]).all(axis=1)&(local_ref[:,2]>half[2])).any())
    local_actual = (np.concatenate([initial[None],q])[:,:3]-center)@rot
    crossed = bool(direction != 0 and direction*local_actual[0,axis]<-half[axis]
        and direction*local_actual[-1,axis]>half[axis]
        and ((abs(local_actual[:,:2])<=half[:2]).all(axis=1)&(local_actual[:,2]>half[2])).any())
    near_end = bool(np.linalg.norm(q[-1,:2]-ref[-1,:2])<=.15)
    coverage = bool(result.get('last_reference_frame',-1)>=len(ref)-1 and len(q)/fps>=(len(ref)-1)/ref_fps)
    start = bool(result.get('start_reference_frame')==0 and initial_error<=1e-5 and initial_v_error<=1e-4)
    ascent = longest([r['both_feet_on_target'] for r in strict['rows']])/fps>=.1
    hand_support = longest(hand_load>=.05*weight)/fps>=.1
    box_stand = bool(strict['terminal_two_seconds_standing'] and np.median(foot_load[-n:].sum(axis=1))/weight>=.8
        and (np.median(foot_load[-n:],axis=0)/weight>=.05).all())
    task_pass = {'box_stand':box_stand,'vault_land':reference_crossing and crossed and hand_support and stable and near_end,
                 'step_up_down_land':ascent and stable and near_end}[task]
    passed = bool(strict['physical_checks_passed'] and all(balance) and strict['entry_on_lower_surface']
        and result.get('policy_alone') and start and coverage and task_pass)
    report = {'clip':a.clip,'task_type':task,'complete_task_verified':passed,'physical_checks_passed':strict['physical_checks_passed'],
        'source_zero_initialization_verified':start,'initial_qpos_max_error':initial_error,'initial_qvel_max_error':initial_v_error,
        'complete_reference_coverage':coverage,'terminal_two_seconds_box_standing':box_stand,'terminal_two_seconds_lower_landing':stable,
        'terminal_near_reference_end':near_end,'terminal_lower_surfaces':sorted(common_surface),'ascent_two_feet_100ms':ascent,
        'reference_crossing_axis':axis,'reference_crossing_direction':direction,'actual_opposite_face_crossing':crossed,
        'reference_opposite_face_crossing':reference_crossing,
        'hand_support_5percent_weight_100ms':hand_support,'peak_hand_vertical_force_over_weight':float(hand_load.max()/weight),
        'contact_root_force_balance_verified':all(balance),'contact_trace_sha256':sha(contact_path),
        'trajectory_sha256':strict['trajectory_sha256'],'model_sha256':strict['model_sha256'],
        'reference_sha256':sha(source),'checkpoint_sha256':strict['checkpoint_sha256'],
        'threshold_scope':'Original physical limits; landing uses original 2s/up.9/COM.1/drift.02/weight.8-1.2. Additional measured foot load sum>=.8 and each>=.05 bodyweight; vault hand load>=.05 bodyweight for100ms.'}
    (a.out/'report.json').write_text(json.dumps(report,indent=2))
    (a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
