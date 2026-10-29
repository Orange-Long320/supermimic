"""Explicit, selective geometry tolerances for training references."""
import json
from pathlib import Path


def load_reference_policy(path=None):
    if path is None:
        return dict(version=1, name='legacy_strict_geometry', contact_penetration_m=.001,
            body_penetration_m=.001, self_penetration_m=.001, terminal_contact_gap_m=.002,
            force_diagnostics_are_training_gate=False)
    policy = json.loads(Path(path).read_text())
    if policy.get('version') != 1 or policy.get('force_diagnostics_are_training_gate') is not False:
        raise ValueError('Explicit version1 training-reference policy required')
    for key in ['contact_penetration_m', 'body_penetration_m', 'self_penetration_m', 'terminal_contact_gap_m']:
        if not isinstance(policy.get(key), (int, float)) or not 0 < policy[key] <= .005:
            raise ValueError('Reference tolerance must be explicit and at most5mm: ' + key)
    return policy


def audit_reference_geometry(model, poses, policy):
    import mujoco
    import numpy as np
    data = mujoco.MjData(model)
    root = int(model.jnt_bodyid[0])
    robot = [g for g in range(model.ngeom) if model.body_rootid[model.geom_bodyid[g]] == root
        and (model.geom_contype[g] or model.geom_conaffinity[g])]
    terrain = [g for g in range(model.ngeom) if model.body_rootid[model.geom_bodyid[g]] != root
        and (model.geom_contype[g] or model.geom_conaffinity[g])]
    if not robot or not terrain:
        raise ValueError('Robot and matching fixed terrain required')
    maxima = dict(contact=0., body=0., self=0.)
    violations, failed_frames, worst, witness = 0, set(), [], np.zeros(6)
    for i, pose in enumerate(poses):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        for g in robot:
            body = model.body(int(model.geom_bodyid[g])).name
            foot, palm = body.endswith('_ankle_roll_link'), body.endswith('_wrist_roll_link')
            for o in terrain:
                distance = float(mujoco.mj_geomDistance(model, data, g, o, .02, witness))
                if distance >= 0:
                    continue
                direction = witness[:3] - witness[3:]
                length = float(np.linalg.norm(direction))
                normal_z = -direction[2] / length if length > 1e-12 else -1.
                # Side-face foot collisions and underside hand collisions do
                # not inherit a support-contact allowance.
                contact = (foot and normal_z >= .7) or (palm and normal_z >= -.1)
                category = 'contact' if contact else 'body'
                limit = policy['contact_penetration_m' if contact else 'body_penetration_m']
                penetration = -distance
                maxima[category] = max(maxima[category], penetration)
                if penetration > limit:
                    violations += 1
                    failed_frames.add(i)
                    worst.append(dict(frame=i, body=body, terrain=model.geom(o).name,
                        category=category, penetration_m=penetration, limit_m=limit, normal_z=normal_z))
        for c in data.contact:
            if int(c.geom1) in robot and int(c.geom2) in robot:
                penetration = max(0., -float(c.dist))
                maxima['self'] = max(maxima['self'], penetration)
                if penetration > policy['self_penetration_m']:
                    violations += 1
                    failed_frames.add(i)
                    worst.append(dict(frame=i, category='self', penetration_m=penetration,
                        limit_m=policy['self_penetration_m']))
    return dict(passed=not violations, policy=policy, samples=len(poses),
        max_penetration_m=max(maxima['contact'], maxima['body']),
        maximum_support_contact_penetration_m=maxima['contact'],
        maximum_other_body_penetration_m=maxima['body'], maximum_self_penetration_m=maxima['self'],
        failed_samples=sorted(failed_frames), violation_count=violations,
        worst_violations=sorted(worst, key=lambda r: r['penetration_m'], reverse=True)[:20],
        scope='Sampled training-reference geometry. Approximate force fits are advisory; actual policy acceptance is separate.')
