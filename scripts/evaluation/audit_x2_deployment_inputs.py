"""Compare recorded pelvis-policy inputs with the PC2 URDF, without actuation.

Only reads files and evaluates forward kinematics. Does not step physics,
correct joint readings, clip targets, or create ROS publishers.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def urdf_geometry(path):
    joints = ET.parse(path).getroot().findall('joint')
    parents = {j.find('parent').get('link') for j in joints}
    children = {j.find('child').get('link') for j in joints}
    roots = parents-children
    if len(roots) != 1:
        raise ValueError('URDF must have one root')
    transforms = {roots.pop(): (np.zeros(3), np.eye(3))}
    remaining = list(joints)
    result = {}
    while remaining:
        pending = []
        for joint in remaining:
            parent = joint.find('parent').get('link')
            if parent not in transforms:
                pending.append(joint)
                continue
            position, rotation = transforms[parent]
            origin = joint.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            child_p = position + rotation @ xyz
            child_r = rotation @ Rotation.from_euler('xyz', rpy).as_matrix()
            transforms[joint.find('child').get('link')] = (child_p, child_r)
            if joint.get('type') in ('revolute', 'continuous'):
                axis_node = joint.find('axis')
                axis = np.fromstring(axis_node.get('xyz', '1 0 0'), sep=' ') if axis_node is not None else np.array([1., 0., 0.])
                axis = child_r @ (axis / np.linalg.norm(axis))
                limit = joint.find('limit')
                if joint.get('type') != 'revolute' or limit is None:
                    raise ValueError('Expected bounded revolute X2 joints')
                result[joint.get('name')] = {
                    'position': child_p, 'axis': axis,
                    'limits': np.array([float(limit.get('lower')), float(limit.get('upper'))])}
        if len(pending) == len(remaining):
            raise ValueError('URDF is disconnected or contains a cycle')
        remaining = pending
    pelvis_p, pelvis_r = transforms['pelvis']
    for row in result.values():
        row['position'] = pelvis_r.T @ (row['position']-pelvis_p)
        row['axis'] = pelvis_r.T @ row['axis']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('profile', 'urdf', 'frames', 'shadow-report', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    spec = json.loads(args.profile.read_text())
    source = json.loads(args.shadow_report.read_text())
    names = spec['joint_names']
    if len(names) != 31 or source['joint_names'] != names:
        raise ValueError('Wrong X2 joint contract')
    if source['policy_sha256'] != '4424f231fc1896da098b2fca366e8522df6791596638b2b9257ba682520bbe4c':
        raise ValueError('Not the selected 11500 policy')
    path = Path(spec['xml_path'])
    model = mujoco.MjModel.from_xml_path(str(path if path.is_absolute() else args.profile.parent/path))
    data = mujoco.MjData(model)
    data.qpos[:] = 0.
    data.qpos[:7] = [0., 0., 0., 1., 0., 0., 0.]
    mujoco.mj_kinematics(model, data)
    urdf = urdf_geometry(args.urdf)
    geometry = {}
    model_limits = []
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0 or name not in urdf or not model.jnt_limited[jid]:
            raise ValueError('Missing or unbounded joint: '+name)
        joint = urdf[name]
        geometry[name] = {
            'zero_axis_error_deg': float(np.degrees(np.arccos(np.clip(data.xaxis[jid] @ joint['axis'], -1., 1.)))),
            'zero_anchor_error_m': float(np.linalg.norm(data.xanchor[jid]-joint['position'])),
            'training_limits_rad': model.jnt_range[jid].tolist(),
            'pc2_urdf_limits_rad': joint['limits'].tolist()}
        model_limits.append(model.jnt_range[jid].copy())
    model_limits = np.asarray(model_limits)
    urdf_limits = np.asarray([urdf[name]['limits'] for name in names])
    measured_bad, output_bad = Counter(), {'feedback': Counter(), 'zero': Counter()}
    output_frames_bad = Counter()
    worst = {key: {'excess_rad': 0.} for key in output_bad}
    first = None
    count = 0
    for line in args.frames.open():
        row = json.loads(line)
        q = np.asarray(row['joint_position'])
        if q.shape != (31,) or not np.isfinite(q).all():
            raise ValueError('Invalid recorded joints')
        outside = (q < urdf_limits[:, 0]) | (q > urdf_limits[:, 1])
        measured_bad.update(names[j] for j in np.flatnonzero(outside))
        if first is None:
            first = {'pelvis_position_m': row['pelvis_position'], 'joint_position_rad': dict(zip(names, q.tolist()))}
        for key in output_bad:
            target = np.asarray(row[key]['target_logging_only'])
            if target.shape != (31,) or not np.isfinite(target).all():
                raise ValueError('Invalid recorded target')
            excess = np.maximum(urdf_limits[:, 0]-target, target-urdf_limits[:, 1])
            bad = excess > 0.
            output_bad[key].update(names[j] for j in np.flatnonzero(bad))
            output_frames_bad[key] += int(bad.any())
            j = int(np.argmax(excess))
            if excess[j] > worst[key]['excess_rad']:
                worst[key] = {'joint': names[j], 'target_rad': float(target[j]),
                              'pc2_urdf_limits_rad': urdf_limits[j].tolist(),
                              'excess_rad': float(excess[j]), 'tick_ns': row['tick_ns']}
        count += 1
    if not count:
        raise ValueError('Empty recorded policy input')
    report = {
        'scope': 'recorded input and model/PC2 URDF comparison; no motor commands',
        'selected_checkpoint': 11500, 'policy_sha256': source['policy_sha256'],
        'sha256': {key: digest(getattr(args, key)) for key in ('profile', 'urdf', 'frames', 'shadow_report')},
        'frames': count, 'zero_joint_geometry': geometry,
        'max_zero_axis_error_deg': max(x['zero_axis_error_deg'] for x in geometry.values()),
        'max_zero_anchor_error_m': max(x['zero_anchor_error_m'] for x in geometry.values()),
        'max_joint_limit_difference_rad': float(np.max(np.abs(model_limits-urdf_limits))),
        'measured_outside_pc2_urdf_by_joint': dict(measured_bad),
        'target_frames_outside_pc2_urdf': dict(output_frames_bad),
        'target_outside_pc2_urdf_by_joint': {k: dict(v) for k, v in output_bad.items()},
        'target_bounds_comparison_semantics': 'PD setpoints are not measured joint angles; this count alone does not establish policy failure',
        'worst_target_violation': worst, 'first_recorded_input': first,
        'hardware_encoder_zero_direction_validated': False,
        'online_pelvis_source_validated': False,
        'motor_deployment_gate_passed': False,
        'reason': 'Measured inputs are incompatible with file bounds; physical calibration and hardware PD command ranges remain unvalidated.'}
    (args.out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: report[k] for k in ('frames', 'max_zero_axis_error_deg', 'max_zero_anchor_error_m',
          'max_joint_limit_difference_rad', 'measured_outside_pc2_urdf_by_joint',
          'target_frames_outside_pc2_urdf', 'worst_target_violation', 'motor_deployment_gate_passed')}, indent=2))


if __name__ == '__main__':
    main()
