"""X2 joint command order for transport adapters; no hardware I/O."""

JOINT_GROUPS = {
    'leg': tuple(f'{side}_{joint}_joint' for side in ('left','right') for joint in
                 ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')),
    'waist': tuple(f'waist_{joint}_joint' for joint in ('yaw','pitch','roll')),
    'arm': tuple(f'{side}_{joint}_joint' for side in ('left','right') for joint in
                 ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw','wrist_pitch','wrist_roll')),
    'head': ('head_yaw_joint','head_pitch_joint'),
}


def group_command(packet):
    """Map a fully validated named proposal into the official HAL group order."""
    names=tuple(packet['names']); expected={n for group in JOINT_GROUPS.values() for n in group}
    if len(names)!=31 or len(set(names))!=31 or set(names)!=expected:
        raise ValueError('Invalid X2 command joint set')
    indices={name:i for i,name in enumerate(names)}
    fields=('position','velocity','effort','stiffness','damping')
    if any(len(packet[k])!=len(names) for k in fields): raise ValueError('Invalid command vector length')
    return {group:[{'name':name,**{k:packet[k][indices[name]] for k in fields}} for name in order]
            for group,order in JOINT_GROUPS.items()}
