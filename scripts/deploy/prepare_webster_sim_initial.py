"""Set initial pose in an isolated copy of official simulator config (simulation only)."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sim-copy', type=Path, required=True)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--stand', type=Path, required=True)
    p.add_argument('--record', type=Path, required=True)
    a = p.parse_args()
    if a.record.exists(): raise ValueError('Record already exists')
    root = a.sim_copy/'configuration/robot/lx2501_3_t2d5/model_info'
    config = root/'default.yaml'; xml = root/'x2.xml'
    spec = json.loads(a.profile.read_text())
    with np.load(a.stand, allow_pickle=False) as data:
        names = tuple(data['joint_names']); q = data['joint_pos'][0]
        pelvis = list(data['body_names']).index('pelvis')
        pos = data['body_pos_w'][0, pelvis]; quat = data['body_quat_w'][0, pelvis]
    if names != tuple(spec['joint_names']): raise ValueError('joint order mismatch')
    values = dict(zip(names, map(float, q)))
    cfg = yaml.safe_load(config.read_text())
    groups = cfg['actual']['active_joint_name']
    if set(values) != set(name for group in groups for name in group): raise ValueError('joint mapping mismatch')
    cfg['logical']['nominal_configuration'] = [[values[name] for name in group] for group in groups]
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    tree = ET.parse(xml); pelvis_body = tree.find("./worldbody/body[@name='pelvis']")
    roll, pitch, yaw = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_euler('xyz')
    initial_quat = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_quat()[[3, 0, 1, 2]]
    pelvis_body.set('pos', '0 0 '+str(float(pos[2])))
    pelvis_body.attrib.pop('euler', None)
    pelvis_body.set('quat', ' '.join(map(str, initial_quat)))
    tree.write(xml, encoding='unicode')
    a.record.write_text(json.dumps({'source': str(a.stand), 'joint_names': names, 'q': q.tolist(),
                                     'height': float(pos[2]), 'quat_wxyz': initial_quat.tolist()}, indent=2)+'\n')


if __name__ == '__main__': main()
