"""Create and compare a mesh-free model for kinematics only, never physics."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('New output directory required')
    profile = json.loads(a.profile.read_text()); model_path = a.profile.parent/profile['xml_path']
    original = mujoco.MjModel.from_xml_path(str(model_path))
    tree = ET.parse(model_path); root = tree.getroot()
    for parent in root.iter():
        for child in list(parent):
            if child.tag in ('asset', 'geom'):
                parent.remove(child)
    a.out.mkdir(parents=True)
    xml = a.out/'fk-only.xml'; tree.write(xml, encoding='unicode')
    reduced = mujoco.MjModel.from_xml_path(str(xml))
    if (original.nq, original.nv, original.nbody) != (reduced.nq, reduced.nv, reduced.nbody):
        raise ValueError('Model dimensions differ')
    if [original.body(i).name for i in range(original.nbody)] != [reduced.body(i).name for i in range(reduced.nbody)]:
        raise ValueError('Body mapping differs')
    if [original.joint(i).name for i in range(original.njnt)] != [reduced.joint(i).name for i in range(reduced.njnt)]:
        raise ValueError('Joint mapping differs')
    d1, d2 = mujoco.MjData(original), mujoco.MjData(reduced)
    rng = np.random.default_rng(217); pos_error = quat_error = 0.
    for _ in range(50):
        q = original.qpos0.copy(); q[:3] = rng.uniform(-1., 1., 3)
        q[3:7] = rng.normal(size=4); q[3:7] /= np.linalg.norm(q[3:7]); q[7:] = rng.uniform(-.5,.5,len(q)-7)
        d1.qpos[:] = d2.qpos[:] = q
        mujoco.mj_kinematics(original, d1); mujoco.mj_kinematics(reduced, d2)
        pos_error = max(pos_error, float(np.max(np.abs(d1.xpos-d2.xpos))))
        quat_error = max(quat_error, float(np.max(np.abs(d1.xquat-d2.xquat))))
    if max(pos_error, quat_error) > 1e-10:
        raise ValueError('Kinematics comparison failed')
    profile['xml_path'] = xml.name
    (a.out/'profile.json').write_text(json.dumps(profile, indent=2)+'\n')
    report = {'original_model_sha256':sha(model_path), 'fk_model_sha256':sha(xml),
              'source_profile_sha256':sha(a.profile), 'fk_profile_sha256':sha(a.out/'profile.json'),
              'samples':50, 'max_position_error_m':pos_error, 'max_quaternion_component_error':quat_error,
              'scope':'matching forward kinematics only; geometry, contact and physical dynamics removed'}
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
