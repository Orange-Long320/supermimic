"""Server-only independent Drake/MuJoCo check of published Omni G1 qpos."""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import zipfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error('Output exists')
    import mujoco
    import numpy as np
    from pydrake.multibody.plant import MultibodyPlant
    from pydrake.multibody.parsing import Parser
    plant = MultibodyPlant(0.)
    Parser(plant).AddModels(str(a.source / 'models/g1/g1_29dof_spherehand.urdf'))
    plant.Finalize()
    context = plant.CreateDefaultContext()
    model = mujoco.MjModel.from_binary_path(str(a.audit / 'source-g1.mjb'))
    data = mujoco.MjData(model)
    config = a.source / 'holosoma-sources/src/holosoma_retargeting/holosoma_retargeting/config_types/data_conversion.py'
    for node in ast.parse(config.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_ROBOT_JOINT_NAMES_DEFAULT' for t in node.targets):
            names = ast.literal_eval(node.value)['g1']
            break
    else:
        raise ValueError('Missing source order')
    addresses = {name: int(plant.GetJointByName(name).position_start()) for name in names}
    if plant.num_positions() != 36 or list(addresses.values()) != list(range(7, 36)):
        raise ValueError(f'Published Drake order differs from conversion: {addresses}')
    columns = [int(model.jnt_qposadr[model.joint(name).id]) for name in names]
    bodies = [(i, plant.GetBodyByName(model.body(i).name)) for i in range(1, model.nbody)]
    rows = []
    with zipfile.ZipFile(a.source / 'robot-terrain.zip') as archive:
        for entry in sorted(n for n in archive.namelist() if n.endswith('.npz')):
            content = archive.read(entry)
            with np.load(io.BytesIO(content), allow_pickle=False) as z:
                q = z['qpos']
            peak = 0.
            rotation_peak = 0.
            for frame in sorted(set([0, len(q)//2, len(q)-1])):
                plant.SetPositions(context, q[frame])
                data.qpos[:3] = q[frame, 4:7]
                data.qpos[3:7] = q[frame, :4]
                data.qpos[columns] = q[frame, 7:]
                mujoco.mj_kinematics(model, data)
                for i, body in bodies:
                    pose = plant.EvalBodyPoseInWorld(context, body)
                    peak = max(peak, float(np.linalg.norm(pose.translation()-data.xpos[i])))
                    rotation_peak = max(rotation_peak, float(np.max(abs(pose.rotation().matrix()-data.xmat[i].reshape(3,3)))))
            rows.append({'motion': entry, 'sha256': hashlib.sha256(content).hexdigest(), 'max_body_position_error_m': peak, 'max_rotation_matrix_error': rotation_peak})
    report = {'joint_position_addresses': addresses, 'motions': rows,
              'max_body_position_error_m': max(r['max_body_position_error_m'] for r in rows),
              'max_rotation_matrix_error': max(r['max_rotation_matrix_error'] for r in rows),
              'scope': 'Every named body, first/middle/last frames of all source clips; order and FK equivalence only, no physical feasibility assertion'}
    report['passed'] = report['max_body_position_error_m'] < 1e-8 and report['max_rotation_matrix_error'] < 1e-8
    a.out.mkdir(parents=True)
    (a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    (a.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'motions'}, indent=2))
    if not report['passed']:
        raise RuntimeError('Drake/MuJoCo FK disagreement')


if __name__ == '__main__':
    main()
