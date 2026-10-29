"""Server-only source/X2 stance attribution; immutable inputs, no training edits."""
import argparse
import ast
import io
import json
from pathlib import Path
import zipfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error('New output required')
    import mujoco
    import numpy as np
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation
    sole_config = json.loads((Path(__file__).resolve().parents[2] / 'configs/robots/x2_ultra-sole-augmentation.json').read_text())['sole_boxes']
    b = a.data_root / 'logs'
    raw = a.data_root / 'data/external/omniretarget_20261028_r1/raw'
    cfg = raw / 'holosoma-sources/src/holosoma_retargeting/holosoma_retargeting/config_types/data_conversion.py'
    for node in ast.parse(cfg.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_ROBOT_JOINT_NAMES_DEFAULT' for t in node.targets):
            names = ast.literal_eval(node.value)['g1']
            break
    g = mujoco.MjModel.from_binary_path(str(b / 'omni_terrain_screen_20261029_r4/audit/source-g1.mjb'))
    cols = [g.jnt_qposadr[g.joint(n).id] for n in names]

    def metrics(m, pose, source=False):
        d = mujoco.MjData(m); d.qpos[:] = pose; mujoco.mj_forward(m, d)
        prefix = '' if source else 'robot/'
        root = d.xmat[m.body(prefix+'pelvis').id].reshape(3, 3)
        forward = root[:, 0].copy(); forward[2] = 0; forward /= np.linalg.norm(forward)
        lateral = np.cross([0, 0, 1], forward)
        feet = []; yaw = []; points = []; knees = []
        for side in ['left', 'right']:
            ankle = m.body(prefix+side + '_ankle_roll_link').id
            if source:
                corners = d.xpos[[m.body(f'{side}_ankle_roll_sphere_{k}_link').id for k in range(1, 5)]].copy()
                axis = corners[2:4].mean(0) - corners[:2].mean(0)
            else:
                item = next(x for x in sole_config if x['side'] == side)
                rot = Rotation.from_quat(np.array(item['quaternion_in_body_wxyz'])[[1,2,3,0]]).as_matrix()
                local = np.array([[-1,-1,-1],[-1,1,-1],[1,-1,-1],[1,1,-1]])*item['half_size_m']
                local = local @ rot.T + item['position_in_body_m']
                corners = local @ d.xmat[ankle].reshape(3,3).T + d.xpos[ankle]
                axis = corners[2:].mean(0)-corners[:2].mean(0)
            yaw.append(float(np.degrees(np.arctan2(axis @ lateral, axis @ forward))))
            feet.append(d.xpos[ankle].copy()); points.extend(corners[:, :2])
            knees.append(d.xpos[m.body(prefix+side + '_knee_link').id].copy())
        hull = ConvexHull(np.array(points))
        com = d.subtree_com[m.body(prefix+'pelvis').id]
        margin = float(np.min(-(hull.equations[:, :2] @ com[:2] + hull.equations[:, 2])))
        physical_margin = None
        if not source:
            contact_vertices = []
            for side in ['left','right']:
                body=m.body(prefix+side+'_ankle_roll_link').id
                for gidx in range(m.ngeom):
                    if m.geom_bodyid[gidx]!=body or not(m.geom_contype[gidx] or m.geom_conaffinity[gidx]):continue
                    mesh=m.geom_dataid[gidx]
                    if mesh<0:continue
                    start,count=m.mesh_vertadr[mesh],m.mesh_vertnum[mesh]
                    vertices=m.mesh_vert[start:start+count]@d.geom_xmat[gidx].reshape(3,3).T+d.geom_xpos[gidx]
                    contact_vertices.extend(vertices[vertices[:,2]<=vertices[:,2].min()+.003,:2])
            actual=ConvexHull(np.array(contact_vertices))
            physical_margin=float(np.min(-(actual.equations[:,:2]@com[:2]+actual.equations[:,2])))
        return {'foot_yaw_relative_pelvis_deg': yaw, 'ankle_lateral_spacing_m': float((feet[0]-feet[1]) @ lateral),
                'knee_lateral_spacing_m': float((knees[0]-knees[1]) @ lateral),
                'com_projection_hull_margin_m': margin,
                'com_projection_near_bottom_mesh_hull_margin_m': physical_margin,
                'pelvis_world_yaw_deg': float(np.degrees(np.arctan2(forward[1],forward[0]))),
                'joint_angles_deg': {m.joint(i).name: float(np.degrees(pose[m.jnt_qposadr[i]])) for i in range(1, m.njnt) if any(t in m.joint(i).name for t in ['hip_roll', 'hip_yaw', 'ankle_roll'])}}

    report = {}
    with zipfile.ZipFile(raw / 'robot-terrain.zip') as archive:
        for name in ['step43_box_stand', 'step28_return_land', 'hand46_vault_land']:
            base = b / 'omni_task_goals_20261029_r1'
            command = json.loads((base / f'{name}-export-command.json').read_text())
            model_path = Path(command[command.index('--native-model')+1]); m = mujoco.MjModel.from_binary_path(str(model_path))
            meta = json.loads((base / name / 'dense/result.json').read_text())
            q = np.load(base / name / 'policy-input/reference-qpos.npz')['qpos']
            parent_path = Path(meta['source_reference_path']); parent = np.load(parent_path)['qpos']
            source = np.load(io.BytesIO(archive.read(meta['source_motion'])))['qpos']
            end = int(round(meta['task_variant']['stop_frame'] / 200 * 30))
            row = {'model': str(model_path), 'reference': str(base/name/'policy-input/reference-qpos.npz'), 'parent': str(parent_path), 'source_motion': meta['source_motion'], 'source_end_frame': end}
            for label, idx in [('start', 0), ('end', end)]:
                pose = np.zeros(g.nq); pose[:3] = source[idx, 4:7]; pose[3:7] = source[idx, :4]; pose[cols] = source[idx, 7:]
                row['source_' + label] = metrics(g, pose, True)
                row['x2_' + label] = metrics(m, q[0 if label == 'start' else -1])
            row['hold_parent_max_qpos_difference'] = float(np.max(abs(q[-1]-parent[meta['task_variant']['stop_frame']])))
            report[name] = row
    a.out.mkdir(parents=True)
    (a.out / 'as-run.py').write_bytes(Path(__file__).read_bytes())
    (a.out / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
