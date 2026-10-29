"""Export collision-mesh vertices in their body coordinates, without stepping."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    m = mujoco.MjModel.from_xml_path(str(a.scene.resolve()))
    data = {}
    names = {}
    for i in range(m.ngeom):
        if m.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if not (m.geom_contype[i] or m.geom_conaffinity[i]):
            continue
        mesh = int(m.geom_dataid[i])
        start, count = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
        vertices = np.asarray(m.mesh_vert[start:start + count], dtype=np.float64)
        q = np.asarray(m.geom_quat[i], dtype=np.float64)
        # MuJoCo mesh vertices are stored in mesh frame; geom rotation and
        # translation bring them into the owning body's coordinates.
        twice_cross = 2.0 * np.cross(q[1:], vertices)
        transformed = vertices + q[0] * twice_cross + np.cross(q[1:], twice_cross)
        data[f'geom_{i}'] = transformed + m.geom_pos[i]
        names[str(i)] = {'name': m.geom(i).name, 'body_id': int(m.geom_bodyid[i]),
                         'vertices': count}
    np.savez_compressed(a.out, **data)
    a.out.with_suffix('.json').write_text(json.dumps({'version': mujoco.__version__,
                                                       'geometries': names}, indent=2) + '\n')
    print(f'{mujoco.__version__}: exported {len(data)} collision meshes', flush=True)


if __name__ == '__main__':
    main()
