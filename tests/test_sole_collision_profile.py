"""Sole profiles must be explicit, match the mesh, and preserve body dynamics."""
import hashlib
import json

import mujoco
import numpy as np
import pytest

from supermimic.robots.sole_collision import apply_sole_box_profile


def fixture_spec(tmp_path):
    vertices=' '.join(str(v) for x in [-.1,.1] for y in [-.05,.05] for z in [0.,.02] for v in [x,y,z])
    xml=f'''<mujoco><asset><mesh name="foot" vertex="{vertices}"/></asset><worldbody>
    <body name="pelvis"><freejoint/><inertial pos="0 0 0" mass="10" diaginertia="1 1 1"/>
    <body name="left_ankle_roll_link" pos="0 .12 -.7"><inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/><geom name="foot_left_collision" type="mesh" mesh="foot"/></body>
    <body name="right_ankle_roll_link" pos="0 -.12 -.7"><inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/><geom name="foot_right_collision" type="mesh" mesh="foot"/></body>
    </body></worldbody></mujoco>'''
    spec=mujoco.MjSpec.from_string(xml);m=spec.compile();items=[]
    for side in ['left','right']:
        g=m.geom('foot_'+side+'_collision').id;k=m.geom_dataid[g];a,n=m.mesh_vertadr[k],m.mesh_vertnum[k]
        items.append({'geom':'foot_'+side+'_collision','side':side,'half_size_m':[.09,.04,.004],
            'position_in_body_m':[0.,0.,.00425],'quaternion_in_body_wxyz':[1.,0.,0.,0.],
            'source_mesh_vertices_sha256':hashlib.sha256(m.mesh_vert[a:a+n].tobytes()).hexdigest()})
    path=tmp_path/'sole.json';path.write_text(json.dumps({'robot':'x2_ultra','sole_boxes':items}))
    return spec,m,path


def test_unset_profile_keeps_original_spec(monkeypatch):
    monkeypatch.delenv('SM_SOLE_BOX_CONFIG',raising=False)
    sentinel=object()
    assert apply_sole_box_profile(sentinel) is sentinel


def test_explicit_profile_preserves_inertia_and_changes_only_target_geoms(tmp_path,monkeypatch):
    monkeypatch.setenv('SM_ROBOT','x2_ultra')
    spec,original,path=fixture_spec(tmp_path)
    result=apply_sole_box_profile(spec,path).compile()
    for name in ['body_mass','body_inertia','body_ipos','body_iquat','jnt_range']:
        np.testing.assert_array_equal(getattr(result,name),getattr(original,name))
    assert all(result.geom_type[result.geom('foot_'+side+'_collision').id]==mujoco.mjtGeom.mjGEOM_BOX for side in ['left','right'])


def test_box_outside_original_mesh_is_rejected_before_mutation(tmp_path,monkeypatch):
    monkeypatch.setenv('SM_ROBOT','x2_ultra')
    spec,original,path=fixture_spec(tmp_path);config=json.loads(path.read_text())
    config['sole_boxes'][1]['half_size_m'][0]=.3;path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='outside'):
        apply_sole_box_profile(spec,path)
    assert spec.geom('foot_left_collision').type==mujoco.mjtGeom.mjGEOM_MESH


def test_profile_for_different_mesh_is_rejected(tmp_path,monkeypatch):
    monkeypatch.setenv('SM_ROBOT','x2_ultra')
    spec,original,path=fixture_spec(tmp_path);config=json.loads(path.read_text())
    config['sole_boxes'][0]['source_mesh_vertices_sha256']='0'*64;path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='does not match'):
        apply_sole_box_profile(spec,path)


def test_augmentation_preserves_original_mesh_and_inertia(tmp_path,monkeypatch):
    monkeypatch.setenv('SM_ROBOT','x2_ultra')
    spec,original,path=fixture_spec(tmp_path);config=json.loads(path.read_text());config['mode']='augment'
    path.write_text(json.dumps(config));result=apply_sole_box_profile(spec,path).compile()
    np.testing.assert_array_equal(result.body_mass,original.body_mass)
    np.testing.assert_array_equal(result.body_inertia,original.body_inertia)
    for side in ['left','right']:
        assert result.geom_type[result.geom('foot_'+side+'_collision').id]==mujoco.mjtGeom.mjGEOM_MESH
        assert result.geom_type[result.geom('foot_'+side+'_sole_collision').id]==mujoco.mjtGeom.mjGEOM_BOX
