"""Prepare X2 scene and motion for UniLab; execute on server in main environment.

Only configuration and motion-body indexing change. Physical arrays are checked
against the project's compiled scene; no simulation stepping is performed here.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET


def number(value):
    import numpy as np
    return " ".join(format(float(x), ".17g") for x in np.asarray(value).reshape(-1))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.project_root.resolve() / "src"))
    import mujoco
    import numpy as np
    from supermimic.robots.paths import select_robot
    select_robot("x2_ultra", check_assets=True)
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg
    from supermimic.robots.dimensions import PROFILE
    from mjlab.scene import Scene

    cfg = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=[], num_envs=1))
    scene = Scene(cfg.scene, device="cpu")
    original = scene.compile()
    cfg.sim.mujoco.apply(original)
    cfg.sim.mujoco.apply(SimpleNamespace(opt=scene.spec.option))
    root = ET.fromstring(scene.spec.to_xml())
    for parent in root.iter("default"):
        for child in list(parent):
            if child.tag == "default" and not child.attrib and not len(child):
                parent.remove(child)
    # MjSpec's XML uses six significant digits. Explicitly restore physical
    # parameters in their owning XML fields before either backend compiles it.
    for el in root.iter("body"):
        body = original.body(el.attrib["name"])
        el.set("pos", number(body.pos))
        el.set("quat", number(body.quat))
        inertia = el.find("inertial")
        if inertia is not None:
            for attr, value in (("pos", body.ipos), ("quat", body.iquat),
                                ("mass", body.mass), ("diaginertia", body.inertia)):
                inertia.set(attr, number(value))
    for el in root.iter("joint"):
        if "name" not in el.attrib:
            continue
        joint = original.joint(el.attrib["name"])
        jid = joint.id
        if original.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        did = original.jnt_dofadr[jid]
        for attr, value in (("pos", joint.pos), ("axis", joint.axis),
                            ("range", joint.range), ("stiffness", joint.stiffness),
                            ("damping", original.dof_damping[did]),
                            ("armature", original.dof_armature[did]),
                            ("frictionloss", original.dof_frictionloss[did])):
            el.set(attr, number(value))
    for el in root.iter("geom"):
        if "name" not in el.attrib:
            continue
        geom = original.geom(el.attrib["name"])
        for attr in ("pos", "quat", "size", "friction", "solref", "solimp",
                     "margin", "gap", "contype", "conaffinity", "condim", "priority"):
            value = getattr(geom, attr)
            if (attr in ("pos", "quat")
                    and original.geom_type[geom.id] == mujoco.mjtGeom.mjGEOM_MESH):
                # Compiled mesh geoms are expressed in the mesh principal-axis
                # frame. MJCF geom poses precede that compiler transform;
                # writing geom_pos/quat directly would apply it twice.
                mesh_id = int(original.geom_dataid[geom.id])
                mesh_inverse = original.mesh_quat[mesh_id].copy()
                mesh_inverse[1:] *= -1
                xml_quat = np.empty(4)
                mujoco.mju_mulQuat(xml_quat, geom.quat, mesh_inverse)
                rotated_offset = np.empty(3)
                mujoco.mju_rotVecQuat(rotated_offset, original.mesh_pos[mesh_id], xml_quat)
                value = geom.pos - rotated_offset if attr == "pos" else xml_quat
            if attr in ("contype", "conaffinity", "condim", "priority"):
                el.set(attr, str(int(np.asarray(value).reshape(-1)[0])))
            else:
                el.set(attr, number(value))
    actuator = root.find("actuator")
    if actuator is None:
        raise ValueError("Missing compiled actuators")
    for el in actuator:
        act = original.actuator(el.attrib["name"])
        for attr in ("gainprm", "biasprm", "dynprm", "forcerange", "ctrlrange"):
            el.set(attr, number(getattr(act, attr)))

    prefix = "robot/"
    def physical_name(name):
        return prefix + name

    bodies = {el.attrib.get("name"): el for el in root.iter("body")}
    for name, site_name in ((PROFILE["root_body"], "x2_root_imu"),
                            (PROFILE["anchor_body"], "x2_anchor_imu")):
        ET.SubElement(bodies[physical_name(name)], "site", name=site_name,
                      pos="0 0 0", size="0.001")
    sensors = root.find("sensor")
    if sensors is None:
        sensors = ET.SubElement(root, "sensor")
    ET.SubElement(sensors, "velocimeter", name="x2_root_local_linvel", site="x2_root_imu")
    ET.SubElement(sensors, "gyro", name="x2_anchor_gyro", site="x2_anchor_imu")
    qpos = original.qpos0.copy()
    qpos[2] = PROFILE["standing_height"]
    for name, value in PROFILE["default_joint_pos"].items():
        qpos[original.joint(physical_name(name)).qposadr[0]] = value
    ctrl = qpos[original.jnt_qposadr[original.actuator_trnid[:, 0]]]
    keys = root.find("keyframe")
    if keys is not None:
        root.remove(keys)
    keys = ET.SubElement(root, "keyframe")
    ET.SubElement(keys, "key", name="stand", qpos=number(qpos), ctrl=number(ctrl))
    xml = a.out / "x2-scene.xml"
    ET.ElementTree(root).write(xml, encoding="unicode")
    restored = mujoco.MjModel.from_xml_path(str(xml))
    errors = {}
    fields = ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat",
              "jnt_pos", "jnt_axis", "jnt_range", "jnt_stiffness", "dof_armature",
              "dof_damping", "dof_frictionloss", "actuator_gainprm", "actuator_biasprm",
              "actuator_forcerange", "actuator_ctrlrange", "geom_pos", "geom_quat", "geom_size",
              "geom_friction", "geom_solref", "geom_solimp", "geom_contype", "geom_conaffinity")
    for field in fields:
        src, dst = getattr(original, field), getattr(restored, field)
        errors[field] = float(np.max(np.abs(src - dst), initial=0))
        np.testing.assert_allclose(dst, src, rtol=0, atol=1e-10, err_msg=field)

    # SuperMimic exports non-world bodies only; UniLab's motion body IDs are
    # MuJoCo model body IDs. Explicit name remapping and a world row are required.
    with np.load(a.motion, allow_pickle=False) as data:
        source = {k: data[k] for k in data.files}
    source_names = [str(n) for n in source["body_names"].tolist()]
    if len(source_names) != len(set(source_names)):
        raise ValueError("Duplicate source body names")
    target_names = [restored.body(i).name for i in range(restored.nbody)]
    frames = source["joint_pos"].shape[0]
    motion = {}
    mapping = []
    mapped_robot_rows = []
    static_rows = []
    static_data = mujoco.MjData(restored)
    # Configuration/FK only; no integration or trajectory rollout.
    mujoco.mj_forward(restored, static_data)
    for target_id, name in enumerate(target_names[1:], start=1):
        if name.startswith(prefix):
            stripped = name.removeprefix(prefix)
            if stripped not in source_names:
                raise ValueError(f"Robot body {name} absent from source motion")
            source_id = source_names.index(stripped)
            mapping.append(source_id)
            mapped_robot_rows.append((target_id, source_id))
        else:
            # Static scene bodies such as mjlab's terrain are absent from the
            # robot-only motion. Fail closed if a nonrobot body is articulated.
            ancestor = target_id
            while ancestor:
                if restored.body_jntnum[ancestor]:
                    raise ValueError(f"Unmapped articulated scene body {name}")
                ancestor = int(restored.body_parentid[ancestor])
            mapping.append(None)
            static_rows.append(target_id)
    for field, width in (("body_pos_w", 3), ("body_quat_w", 4),
                         ("body_lin_vel_w", 3), ("body_ang_vel_w", 3)):
        array = np.zeros((frames, restored.nbody, width), dtype=np.float32)
        if width == 4:
            array[:, 0, 0] = 1
        for target_id, source_id in mapped_robot_rows:
            array[:, target_id] = source[field][:, source_id]
        if field in ("body_pos_w", "body_quat_w"):
            values = static_data.xpos if field == "body_pos_w" else static_data.xquat
            for target_id in static_rows:
                array[:, target_id] = values[target_id]
        motion[field] = array
    joints = [restored.joint(i).name for i in range(1, restored.njnt)]
    source_joint_names = [str(n) for n in source["joint_names"].tolist()]
    joint_map = [source_joint_names.index(n.removeprefix(prefix)) for n in joints]
    for field in ("joint_pos", "joint_vel"):
        motion[field] = source[field][:, joint_map]
    motion.update(fps=source["fps"], body_names=np.array(target_names), joint_names=np.array(joints))
    npz = a.out / "motion.npz"
    np.savez(npz, **motion)
    metadata = dict(robot="x2_ultra", scene=str(xml), motion=str(npz),
                    joints=joints, actuators=[restored.actuator(i).name for i in range(restored.nu)],
                    tracked_bodies=[physical_name(n) for n in PROFILE["tracked_bodies"]],
                    root_body=physical_name(PROFILE["root_body"]),
                    anchor_body=physical_name(PROFILE["anchor_body"]),
                    action_scale={physical_name(k): v for k, v in PROFILE["action_scale"].items()},
                    source_motion=str(a.motion.resolve()), source_motion_sha256=digest(a.motion),
                    motion_sha256=digest(npz), scene_sha256=digest(xml),
                    source_body_indices=mapping, source_joint_indices=joint_map,
                    static_scene_body_rows={target_names[i]: i for i in static_rows},
                    static_scene_body_source="Restored scene MuJoCo FK at qpos0; zero velocities; no simulation stepping",
                    physical_parameter_max_errors=errors,
                    note="Shared X2 physical scene with official UniLab FlashSAC tracking task terms; not SuperMimic algorithm parity")
    (a.out / "adapter.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
