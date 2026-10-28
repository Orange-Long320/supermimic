"""Compile and export an isolated X2+box diagnostic scene on the server.

Does not modify the baseline tracking configuration or approve policy training.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front-x", type=float, required=True, help="Explicit box front in reference frame")
    parser.add_argument("--center-y", type=float, default=0.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists")
    from supermimic.robots.paths import select_robot
    select_robot("x2_ultra", check_assets=True)
    import mujoco
    import numpy as np
    from mjlab.scene import Scene, SceneCfg
    from mjlab.terrains import TerrainEntityCfg
    from supermimic.robots.assets import get_robot_cfg
    from supermimic.envs.box_scene import FixedBox

    task_path = Path(__file__).resolve().parents[2] / "configs/video/box45-IMG4868.json"
    task = json.loads(task_path.read_text())
    cfg = task["box"]
    box = FixedBox(front_x=args.front_x, center_y=args.center_y,
                   depth=cfg["depth_m"], width=cfg["front_width_m"],
                   height=cfg["height_m"], friction=cfg["friction"]["value"])
    scene = Scene(SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"),
                          entities={"robot": get_robot_cfg()}), device="cpu")
    scene.spec.option.timestep = 0.005
    baseline = scene.compile()
    box.add_to_spec(scene.spec)
    model = scene.compile()
    assert (model.nq, model.nv, model.nu, model.nbody) == (baseline.nq, baseline.nv, baseline.nu, baseline.nbody)
    for name in ("body_mass", "body_inertia", "jnt_range", "actuator_ctrlrange", "actuator_forcerange", "actuator_gainprm", "actuator_biasprm"):
        np.testing.assert_array_equal(getattr(model, name), getattr(baseline, name))
    geom = model.geom("box45_collision")
    np.testing.assert_allclose(geom.size, box.half_size, atol=1e-12)
    np.testing.assert_allclose(geom.pos, box.center, atol=1e-12)
    assert int(geom.bodyid[0]) == 0, "Box must be static"
    assert int(geom.contype[0]) == 1 and int(geom.conaffinity[0]) == 1
    # Audit actual contact generation with a free sphere at the top and front;
    # do not move the robot or claim dynamic step-up success.
    probes = []
    for label, point in [
        ("top", (box.center[0], box.center_y, box.height + 0.019)),
        ("front", (box.front_x - 0.019, box.center_y, box.height / 2)),
    ]:
        spec = scene.spec.copy()
        body = spec.worldbody.add_body(name="contact_probe", pos=point)
        body.add_freejoint()
        body.add_geom(name="probe_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                      size=(0.02, 0, 0), mass=0.01, contype=1, conaffinity=0)
        probe_model = spec.compile()
        data = mujoco.MjData(probe_model)
        mujoco.mj_forward(probe_model, data)
        ids = {probe_model.geom("box45_collision").id, probe_model.geom("probe_geom").id}
        distances = [float(c.dist) for c in data.contact if {int(c.geom1), int(c.geom2)} == ids]
        if not distances:
            raise ValueError(f"Missing physical box contact: {label}")
        probes.append({"surface": label, "contact_distances_m": distances})
    args.out.mkdir(parents=True, exist_ok=False)
    scene.write(args.out / "scene")
    report = {"box": asdict(box), "nq": model.nq, "nv": model.nv, "nu": model.nu,
              "timestep_s": model.opt.timestep, "robot_inertia_limits_actuators_unchanged": True,
              "contact_probes": probes, "task_sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
              "placement_status": "explicit diagnostic placement; not yet aligned to recovered motion",
              "status": "compiled geometry and contact probes only; not robot dynamics or task success"}
    (args.out / "scene-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
