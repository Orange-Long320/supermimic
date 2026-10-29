"""Run X2 through the fixed official UniLab FlashSAC Manager/runtime stack."""

import argparse
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--unilab-root", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--backend", choices=["mujoco", "mjwarp"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("overrides", nargs="*")
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    adapter = json.loads((a.prepared / "adapter.json").read_text())
    root = a.unilab_root.resolve()
    sys.path.insert(0, str(root / "src"))
    package_dir = str(Path(__file__).resolve().parent)
    os.environ["PYTHONPATH"] = os.pathsep.join([package_dir, str(root / "src")])
    os.environ["UNILAB_EXTRA_REGISTRY_PACKAGES"] = "x2_unilab"
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict
    from unilab.scripts.train_offpolicy import main as train_offpolicy

    with initialize_config_dir(version_base="1.3", config_dir=str(root / "src/unilab/conf/flashsac")):
        cfg = compose(config_name="config", overrides=[
            f"task=g1_motion_tracking/{a.backend}", *a.overrides])
    with open_dict(cfg):
        cfg.training.task_name = "X2UltraMotionTracking"
        cfg.training.log_dir = str(a.out.resolve())
        cfg.training.no_play = True
        cfg.training.export_onnx = False
        cfg.env.scene.model_file = adapter["scene"]
        cfg.env.scene.default_keyframe_name = "stand"
        cfg.env.scene.entities.robot = dict(root_body_name=adapter["root_body"],
            joint_names=adapter["joints"], actuator_names=adapter["actuators"],
            body_names=adapter["tracked_bodies"])
        cfg.env.sim_dt = 0.005
        cfg.env.ctrl_dt = 0.02
        cfg.env.mjwarp_nconmax = 256
        cfg.env.mjwarp_njmax = 1024
        cfg.env.actions.joint_pos.scale = adapter["action_scale"]
        command = cfg.env.commands.motion.params
        command.motion_file = adapter["motion"]
        command.anchor_body_name = adapter["anchor_body"]
        command.body_names = adapter["tracked_bodies"]
        # A deterministic reset baseline for both backends, matching DR-free
        # owner intent and avoiding uncalibrated X2 reset perturbations.
        command.pose_range = {}
        command.velocity_range = {}
        command.joint_position_range = [0.0, 0.0]
        for group in (cfg.env.observations.actor, cfg.env.observations.critic):
            group.terms.base_lin_vel.params.sensor_name = "x2_root_local_linvel"
            group.terms.base_ang_vel.params.sensor_name = "x2_anchor_gyro"
        cfg.env.observations.critic.terms.sac_base_lin_vel.params.sensor_name = "x2_root_local_linvel"
        # The original G1 end-effector body names do not apply to X2.
        hands_and_feet = [n for n in adapter["tracked_bodies"]
                          if "ankle" in n or "wrist_roll" in n]
        cfg.env.terminations.ee_body_pos.params.body_names = hands_and_feet
        cfg.reward.undesired_contacts.params.body_names = [
            n for n in adapter["tracked_bodies"] if n not in hands_and_feet]
    a.out.mkdir(parents=True)
    OmegaConf.save(cfg, a.out / "resolved-x2-config.yaml", resolve=True)
    (a.out / "adapter.json").write_text(json.dumps(adapter, indent=2) + "\n")
    # Direct official implementation, identical to train_flashsac.py after
    # Hydra composition. Registry/EnvFactory remains pickleable for collectors.
    train_offpolicy(cfg)


if __name__ == "__main__":
    main()
