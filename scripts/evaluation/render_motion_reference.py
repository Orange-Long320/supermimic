"""Render an NPZ robot reference kinematically; no policy or physics rollout."""

import argparse
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", required=True)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists")
    from supermimic.robots.paths import select_robot
    select_robot(args.robot, check_assets=True)
    import mujoco
    import numpy as np
    from supermimic.data.motion_convert import load_g1_model

    model = load_g1_model()
    data = mujoco.MjData(model)
    reference = np.load(args.npz)
    names = reference["body_names"].tolist()
    root_name = model.body(int(model.jnt_bodyid[0])).name
    root = names.index(root_name)
    positions = reference["body_pos_w"][:, root]
    rotations = reference["body_quat_w"][:, root]
    fps = float(reference["fps"].reshape(-1)[0])
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = positions.mean(axis=0)
    camera.lookat[2] = 0.65
    camera.distance = 4.0
    camera.azimuth = 120
    camera.elevation = -12
    args.out.parent.mkdir(parents=True, exist_ok=True)
    width = min(960, model.vis.global_.offwidth)
    height = min(720, model.vis.global_.offheight)
    renderer = mujoco.Renderer(model, width=width, height=height)
    label = "drawtext=text='X2 Ultra | REFERENCE ONLY | no policy rollout':x=15:y=15:fontsize=16:fontcolor=white:box=1:boxcolor=black@0.7"
    command = ["ffmpeg", "-nostdin", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0", "-an", "-vf", label, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-threads", "2", "-movflags", "+faststart", "-n", str(args.out)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for frame in range(len(positions)):
            data.qpos[:3] = positions[frame]
            data.qpos[3:7] = rotations[frame]
            data.qpos[7:] = reference["joint_pos"][frame]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            scene = renderer.scene
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_PLANE,
                               np.array([10, 10, .1]), np.zeros(3), np.eye(3).reshape(-1),
                               np.array([.7, .72, .75, 1], dtype=np.float32))
            scene.ngeom += 1
            rgb = renderer.render().copy()
            process.stdin.write(rgb.tobytes())
    finally:
        renderer.close()
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("FFmpeg failed")
    print(f"Rendered {len(positions)} reference frames at {fps} Hz; EGL device {os.environ.get('MUJOCO_EGL_DEVICE_ID')}")


if __name__ == "__main__":
    main()
