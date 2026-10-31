"""Render a camera-follow replay of logged official simulator poses on the server."""
import argparse
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--start', type=float, default=0.)
    p.add_argument('--end', type=float, default=float('inf'))
    p.add_argument('--width', type=int, default=960)
    p.add_argument('--height', type=int, default=720)
    a = p.parse_args()
    if a.out.exists(): raise ValueError('output exists')
    if a.width <= 0 or a.height <= 0 or a.width % 2 or a.height % 2:
        raise ValueError('Render dimensions must be positive even numbers')
    spec = json.loads(a.profile.read_text()); names = spec['joint_names']; n = len(names)
    model = mujoco.MjModel.from_xml_path(str(a.scene)); data = mujoco.MjData(model)
    adr = []
    for name in names:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0: raise ValueError('missing official model joint '+name)
        adr.append(model.jnt_qposadr[joint])
    default = np.asarray([spec['default_joint_pos'][name] for name in names])
    frames = [row for line in a.trace.open() if a.start <= (row := json.loads(line))['elapsed'] <= a.end]
    if not frames: raise ValueError('no frames in requested interval')
    model.vis.global_.offwidth = a.width; model.vis.global_.offheight = a.height
    renderer = mujoco.Renderer(model, width=a.width, height=a.height)
    camera = mujoco.MjvCamera(); camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = 135; camera.elevation = -17; camera.distance = 2.2
    proc = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{a.width}x{a.height}', '-r', '25', '-i', '-', '-c:v', 'libx264', '-preset', 'fast',
        '-pix_fmt', 'yuv420p', '-y', str(a.out)], stdin=subprocess.PIPE)
    try:
        for frame in frames[::2]:
            data.qpos[:7] = np.r_[frame['root'], frame['q']]
            data.qpos[adr] = frame['joint_pos'] if 'joint_pos' in frame else np.asarray(frame['observation'][9*n:10*n])+default
            mujoco.mj_forward(model, data)
            camera.lookat[:] = [frame['root'][0], frame['root'][1], frame['root'][2]+.1]
            renderer.update_scene(data, camera=camera)
            proc.stdin.write(renderer.render().tobytes())
    finally:
        proc.stdin.close(); code = proc.wait(); renderer.close()
    if code: raise RuntimeError('ffmpeg failed')


if __name__ == '__main__': main()
