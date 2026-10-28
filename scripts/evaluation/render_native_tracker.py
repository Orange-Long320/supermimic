"""Render recorded native feedback states, without running or changing physics."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():p.error('Output already exists')
    report=json.loads((args.trace/'result.json').read_text())
    model_path=Path(report['configuration']['model'])
    if hashlib.sha256(model_path.read_bytes()).hexdigest()!=report['model_sha256']:
        raise ValueError('Recorded model hash mismatch')
    m=mujoco.MjModel.from_binary_path(str(model_path));d=mujoco.MjData(m)
    z=np.load(args.trace/'trajectory.npz');states=z['state'];dt=float(z['dt'])
    stride=max(1,round(1/(50*dt)));fps=1/(stride*dt)
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:]=states[:,:3].mean(axis=0);camera.lookat[2]=.65
    camera.distance=2.7;camera.azimuth=120;camera.elevation=-12
    width=min(960,m.vis.global_.offwidth);height=min(720,m.vis.global_.offheight)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    label="drawtext=text='X2 Ultra | recorded native feedback | no learned policy':x=15:y=15:fontsize=16:fontcolor=white:box=1:boxcolor=black@0.7"
    command=['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
        '-r',str(fps),'-i','pipe:0','-an','-vf',label,'-c:v','libx264','-pix_fmt','yuv420p','-crf','18',
        '-threads','2','-movflags','+faststart','-n',str(args.out)]
    renderer=mujoco.Renderer(m,width=width,height=height)
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        for state in states[::stride]:
            d.qpos[:]=state[:m.nq];d.qvel[:]=state[m.nq:];mujoco.mj_forward(m,d)
            renderer.update_scene(d,camera=camera);process.stdin.write(renderer.render().tobytes())
    finally:
        renderer.close();process.stdin.close()
    if process.wait()!=0:raise RuntimeError('FFmpeg failed')
    args.out.with_suffix('.json').write_text(json.dumps({'trace':str(args.trace),
        'trajectory_sha256':hashlib.sha256((args.trace/'trajectory.npz').read_bytes()).hexdigest(),
        'model_sha256':report['model_sha256'],'video_sha256':hashlib.sha256(args.out.read_bytes()).hexdigest(),
        'frames':len(states[::stride]),'fps':fps,'recorded_native_feedback':True,'learned_policy':False},indent=2))
    print(args.out,flush=True)


if __name__=='__main__':main()
