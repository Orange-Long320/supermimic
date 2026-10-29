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
    p.add_argument('--support-height',type=float,default=0.)
    p.add_argument('--preview-fps',type=float,default=50.)
    p.add_argument('--width',type=int,default=960)
    p.add_argument('--height',type=int,default=720)
    p.add_argument('--label',default='X2 Ultra | recorded native feedback | no learned policy')
    p.add_argument('--visual-model',type=Path,help='Optional original visual geometry; physics continues to be identified by the recorded model hash')
    args=p.parse_args()
    if args.out.exists():p.error('Output already exists')
    if not 0<args.preview_fps<=50 or min(args.width,args.height)<160:p.error('Positive rate at most 50 and dimensions at least 160 required')
    report=json.loads((args.trace/'result.json').read_text())
    model_path=Path(report['configuration']['model'])
    if hashlib.sha256(model_path.read_bytes()).hexdigest()!=report['model_sha256']:
        raise ValueError('Recorded model hash mismatch')
    physical=mujoco.MjModel.from_binary_path(str(model_path));m=physical
    if args.visual_model:
        m=mujoco.MjModel.from_binary_path(str(args.visual_model))
        for field in ['nq','nv','nbody','njnt']:
            if getattr(m,field)!=getattr(physical,field):raise ValueError('Visual model layout differs')
        for field in ['jnt_qposadr','jnt_dofadr','jnt_axis','body_pos','body_quat','body_mass','body_inertia']:
            np.testing.assert_allclose(getattr(m,field),getattr(physical,field),rtol=0,atol=1e-12)
    d=mujoco.MjData(m)
    z=np.load(args.trace/'trajectory.npz');states=z['state'];dt=float(z['dt'])
    stride=max(1,round(1/(args.preview_fps*dt)));fps=1/(stride*dt)
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:]=states[:,:3].mean(axis=0);camera.lookat[2]=.65+args.support_height
    camera.distance=2.7;camera.azimuth=120;camera.elevation=-12
    width=min(args.width,m.vis.global_.offwidth);height=min(args.height,m.vis.global_.offheight)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    safe_label=args.label.replace('\\','\\\\').replace("'",'').replace(':','\\:')
    label=f"drawtext=text='{safe_label}':x=8:y=8:fontsize=12:fontcolor=white:box=1:boxcolor=black@0.7"
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
        'visual_model':str(args.visual_model) if args.visual_model else str(model_path),
        'visual_model_sha256':hashlib.sha256((args.visual_model or model_path).read_bytes()).hexdigest(),
        'frames':len(states[::stride]),'fps':fps,'label':args.label,'recorded_native_feedback':True,'learned_policy':False},indent=2))
    print(args.out,flush=True)


if __name__=='__main__':main()
