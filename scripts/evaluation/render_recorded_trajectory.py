"""Render saved physical qpos without rerunning or altering the trajectory."""
import argparse
import os
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trajectory',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--playback-speed',type=float,default=.5)
    p.add_argument('--azimuth',type=float,default=90.)
    p.add_argument('--label',default='RECORDED PHYSICS | FAILED CANDIDATE',help='Display label; does not change recorded motion')
    p.add_argument('--distance',type=float,default=3.5)
    args=p.parse_args()
    if args.out.exists():p.error('Output already exists')
    if not 0<args.playback_speed<=2 or args.distance<=0:p.error('Invalid playback speed or camera distance')
    if any(c in args.label for c in "':\\"):p.error('Label cannot contain quotes, colon or backslash')
    os.environ.setdefault('MUJOCO_GL','egl')
    import mujoco
    import numpy as np
    from supermimic.data.motion_convert import g1_xml_path
    # This model is for display only. A compiled floor and lights avoid the
    # clipping artefacts of appending an unlit infinite plane to MjvScene.
    spec=mujoco.MjSpec.from_file(str(g1_xml_path()))
    spec.add_texture(name='video_grid',type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,rgb1=[.24,.28,.33],rgb2=[.34,.38,.43],width=256,height=256)
    textures=['']*10;textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)]='video_grid'
    spec.add_material(name='video_floor_material',textures=textures,texrepeat=[2,2],texuniform=True,
        rgba=[1,1,1,1],reflectance=0.)
    spec.worldbody.add_geom(name='video_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0,0,.1],material='video_floor_material',contype=0,conaffinity=0)
    spec.worldbody.add_light(name='video_light',pos=[0,-3,4],dir=[0,0,-1],
        ambient=[.3,.3,.3],diffuse=[.7,.7,.7],castshadow=False)
    m=spec.compile();d=mujoco.MjData(m)
    with np.load(args.trajectory) as z:
        q=z['qpos'];fps=float(z['fps'].reshape(-1)[0])
    if q.ndim!=2 or q.shape[1]!=m.nq or not np.isfinite(q).all():
        raise ValueError('Expected finite, origin-free positions of this robot')
    width,height=960,720
    m.vis.global_.offwidth=max(width,m.vis.global_.offwidth)
    m.vis.global_.offheight=max(height,m.vis.global_.offheight)
    cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance=args.distance;cam.azimuth=args.azimuth;cam.elevation=-12
    cam.lookat[:]=np.r_[q[:,:2].mean(axis=0),.65]
    renderer=mujoco.Renderer(m,width=width,height=height)
    label=f"drawtext=text='{args.label} | {args.playback_speed}x':x=15:y=15:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.7"
    args.out.parent.mkdir(parents=True,exist_ok=True)
    writer=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{width}x{height}','-r',str(fps*args.playback_speed),'-i','pipe:0','-an','-vf',label,
        '-c:v','libx264','-pix_fmt','yuv420p','-crf','20','-threads','2','-movflags','+faststart','-n',str(args.out)],stdin=subprocess.PIPE)
    try:
        for pose in q:
            cam.lookat[:2]=pose[:2]
            d.qpos[:]=pose;mujoco.mj_forward(m,d);renderer.update_scene(d,camera=cam)
            writer.stdin.write(renderer.render().tobytes())
    finally:
        renderer.close();writer.stdin.close()
    if writer.wait():raise RuntimeError('Video encoding failed')
    print(f'Rendered {len(q)} recorded states at {fps} Hz; no dynamics recomputed')


if __name__=='__main__':main()
