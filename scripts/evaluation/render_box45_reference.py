"""CPU render of aligned box45 diagnostic qpos; never a policy rollout."""
import argparse
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aligned', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output already exists')
    os.environ.setdefault('MUJOCO_GL', 'osmesa')
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra', check_assets=True)
    import mujoco
    import numpy as np
    from supermimic.robots.assets import get_spec
    from supermimic.envs.box_scene import FixedBox

    source = np.load(args.aligned)
    center, half = source['box_center'], source['box_half_size']
    box = FixedBox(front_x=float(center[0]-half[0]),center_y=float(center[1]),
                   depth=float(half[0]*2),width=float(half[1]*2),height=float(half[2]*2))
    spec = get_spec()
    box.add_to_spec(spec)
    if not any(g.type == mujoco.mjtGeom.mjGEOM_PLANE for g in spec.geoms):
        spec.worldbody.add_geom(name='render_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,
                               size=(0,0,.05),rgba=(.8,.8,.8,1))
    spec.worldbody.add_light(pos=(0,-2,4),dir=(0,0,-1))
    model = spec.compile()
    model.vis.global_.offwidth = max(800, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(600, model.vis.global_.offheight)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (center[0]-.25,center[1],.7)
    camera.distance, camera.azimuth, camera.elevation = 3.2, 135, -18
    fps=float(source['fps'])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    label="drawtext=text='BOX45 | REFERENCE DIAGNOSTIC | no policy':x=12:y=12:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.7"
    command=['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','800x600','-r',str(fps),'-i','pipe:0','-an','-vf',label,'-c:v','libx264','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart','-n',str(args.out)]
    with mujoco.Renderer(model,width=800,height=600) as renderer:
        process=subprocess.Popen(command,stdin=subprocess.PIPE)
        try:
            for row in source['qpos']:
                data.qpos[:]=row
                mujoco.mj_forward(model,data)
                renderer.update_scene(data,camera=camera)
                process.stdin.write(renderer.render().tobytes())
        finally:
            process.stdin.close()
        if process.wait()!=0:
            raise RuntimeError('FFmpeg failed')
    print(f'Rendered {len(source["qpos"])} diagnostic frames; no dynamic rollout')


if __name__=='__main__':
    main()
