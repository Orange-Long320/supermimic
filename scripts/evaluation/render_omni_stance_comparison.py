"""Server CPU comparison of isolated original/repaired terminal targets."""
import argparse
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attribution',type=Path,required=True)
    p.add_argument('--repair',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('New output required')
    os.environ.setdefault('MUJOCO_GL','osmesa')
    import mujoco
    import numpy as np
    from PIL import Image,ImageDraw
    canvas=Image.new('RGB',(960,960),'white'); draw=ImageDraw.Draw(canvas)
    for i,(name,row) in enumerate(json.loads(a.attribution.read_text()).items()):
        m=mujoco.MjModel.from_binary_path(row['model']); d=mujoco.MjData(m)
        m.vis.global_.offwidth=480; m.vis.global_.offheight=300
        z=np.load(a.repair/name/'terminal-candidates.npz')
        with mujoco.Renderer(m,width=480,height=300) as renderer:
            for j,label in enumerate(['original','candidate']):
                d.qpos[:]=z[label]; mujoco.mj_forward(m,d)
                camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
                camera.lookat[:]=d.qpos[:3]; camera.lookat[2]-=.25;camera.distance=2.2
                root=int(m.jnt_bodyid[0]); axis=d.xmat[root].reshape(3,3)[:,0]
                camera.azimuth=float(np.degrees(np.arctan2(axis[1],axis[0])))+180;camera.elevation=-20
                renderer.update_scene(d,camera=camera)
                canvas.paste(Image.fromarray(renderer.render()),(j*480,i*320+20))
                draw.text((j*480+8,i*320+5),f'{name} | {label} | terminal target, not policy',fill='black')
    a.out.parent.mkdir(parents=True,exist_ok=True);canvas.save(a.out)
    a.out.with_suffix('.as-run.py').write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':main()
