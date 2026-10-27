"""Append terminal goals to isolate frozen moving-reference behavior.

These variants deliberately preserve the original clip exactly. The zero-rate
boundary is discontinuous and is only a frozen-policy diagnostic, not an
approved reference for training or a dynamically certified standing motion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=3.);args=p.parse_args()
    if args.seconds<=0:p.error('Positive diagnostic tail length required')
    args.out.mkdir(parents=True,exist_ok=False)
    with np.load(args.source) as z:data={k:z[k].copy() for k in z.files}
    fps=float(data['fps'].reshape(-1)[0]);count=len(data['joint_pos']);tail=round(args.seconds*fps)
    fields=['joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w']
    rates={'joint_vel','body_lin_vel_w','body_ang_vel_w'}
    for mode in ['repeat_velocity','zero_velocity']:
        out=args.out/mode;out.mkdir();variant={k:v.copy() for k,v in data.items()}
        for key in fields:
            extension=np.repeat(data[key][-1:],tail,axis=0)
            if mode=='zero_velocity' and key in rates:extension[:]=0
            variant[key]=np.concatenate([data[key],extension])
            np.testing.assert_array_equal(variant[key][:count],data[key])
        variant['diagnostic_only']=np.array(True)
        np.savez(out/'webster.npz',**variant)
        metadata={'source':str(args.source),'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  'mode':mode,'original_frames':count,'tail_frames':tail,'fps':fps,'prefix_unchanged':True,
                  'approved_for_training':False,'purpose':__doc__,'pose_contact_feasibility_verified':False}
        (out/'DIAGNOSTIC_ONLY.json').write_text(json.dumps(metadata,indent=2))
    (args.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'out':str(args.out),'original_frames':count,'tail_frames':tail,'approved_for_training':False}))


if __name__=='__main__':main()
