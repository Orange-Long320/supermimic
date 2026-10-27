"""Build bounded midair command ablations from a recorded physical solution."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--solution',type=Path,required=True);p.add_argument('--configuration',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--time',type=float,default=.8)
    p.add_argument('--servo-target',nargs=2,action='append',default=[])
    p.add_argument('--servo-delta',nargs=2,action='append',default=[])
    p.add_argument('--scales',type=float,nargs='+',default=[0,.25,.5,.75,1,1.5,2])
    p.add_argument('--last-knot-fraction',type=float,default=.25)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    from supermimic.robots.dimensions import PROFILE
    groups=json.loads(args.configuration.read_text())['groups'];count=len(groups);names=PROFILE['joint_names']
    with np.load(args.solution) as z:
        parameters=z['parameters'].copy();fps=float(z['fps'].reshape(-1)[0]);index=round(args.time*fps)-1
        action=z['actions'][index].copy();position=z['qpos'][index,7:].copy()
    if parameters.shape!=(1+3*count,):raise ValueError('Expected a three-knot aerial parameter vector')
    scale=PROFILE['action_scale'];scale=np.array([scale[n] for n in names]) if isinstance(scale,dict) else scale
    control=np.array([PROFILE['default_joint_pos'][n] for n in names])+action*scale
    desired={name:float(value) for name,value in args.servo_target}
    for name,value in args.servo_delta:
        if name in desired:raise ValueError('Conflicting target/delta for '+name)
        desired[name]=float(control[names.index(name)]+float(value))
    changes=np.zeros(count);details={}
    for name,target in desired.items():
        group=next(i for i,g in enumerate(groups) if len(g)==1 and g[0][0]==name and g[0][1]==1)
        joint=names.index(name);delta=target-control[joint]
        changes[group]=delta*PROFILE['stiffness'][name]/PROFILE['effort_limit'][name]
        details[name]={'observed_joint_position':float(position[joint]),'original_servo_target':float(control[joint]),
                       'proposed_servo_target':target,'effort_fraction_delta':float(changes[group])}
    candidates=[]
    for amplitude in args.scales:
        row=parameters.copy();row[1+count:1+2*count]+=amplitude*changes
        row[1+2*count:]+=amplitude*args.last_knot_fraction*changes
        row[1:]=np.clip(row[1:],-1.5,1.5);candidates.append(row)
    np.savez(args.out/'parameter-grid.npz',parameters=np.asarray(candidates))
    report={'purpose':__doc__,'sample_end_time_s':args.time,'scales':args.scales,'joint_commands':details,
            'changed_knots':['middle','last'],'last_knot_fraction':args.last_knot_fraction,
            'actual_state_modified':False,'approved_for_training':False,
            'note':'Servo targets are commands, not actual joint positions. The native motors retain their original limits; every candidate still requires full physical checks.',
            'source_sha256':{k:hashlib.sha256(getattr(args,k).read_bytes()).hexdigest() for k in ['solution','configuration']}}
    (args.out/'configuration.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
