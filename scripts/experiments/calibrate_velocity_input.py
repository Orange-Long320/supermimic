"""Fold a measured longitudinal command calibration into encoder weights.

This produces an unverified candidate; physical forward/zero/backward tests are mandatory.
"""
import argparse,json,hashlib
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['checkpoint','evaluation','out']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();a.out.mkdir(exist_ok=False)
 import torch,numpy as np
 d=json.loads(a.evaluation.read_text())
 if Path(d['checkpoint']).resolve()!=a.checkpoint.resolve():raise ValueError('Evaluation checkpoint mismatch')
 if d['survived_without_reset']!=d['num_envs']:raise ValueError('Failed physical trials cannot calibrate this interface')
 points=[]
 for lo,hi,c in [(4,10,.3),(11,15,0.),(16,22,-.2)]:
  samples=[s for s in d['samples'] if lo<=s['time_s']<hi]
  if not samples or not all(all(s['active']) for s in samples):raise ValueError('Missing or failed calibration phase')
  points.append([c,float(np.array([s['velocity'] for s in samples])[:,:,0].mean())])
 slope,offset=np.polyfit(np.array(points)[:,0],np.array(points)[:,1],1)
 if not np.isfinite([slope,offset]).all() or slope<=0:raise ValueError('No positive command response to calibrate')
 gain,bias=1/slope,-offset/slope
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
 sd=ck['policy'];w=sd['encoders.velocity.mlp.net.0.weight'];b=sd['encoders.velocity.mlp.net.0.bias']
 column=w[:,-3].clone();scale=sd['encoders.velocity.command_scale'][0]
 b+=column*bias/scale;w[:,-3]*=gain
 ck.pop('velocity_training',None)
 meta=dict(source_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),evaluation=str(a.evaluation),evaluation_sha256=hashlib.sha256(a.evaluation.read_bytes()).hexdigest(),points=points,gain=gain,bias=bias,scope='Input calibration candidate; physical acceptance required; not resumable.')
 ck['input_calibration']=meta;torch.save(ck,a.out/'candidate.pt')
 (a.out/'report.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta))
if __name__=='__main__':main()
