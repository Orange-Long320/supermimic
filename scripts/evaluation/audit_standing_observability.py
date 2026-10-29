"""Demonstrate task-state aliasing in the actual standing actor's observation."""
from pathlib import Path
import argparse,json

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['checkpoint','reference','out']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();assert not a.out.exists()
 import torch,numpy as np
 from supermimic.eval.benchmark import load_policy_from_checkpoint
 from supermimic.world_model.heads import analytic_observation,flatten_history_term_major
 from supermimic.robots.dimensions import STATE_DIM,NQ,XI_DIM
 torch.set_num_threads(2);policy,cfg=load_policy_from_checkpoint(a.checkpoint,'cpu');q=np.load(a.reference)['joint_pos'][0];inv=np.load(a.reference.parent/'standing_inverse_target.npz');s=torch.zeros(6,STATE_DIM);s[:,2]=.64;s[:,3]=1.;s[:,7:NQ]=torch.tensor(q)
 s[0,0]=.1;s[1,0]=-.1;s[2,3]=s[3,3]=np.cos(.1);s[2,6]=np.sin(.1);s[3,6]=-np.sin(.1);s[4,NQ]=.02;s[5,NQ]=-.02
 previous=torch.zeros(6,len(q));xi=torch.zeros(6,XI_DIM);xi[:,0]=1.;default=torch.tensor(list(q))
 obs=analytic_observation(s,previous,xi,default);history=flatten_history_term_major(obs.expand(cfg['policy']['history'],-1,-1));command=torch.zeros(6,3)
 with torch.no_grad():action=policy.velocity_action(history,command)
 report=dict(checkpoint=str(a.checkpoint),pairs=['opposite_position_errors','opposite_heading_errors','opposite_planar_velocities'],all_observations_exactly_equal=bool(torch.equal(history,history[:1].expand_as(history))),max_action_difference=float((action-action[:1]).abs().max()),root_states=s[:,:7].tolist(),world_linear_velocity=s[:,NQ:NQ+3].tolist(),conclusion='The actor receives no planar velocity or position/heading error relative to the stop target. Identical proprioceptive histories cannot select opposite corrections for these task states.',scope='Representational counterexample; does not on its own prove a new policy will succeed.')
 assert report['all_observations_exactly_equal'] and report['max_action_difference']==0
 a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
