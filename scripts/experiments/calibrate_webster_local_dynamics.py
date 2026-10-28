"""Bounded calibration of dynamics values and local action differences.

Policy, critic, native physics, acceptance gates and existing jobs are untouched.
Validation holds out complete recorded states, including the diagnostic fixtures.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from supermimic.robots.dimensions import XI_DIM
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModel,WorldModelCfg


def cpu_copy(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_copy(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_copy(v) for v in value)
    return deepcopy(value)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--shared-pids',type=int,nargs='+',required=True)
    p.add_argument('--steps',type=int,default=1500);p.add_argument('--max-seconds',type=float,default=180.)
    p.add_argument('--batch',type=int,default=256);p.add_argument('--lr',type=float,default=5e-5)
    p.add_argument('--derivative-weight',type=float,default=1.)
    args=p.parse_args()
    if args.out.exists() or args.ckpt.name=='ckpt_latest.pt':p.error('Use a fresh output and immutable source')
    if not 1<=args.steps<=3000 or not 0<args.lr<=1e-4 or not 1<=args.batch<=512 or not 0<=args.derivative_weight<=10:
        p.error('Invalid bounded calibration settings')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.gpu):raise ValueError('Explicit physical GPU selection is required')
    def query(kind,fields):
        return subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-'+kind+'='+fields,'--format=csv,noheader,nounits'],text=True).strip()
    free=int(query('gpu','memory.free'));owners=[int(line) for line in query('compute-apps','pid').splitlines() if line.strip().isdigit()]
    if free<4000 or not owners or not set(owners)<=set(args.shared_pids):raise ValueError('GPU occupants or headroom differ from the explicit sharing plan')
    if any(Path(f'/proc/{pid}').stat().st_uid!=os.getuid() for pid in owners):raise ValueError('Sharing is limited to processes owned by this project user')
    torch.set_num_threads(1);torch.manual_seed(0);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.1)
    args.out.mkdir(parents=True);device='cuda:0'
    report=json.loads((args.data/'result.json').read_text());dataset=args.data/'transitions.npz'
    if not report['source_alignment_passed'] or report['dataset_sha256']!=hashlib.sha256(dataset.read_bytes()).hexdigest():
        raise ValueError('Native dataset alignment or checksum failed')
    ck=torch.load(args.ckpt,map_location='cpu',weights_only=False)
    config=dict(ck['config']['world_model']);config['state_spec']=StateSpecCfg(**config['state_spec'])
    wm=WorldModel(WorldModelCfg(**config),None,device);wm.load_state_dict(ck['wm'])
    wm.term_head.requires_grad_(False)
    opt=torch.optim.Adam(wm.parameters(),lr=args.lr)
    z=np.load(dataset)
    tensors={k:torch.tensor(z[k],dtype=torch.float32,device=device) for k in ['states','actions','base','plus','minus','directions']}
    n,directions=tensors['directions'].shape[:2];epsilon=float(z['epsilon'][0])
    train=torch.tensor(np.flatnonzero(z['training']),device=device)
    validation=torch.tensor(np.flatnonzero(~z['training']),device=device)
    if not len(train) or not len(validation):raise ValueError('Whole-state holdout split is required')
    def loss_for(ids,di):
        state=tensors['states'][ids];action=tensors['actions'][ids];direction=tensors['directions'][ids,di]
        xi=torch.zeros((len(ids),XI_DIM),device=device);xi[:,0]=report['context_foot_friction']
        predicted=wm.step(state,action,xi)
        high=wm.step(state,action+epsilon*direction,xi);low=wm.step(state,action-epsilon*direction,xi)
        state_error=wm.spec.error_vector(predicted,tensors['base'][ids]).square().mean()
        predicted_difference=wm.spec.error_vector(high,low)/(2*epsilon)
        actual_difference=wm.spec.error_vector(tensors['plus'][ids,di],tensors['minus'][ids,di])/(2*epsilon)
        derivative_error=(predicted_difference-actual_difference).square().mean()
        return state_error,derivative_error
    @torch.no_grad()
    def evaluate(ids):
        ii=ids.repeat_interleave(directions);jj=torch.arange(directions,device=device).repeat(len(ids));values=[]
        for start in range(0,len(ii),args.batch):
            last=min(start+args.batch,len(ii));a,b=loss_for(ii[start:last],jj[start:last]);values.append(torch.stack([a,b])*(last-start))
        value=torch.stack(values).sum(0)/len(ii)
        return {'state_error':float(value[0]),'action_difference_error':float(value[1])}
    initial=evaluate(validation);initial_train=evaluate(train)
    initial_by_state={int(z['indices'][i]):evaluate(torch.tensor([i],device=device)) for i in validation.tolist()}
    history=[];best=None;best_score=float('inf');started=time.monotonic()
    metadata={'arguments':vars(args),'source_sha256':hashlib.sha256(args.ckpt.read_bytes()).hexdigest(),
        'dataset_sha256':report['dataset_sha256'],'gpu_occupants_at_start':owners,'free_gpu_mib_at_start':free,
        'torch_memory_fraction':.1,'policy_and_critic_unchanged':True,'native_physics_unchanged':True,
        'initial_validation':initial,'initial_training':initial_train,
        'initial_validation_by_state':initial_by_state,'approved_for_training':False,'physical_candidate_passed':False}
    (args.out/'metadata.json').write_text(json.dumps(metadata,indent=2,default=str))
    (args.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    for step in range(1,args.steps+1):
        ids=train[torch.randint(len(train),(args.batch,),device=device)];di=torch.randint(directions,(args.batch,),device=device)
        a,b=loss_for(ids,di);loss=a+args.derivative_weight*b
        if not torch.isfinite(loss):raise ValueError('Non-finite calibration loss')
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(wm.parameters(),1.);opt.step()
        if step%100==0 or step==args.steps or time.monotonic()-started>args.max_seconds:
            value=evaluate(validation);score=value['state_error']+args.derivative_weight*value['action_difference_error']
            record={'step':step,'elapsed_s':time.monotonic()-started,'validation':value}
            history.append(record);print(json.dumps(record),flush=True)
            if value['state_error']<initial['state_error'] and value['action_difference_error']<initial['action_difference_error'] and score<best_score:
                best_score=score;best=(cpu_copy(wm.state_dict()),cpu_copy(opt.state_dict()),record)
            (args.out/'history.json').write_text(json.dumps(history,indent=2))
        if time.monotonic()-started>args.max_seconds:break
    final_validation=evaluate(validation);final_training=evaluate(train)
    final_by_state={int(z['indices'][i]):evaluate(torch.tensor([i],device=device)) for i in validation.tolist()}
    torch.save({'wm':cpu_copy(wm.state_dict()),'config':ck['config'],'model_evaluation_only':True,
        'approved_for_training':False,'local_dynamics_calibration':metadata},args.out/'diagnostic_final_model.pt')
    result={'steps_completed':step,'elapsed_s':time.monotonic()-started,'initial_validation':initial,
        'initial_training':initial_train,'final_training':final_training,'final_validation':final_validation,
        'final_validation_by_state':final_by_state,
        'both_validation_errors_improved':best is not None,'physical_candidate_passed':False,'approved_for_training':False,
        'torch_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'torch_peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20}
    if best is not None:
        ck['wm']=best[0];ck['wm_opt']=best[1]
        ck['local_dynamics_calibration']={'metadata':metadata,'selected_record':best[2],
            'environment_iterations_and_policy_update_counters_unchanged':True}
        torch.save(ck,args.out/'calibrated_model.pt')
        result['selected_record']=best[2]
        result['checkpoint_sha256']=hashlib.sha256((args.out/'calibrated_model.pt').read_bytes()).hexdigest()
    (args.out/'completed.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
