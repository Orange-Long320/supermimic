"""Fit an existing policy to recorded hybrid commands and source feedback.

The example is an unqualified prefix or additive-residual controller. Only the existing
decoder is fitted; the motion encoder and reference NPZ remain unchanged.
Command-fit errors do not certify closed-loop physics or motion success.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.robots.dimensions import PROFILE


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt',type=Path,required=True);p.add_argument('--demonstration',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--shared-pids',type=int,nargs='+')
    p.add_argument('--allow-idle-gpu',action='store_true',help='Require an empty GPU instead of sharing explicit project processes')
    p.add_argument('--residual-expert',action='store_true',help='Fit a recorded additive-residual controller instead of a recorded command prefix')
    p.add_argument('--steps',type=int,default=1000);p.add_argument('--batch',type=int,default=256)
    p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--late-weight',type=float,default=4.)
    p.add_argument('--max-seconds',type=float,default=180.)
    p.add_argument('--initialize',type=Path,help='An earlier immutable imitation policy from this same source')
    p.add_argument('--on-policy-captures',type=Path,nargs='*',default=[],help='Actual pure-policy observations to label with the source expert')
    args=p.parse_args()
    if args.allow_idle_gpu==bool(args.shared_pids):p.error('Choose an idle GPU or explicit shared PIDs')
    if args.out.exists() or args.ckpt.name=='ckpt_latest.pt':p.error('Use a fresh output and immutable source')
    if args.initialize and args.initialize.name=='ckpt_latest.pt':p.error('Initialization must be immutable')
    if not 100<=args.steps<=3000 or not 2<=args.batch<=512 or args.batch%2 or not 0<args.lr<=3e-5 or not 1<=args.late_weight<=10:
        p.error('Invalid bounded command-imitation configuration')
    if args.residual_expert and args.batch<6:p.error('Residual imitation needs nonempty preparation/landing/tail strata')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.gpu):raise ValueError('Explicit physical GPU selection is required')
    query=lambda fields:subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu='+fields,'--format=csv,noheader,nounits'],text=True).strip()
    owners=[int(x) for x in subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines() if x.strip().isdigit()]
    free=int(query('memory.free'))
    occupancy_ok=not owners if args.allow_idle_gpu else bool(owners) and set(owners)<=set(args.shared_pids)
    if free<4000 or not occupancy_ok:raise ValueError('GPU occupants or headroom differ from the explicit allocation')
    if any(Path(f'/proc/{pid}').stat().st_uid!=os.getuid() for pid in owners):raise ValueError('Only explicitly selected project-user processes may be shared')
    demonstration=json.loads((args.demonstration/'result.json').read_text())
    source_sha=hashlib.sha256(args.ckpt.read_bytes()).hexdigest()
    if demonstration['checkpoint_sha256']!=source_sha or demonstration['policy_alone']:
        raise ValueError('Expected a matching explicitly recorded hybrid demonstration')
    if demonstration['failure_physics_steps'] or demonstration['actions']!=350:
        raise ValueError('Demonstration did not complete seven seconds')
    residual_path=None
    if args.residual_expert:
        if not demonstration.get('recorded_residual') or demonstration.get('recorded_prefix'):
            raise ValueError('Expected a recorded additive residual without a prefix')
        residual_path=Path(demonstration['recorded_residual']['path'])
        if hashlib.sha256(residual_path.read_bytes()).hexdigest()!=demonstration['recorded_residual']['sha256']:
            raise ValueError('Recorded expert residual changed')
        canonical=json.loads((args.demonstration/'canonical-gates.json').read_text())
        if canonical['trajectory_sha256']!=hashlib.sha256((args.demonstration/'trajectory.npz').read_bytes()).hexdigest() or not canonical['uninterrupted_seven_seconds'] or not canonical['gates']['last_one_second_stable']:
            raise ValueError('Residual example needs independent seven-second and stability evidence')
    elif not demonstration.get('recorded_prefix') or demonstration['prefix_reproduction_maximum_qpos_error'] is None or demonstration['prefix_reproduction_maximum_qpos_error']>=2e-5:
        raise ValueError('Recorded preparation was not reproduced')
    reference=Path(demonstration['arguments']['source'])
    if hashlib.sha256(reference.read_bytes()).hexdigest()!=demonstration['source_sha256']:
        raise ValueError('Reference NPZ changed after the demonstration')
    with np.load(args.demonstration/'trajectory.npz') as z:
        actions=z['actions'].copy();physical=z['metrics'].copy();physics_dt=float(z['physics_dt'][0])
    if float(physical[:round(.7/physics_dt),15].max())>=1e-4:raise ValueError('Demonstration did not improve early hard-range feasibility')
    z=np.load(args.demonstration/'baseline-observations.npz')
    if len(z['policy'])!=len(actions) or abs(float(z['fps'][0])-50)>1e-6:raise ValueError('Observation/action alignment differs')
    torch.set_num_threads(1);torch.manual_seed(0);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.1)
    args.out.mkdir(parents=True);device='cuda:0'
    policy,config=load_policy_from_checkpoint(args.ckpt,device)
    policy.encoders.requires_grad_(False);policy.decoder.requires_grad_(True)
    original_encoder={k:v.detach().cpu().clone() for k,v in policy.encoders.state_dict().items()}
    obs=torch.tensor(z['policy'],device=device);goal=torch.tensor(z['goal'],device=device)
    target=torch.tensor(actions,dtype=torch.float32,device=device)
    residual=None;recorded_expert_error=None
    if args.residual_expert:
        with np.load(residual_path) as proposal:
            if str(proposal['source_checkpoint_sha256'])!=source_sha or str(proposal['reference_sha256'])!=demonstration['source_sha256'] or list(proposal['joint_names'])!=list(PROFILE['joint_names']) or abs(float(proposal['fps'][0])-50)>1e-6:
                raise ValueError('Expert residual inputs, joint order or clock differ')
            residual=torch.tensor(proposal['actions'],dtype=torch.float32,device=device)
        if residual.shape!=target.shape or not torch.isfinite(residual).all():raise ValueError('Incomplete residual commands')
        changed=residual.abs().amax(-1)>1e-8
        cut=int(changed.nonzero()[-1])+1 if changed.any() else 35
        if not 35<cut<len(actions):raise ValueError('Residual example needs a nonempty landing and unmodified tail')
        end=cut/50
        with torch.no_grad():recorded_expert_error=float((policy(obs,goal)+residual-target).square().mean())
        if recorded_expert_error>=1e-8:raise ValueError('Recorded actions do not match source feedback plus expert residual')
        if residual[cut:].abs().max()>1e-8:raise ValueError('Late expert commands must be the original feedback')
    else:
        end=demonstration['recorded_prefix']['until_s']+demonstration['recorded_prefix']['blend_s']
    cut=round(end*50)
    with torch.no_grad():source_late_error=float((policy(obs[cut:],goal[cut:])-target[cut:]).square().mean())
    if source_late_error>=1e-6:raise ValueError('Recorded later commands do not match the source policy')
    parts=[{'source':str(args.demonstration),'count':len(actions),'kind':'recorded hybrid commands'}]
    input_files=[args.demonstration/'result.json',args.demonstration/'baseline-observations.npz',args.demonstration/'trajectory.npz']
    if args.residual_expert:input_files.extend([residual_path,args.demonstration/'canonical-gates.json'])
    flags=[torch.arange(len(actions),device=device)<cut]
    preparation_flags=[torch.arange(len(actions),device=device)<35]
    part_ids=[torch.zeros(len(actions),dtype=torch.long,device=device)]
    obs_parts=[obs];goal_parts=[goal];target_parts=[target]
    if args.on_policy_captures:
        if not args.residual_expert:
            prefix_info=demonstration['recorded_prefix']
            if prefix_info['joint_names']!=list(PROFILE['joint_names']):raise ValueError('Expert query currently requires an all-joint prefix')
            prefix_path=Path(prefix_info['trace'])/'baseline-trajectory.npz'
            if hashlib.sha256(prefix_path.read_bytes()).hexdigest()!=prefix_info['actions_sha256']:
                raise ValueError('Recorded prefix commands changed')
            input_files.append(prefix_path)
            prefix=torch.tensor(np.load(prefix_path)['actions'],dtype=torch.float32,device=device)
        for part,capture in enumerate(args.on_policy_captures,1):
            info=json.loads((capture/'result.json').read_text())
            if not info['policy_alone'] or info['source_sha256']!=demonstration['source_sha256'] or info['start_reference_frame']!=0:
                raise ValueError('Additional observations must be pure policies on the same initial reference')
            with np.load(capture/'baseline-observations.npz') as observed:
                if len(observed['policy'])!=info['actions'] or abs(float(observed['fps'][0])-50)>1e-6:
                    raise ValueError('On-policy observations are not aligned to actual actions')
                o=torch.tensor(observed['policy'],device=device);g=torch.tensor(observed['goal'],device=device)
            with torch.no_grad():
                labels=policy(o,g).clone()
                if args.residual_expert:
                    if len(labels)>len(residual):raise ValueError('Learner trace exceeds expert clock')
                    labels+=residual[:len(labels)]
                else:
                    for i in range(min(cut,len(labels))):
                        fraction=min(max((i*.02-prefix_info['until_s'])/max(prefix_info['blend_s'],1e-12),0),1)
                        labels[i]=(1-fraction)*prefix[i]+fraction*labels[i]
            obs_parts.append(o);goal_parts.append(g);target_parts.append(labels)
            flags.append(torch.arange(len(o),device=device)<cut)
            preparation_flags.append(torch.arange(len(o),device=device)<35)
            part_ids.append(torch.full((len(o),),part,dtype=torch.long,device=device))
            parts.append({'source':str(capture),'count':len(o),'kind':'source-expert queries on actually executed learner states',
                'queried_commands_were_executed':False})
            input_files.extend([capture/'result.json',capture/'baseline-observations.npz',capture/'baseline-trajectory.npz'])
    obs=torch.cat(obs_parts);goal=torch.cat(goal_parts);target=torch.cat(target_parts)
    early_flags=torch.cat(flags);sample_parts=torch.cat(part_ids)
    early=early_flags.nonzero().flatten();late=(~early_flags).nonzero().flatten()
    preparation=torch.cat(preparation_flags).nonzero().flatten()
    landing=(early_flags&~torch.cat(preparation_flags)).nonzero().flatten()
    initialization_sha=None
    if args.initialize:
        initialization=torch.load(args.initialize,map_location=device,weights_only=False)
        if initialization.get('source_checkpoint_sha256')!=source_sha or initialization.get('reference_sha256')!=demonstration['source_sha256']:
            raise ValueError('Initialization is not an imitation policy from this source/reference')
        if args.residual_expert and initialization.get('expert_residual_sha256')!=hashlib.sha256(residual_path.read_bytes()).hexdigest():
            raise ValueError('Initialization uses a different residual expert')
        policy.load_state_dict(initialization['policy'])
        initialization_sha=hashlib.sha256(args.initialize.read_bytes()).hexdigest()
        if not all(torch.equal(v.cpu(),original_encoder[k]) for k,v in policy.encoders.state_dict().items()):
            raise ValueError('Initialization changed the frozen motion encoder')
    scale=PROFILE['action_scale'];names=list(PROFILE['joint_names'])
    scale=torch.tensor([scale[n] for n in names] if isinstance(scale,dict) else [scale]*len(names),device=device)
    @torch.no_grad()
    def errors():
        delta=policy(obs,goal)-target
        result={'early_action_mse':float(delta[early].square().mean()),
            'late_action_mse':float(delta[late].square().mean()),
            'early_servo_target_rms_rad':float((delta[early]*scale).square().mean().sqrt()),
            'late_servo_target_rms_rad':float((delta[late]*scale).square().mean().sqrt())}
        if args.residual_expert:
            result.update(preparation_action_mse=float(delta[preparation].square().mean()),landing_action_mse=float(delta[landing].square().mean()))
        return result
    initial=errors()
    opt=torch.optim.Adam(policy.decoder.parameters(),lr=args.lr)
    metadata={'scope':'Recorded-command imitation only; not a physically qualified teacher or motion certificate',
        'source_checkpoint_sha256':source_sha,'reference_sha256':demonstration['source_sha256'],
        'demonstration_policy_alone':False,'demonstration_physically_qualified':False,
        'demonstration_early_hard_violation_rad':float(physical[:round(.7/physics_dt),15].max()),
        'input_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in input_files},
        'source_late_command_mse':source_late_error,'initialization_sha256':initialization_sha,
        'expert_type':'additive_residual' if args.residual_expert else 'recorded_prefix',
        'expert_residual_sha256':hashlib.sha256(residual_path.read_bytes()).hexdigest() if residual_path else None,
        'recorded_full_expert_command_mse':recorded_expert_error,
        'dataset_parts':parts,'supervision_is_not_an_executed_trajectory':bool(args.on_policy_captures),
        'trained_module':'decoder only','motion_encoder_frozen':True,'reference_npz_unchanged':True,
        'gpu_occupants_at_start':owners,'free_gpu_mib_at_start':free,'torch_memory_fraction':.1,
        'initial_fit_errors':initial,'arguments':vars(args),'physical_candidate_passed':False,'approved_for_training':False}
    (args.out/'metadata.json').write_text(json.dumps(metadata,indent=2,default=str))
    (args.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
    np.savez_compressed(args.out/'command-targets.npz',target_action=target.cpu().numpy(),
        early=early_flags.cpu().numpy(),part_id=sample_parts.cpu().numpy())
    history=[];best=None;best_score=float('inf');started=time.monotonic()
    for step in range(1,args.steps+1):
        half=args.batch//2
        if args.residual_expert:
            third=args.batch//3
            ids=torch.cat([preparation[torch.randint(len(preparation),(third,),device=device)],
                landing[torch.randint(len(landing),(third,),device=device)],late[torch.randint(len(late),(args.batch-2*third,),device=device)]])
        else:ids=torch.cat([early[torch.randint(len(early),(half,),device=device)],late[torch.randint(len(late),(half,),device=device)]])
        prediction=policy(obs[ids],goal[ids]);error=(prediction-target[ids]).square().mean(-1)
        loss=(error[:third].mean()+error[third:2*third].mean()+args.late_weight*error[2*third:].mean()
            if args.residual_expert else error[:half].mean()+args.late_weight*error[half:].mean())
        if not torch.isfinite(loss):raise ValueError('Non-finite command-imitation loss')
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(policy.decoder.parameters(),1.);opt.step()
        if step%100==0 or step==args.steps or time.monotonic()-started>args.max_seconds:
            fit=errors();record={'step':step,'elapsed_s':time.monotonic()-started,'fit_errors':fit}
            history.append(record);print(json.dumps(record),flush=True)
            score=(fit['preparation_action_mse']+fit['landing_action_mse']+args.late_weight*fit['late_action_mse']
                if args.residual_expert else fit['early_action_mse']+args.late_weight*fit['late_action_mse'])
            if score<best_score:
                best_score=score;best=({k:v.detach().cpu().clone() for k,v in policy.state_dict().items()},record)
            (args.out/'history.json').write_text(json.dumps(history,indent=2))
        if time.monotonic()-started>args.max_seconds:break
    if not all(torch.equal(v.cpu(),original_encoder[k]) for k,v in policy.encoders.state_dict().items()):raise ValueError('Frozen motion encoder changed')
    finite=all(torch.isfinite(v).all().item() for v in policy.state_dict().values())
    if not finite or best is None:raise ValueError('No finite fitted policy')
    def checkpoint(weights,record):
        return {'config':config,'policy':weights,'source_checkpoint_sha256':source_sha,
            'reference_sha256':demonstration['source_sha256'],
            'initialization_sha256':initialization_sha,'imitation_data_parts':parts,
            'expert_residual_sha256':metadata['expert_residual_sha256'],
            'optimizer_algorithm':'recorded_residual_behavior_cloning' if args.residual_expert else 'recorded_prefix_behavior_cloning','supervised_fit':record,
            'demonstration_physically_qualified':False,'policy_evaluation_only':True,'approved_for_training':False}
    torch.save(checkpoint(best[0],best[1]),args.out/'policy_evaluation.pt')
    torch.save({'policy':{k:v.detach().cpu() for k,v in policy.state_dict().items()},'optimizer':opt.state_dict(),
        'config':config,'metadata':metadata,'last_step':step,'selected_policy_step':best[1]['step']},args.out/'bc_last_state.pt')
    report={'steps_completed':step,'elapsed_s':time.monotonic()-started,'initial_fit_errors':initial,
        'selected_fit':best[1],'selection_uses_command_fit_not_physical_quality':True,
        'parameters_finite':finite,'motion_encoder_unchanged':True,
        'torch_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'torch_peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
        'evaluation_checkpoint_sha256':hashlib.sha256((args.out/'policy_evaluation.pt').read_bytes()).hexdigest(),
        'physical_evaluation_pending':True,'physical_candidate_passed':False,'approved_for_training':False}
    (args.out/'completed.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
