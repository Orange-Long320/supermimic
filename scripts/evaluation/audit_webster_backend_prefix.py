"""Replay a recorded command prefix on GPU and compare with its CPU recording.

No policy/dynamics-network inference is involved. Native measurements use an
independent CPU MjData and never refresh or write the executing GPU simulation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import mujoco
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--cpu-capture',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--shared-pids',nargs='+',type=int,required=True)
    p.add_argument('--envs',type=int,default=256);p.add_argument('--seconds',type=float,default=.16)
    args=p.parse_args()
    if args.out.exists() or args.ckpt.name=='ckpt_latest.pt' or not 1<=args.envs<=256 or not 0<args.seconds<=.2:
        p.error('Use a fresh output, immutable checkpoint and bounded prefix')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.gpu):raise ValueError('Explicit GPU selection is required')
    query=lambda fields:subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu='+fields,'--format=csv,noheader,nounits'],text=True).strip()
    free=int(query('memory.free'))
    owners=[int(x) for x in subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines() if x.strip().isdigit()]
    if free<4000 or not owners or not set(owners)<=set(args.shared_pids):raise ValueError('GPU occupancy differs from the explicit sharing plan')
    if any(Path(f'/proc/{pid}').stat().st_uid!=os.getuid() for pid in owners):raise ValueError('Only this project user may be shared')
    source=json.loads((args.cpu_capture/'result.json').read_text())
    if source['checkpoint_sha256']!=hashlib.sha256(args.ckpt.read_bytes()).hexdigest() or source['source_sha256']!=hashlib.sha256(args.source.read_bytes()).hexdigest():
        raise ValueError('CPU prefix does not match the checkpoint/reference')
    cfg=torch.load(args.ckpt,map_location='cpu',weights_only=False)['config']
    torch.set_num_threads(1);torch.manual_seed(0);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.08)
    args.out.mkdir(parents=True);device='cuda:0'
    from supermimic.envs import SuperMimicEnvCfgArgs,make_env
    from supermimic.data.design_metrics import BatchedDesignMetrics
    env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(args.source)],num_envs=args.envs,
        future_frames=cfg['env']['future_frames'],
        action_noise_std=0.,obs_noise=False,reset_noise=False,domain_randomization=False,
        sampling_mode='start',record_transitions=False,anchor_obs=cfg['env'].get('anchor_obs',False),seed=0),device)
    try:
        env.reset();origin=env.scene.env_origins[0].cpu().numpy()
        model=env.sim.mj_model;native=mujoco.MjData(model);measure=BatchedDesignMetrics(model,'cpu')
        mujoco.mj_saveModel(model,str(args.out/'native-model.mjb'))
        cpu_model=mujoco.MjModel.from_binary_path(str(args.cpu_capture/'native-model.mjb'))
        fields=['jnt_range','actuator_forcerange','actuator_gainprm','actuator_biasprm','body_mass',
            'body_inertia','body_ipos','geom_friction','geom_solref','geom_solimp','geom_condim']
        parameter_errors={k:float(np.max(abs(np.asarray(getattr(model,k))-np.asarray(getattr(cpu_model,k))))) for k in fields}
        runtime_friction=env.sim.model.geom_friction[:].cpu().numpy()
        if runtime_friction.ndim==3:runtime_friction=runtime_friction[0]
        friction_error=float(abs(runtime_friction-cpu_model.geom_friction).max())
        option_fields=['timestep','iterations','ls_iterations','impratio','cone','solver','integrator','enableflags','disableflags']
        option_errors={k:float(abs(getattr(model.opt,k)-getattr(cpu_model.opt,k))) for k in option_fields}
        initial_q=env.sim.data.qpos[0].cpu().numpy().copy();initial_q[:3]-=origin
        initial_v=env.sim.data.qvel[0].cpu().numpy().copy()
        with np.load(args.cpu_capture/'initial-state.npz') as z:
            initial_error=float(abs(np.r_[initial_q,initial_v]-z['state'][0]).max())
        if initial_error>=1e-5:raise ValueError('Initial states differ; no rewrite is permitted')
        with np.load(args.cpu_capture/'trajectory.npz') as z:
            if abs(float(z['action_dt'][0])-env.step_dt)>1e-8:raise ValueError('CPU and GPU control clocks differ')
            steps=round(args.seconds/env.step_dt);actions=z['actions'][:steps].copy()
            expected_q=z['qpos'].copy();expected_v=z['qvel'].copy();expected_metrics=z['metrics'].copy()
        positions=[];velocities=[];metrics=[];original=env.sim.step
        def capture(*a,**kw):
            result=original(*a,**kw)
            q=env.sim.data.qpos[0].cpu().numpy().copy();q[:3]-=origin
            v=env.sim.data.qvel[0].cpu().numpy().copy();positions.append(q);velocities.append(v)
            native.qpos[:]=q;native.qvel[:]=v;mujoco.mj_forward(model,native)
            tensors={k:torch.as_tensor(np.array(getattr(native,k)),dtype=torch.float32)[None]
                for k in ['xpos','xquat','cvel','geom_xpos','geom_xmat','qvel']}
            metrics.append(measure(SimpleNamespace(**tensors))[0].numpy())
            return result
        env.sim.step=capture;started=time.monotonic();failed=False
        with torch.no_grad():
            for action in actions:
                command=torch.tensor(action,device=device).expand(args.envs,-1)
                _,_,term,timeout,_=env.step(command)
                if bool((term|timeout).any()):failed=True;break
        positions=np.asarray(positions);velocities=np.asarray(velocities);metrics=np.asarray(metrics);n=len(positions)
        lo,hi=model.jnt_range[1:].T
        hard=lambda q:float(np.maximum(np.maximum(lo-q[:,7:],q[:,7:]-hi),0).max())
        np.savez(args.out/'trace.npz',qpos=positions,qvel=velocities,metrics=metrics,actions=actions,
            initial_qpos=initial_q,initial_qvel=initial_v,physics_dt=np.array([model.opt.timestep]))
        report={'scope':'Fixed recorded commands, GPU environment zero versus CPU prefix; no motion-success certificate',
            'gpu':args.gpu,'gpu_uuid':query('uuid'),'gpu_occupants_at_start':owners,'free_gpu_mib_at_start':free,
            'num_envs':args.envs,'measured_environment':0,'physics_steps':n,'original_failure_or_timeout':failed,
            'initial_state_maximum_error':initial_error,'native_parameter_maximum_errors':parameter_errors,
            'maximum_runtime_friction_error_vs_cpu_model':friction_error,
            'native_option_errors':option_errors,'maximum_qpos_error':float(abs(positions-expected_q[:n]).max()),
            'maximum_qvel_error':float(abs(velocities-expected_v[:n]).max()),
            'maximum_native_metric_errors':abs(metrics-expected_metrics[:n,:12]).max(0).tolist(),
            'cpu_maximum_hard_violation_rad':hard(expected_q[:n]),'gpu_maximum_hard_violation_rad':hard(positions),
            'elapsed_capture_s':time.monotonic()-started,'torch_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'physical_candidate_passed':False,'approved_for_training':False,
            'recorded_actions_sha256':hashlib.sha256((args.cpu_capture/'baseline-trajectory.npz').read_bytes()).hexdigest()}
        (args.out/'result.json').write_text(json.dumps(report,indent=2));(args.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
        print(json.dumps(report),flush=True)
    finally:env.close()


if __name__=='__main__':main()
