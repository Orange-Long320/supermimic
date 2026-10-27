"""Compare independent one-step native/Warp solves at recorded physical states."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import mujoco
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--iterations',type=int,required=True)
    p.add_argument('--device',default='cuda:0');args=p.parse_args()
    if args.iterations<1:p.error('Positive solver budget required')
    args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    from supermimic.envs import SuperMimicEnvCfgArgs,make_env_cfg
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()
    with np.load(args.trace/'trajectory.npz') as z:states=z['state'].copy();controls=z['servo_target'].copy()
    with np.load(args.trace/'initial-state.npz') as z:initial=z['state'][0].copy()
    previous=np.r_[initial[None],states[:-1]];n=len(states)
    cfg=make_env_cfg(SuperMimicEnvCfgArgs(npz_files=[str(args.source)],num_envs=n,action_noise_std=0.,
        obs_noise=False,reset_noise=False,domain_randomization=False,sampling_mode='start',record_transitions=False))
    cfg.sim.mujoco.iterations=args.iterations
    env=ManagerBasedRlEnv(cfg,device=args.device)
    try:
        m=env.sim.mj_model;old=mujoco.MjModel.from_binary_path(str(args.trace/'native-model.mjb'))
        for key in ['geom_friction','geom_solref','geom_solimp','actuator_gainprm','actuator_biasprm','actuator_forcerange','jnt_range','body_mass','body_inertia']:
            np.testing.assert_allclose(getattr(m,key),getattr(old,key),atol=1e-12,rtol=0)
        assert m.opt.timestep==old.opt.timestep and m.opt.integrator==old.opt.integrator
        assert previous.shape[1]==m.nq+m.nv and controls.shape==(n,m.nu)
        # Each world is a separate, independent one-step counterfactual. Put all
        # at the same world origin to isolate solver error from origin rounding.
        env.sim.data.qpos[:]=torch.as_tensor(previous[:,:m.nq],dtype=torch.float32,device=args.device)
        env.sim.data.qvel[:]=torch.as_tensor(previous[:,m.nq:],dtype=torch.float32,device=args.device)
        env.sim.data.ctrl[:]=torch.as_tensor(controls,dtype=torch.float32,device=args.device)
        env.sim.data.qacc_warmstart[:]=0;env.sim.forward();env.sim.data.qacc_warmstart[:]=0
        if args.device.startswith('cuda'):torch.cuda.synchronize()
        started=time.monotonic();env.sim.step()
        if args.device.startswith('cuda'):torch.cuda.synchronize()
        warp_seconds=time.monotonic()-started
        warp=np.c_[env.sim.data.qpos[:].cpu().numpy(),env.sim.data.qvel[:].cpu().numpy()]
        native=[];d=mujoco.MjData(m);started=time.monotonic()
        for state,control in zip(previous,controls):
            d.qpos[:]=state[:m.nq];d.qvel[:]=state[m.nq:];d.ctrl[:]=control;d.qacc_warmstart[:]=0
            mujoco.mj_step(m,d);native.append(np.r_[d.qpos.copy(),d.qvel.copy()])
        native=np.asarray(native);native_seconds=time.monotonic()-started
        difference=warp-native
        summarize=lambda a:{'maximum_absolute':float(abs(a).max()),'rms':float(np.sqrt(np.mean(a*a)))}
        report={'steps':n,'solver_iterations':args.iterations,'device':args.device,'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'native_vs_warp_qpos':summarize(difference[:,:m.nq]),'native_vs_warp_qvel':summarize(difference[:,m.nq:]),
            'worst_step':int(abs(difference[:,m.nq:]).max(axis=1).argmax()),'batched_warp_step_seconds':warp_seconds,
            'serial_native_steps_seconds':native_seconds,'physical_parameters_equal_to_recorded_model':True,
            'scope':'Independent one-step solver diagnostic, no training or continuous-motion approval',
            'trace_sha256':hashlib.sha256((args.trace/'trajectory.npz').read_bytes()).hexdigest(),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'approved_for_training':False}
        np.savez(args.out/'comparison.npz',previous=previous,control=controls,native_next=native,warp_next=warp)
        mujoco.mj_saveModel(m,str(args.out/'diagnostic-model.mjb'))
        (args.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
        (args.out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    finally:env.close()


if __name__=='__main__':main()
