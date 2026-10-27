"""Check proposed motor commands with independent Warp planning branches.

Only the planning worlds are reinitialized between queries. The executing
robot state is never written here. Passing this short prediction is not a
certificate for a continuous motion; the real trajectory must be audited.
"""
import time
import mujoco
import numpy as np
import torch
from supermimic.envs import SuperMimicEnvCfgArgs,make_env
from supermimic.data.design_metrics import BatchedDesignMetrics
from supermimic.robots.dimensions import PROFILE
from supermimic.kinematics.rotations import quat_apply


class WarpCommandFilter:
    def __init__(self,source,main_model,device='cuda:0',steps=4,candidates=9):
        if steps<1 or candidates<2:raise ValueError('Invalid planning horizon/population')
        self.env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(source)],num_envs=candidates+1,
            action_noise_std=0.,obs_noise=False,reset_noise=False,domain_randomization=False,
            sampling_mode='start',record_transitions=False,episode_length_s=1e5),device)
        self.device=torch.device(device);self.steps=steps;self.candidates=candidates
        self.model=self.env.sim.mj_model;m=self.model
        for name in ['geom_friction','geom_solref','geom_solimp','actuator_gainprm','actuator_biasprm',
                     'actuator_forcerange','jnt_range','body_mass','body_inertia']:
            np.testing.assert_allclose(getattr(m,name),getattr(main_model,name),atol=1e-12,rtol=0)
        for name in ['timestep','integrator','iterations','ls_iterations']:
            if getattr(m.opt,name)!=getattr(main_model.opt,name):raise ValueError('Planning numerical settings differ: '+name)
        if (m.nq,m.nv,m.nu)!=(main_model.nq,main_model.nv,main_model.nu):raise ValueError('Planning dimensions differ')
        tensor=lambda a:torch.as_tensor(a,dtype=torch.float32,device=self.device)
        self.alphas=torch.linspace(0,1,candidates,device=self.device)
        self.low=tensor(m.jnt_range[1:,0]);self.high=tensor(m.jnt_range[1:,1])
        self.velocity=tensor([PROFILE['velocity_limit'][n] for n in PROFILE['joint_names']])
        self.metrics=BatchedDesignMetrics(m,self.device)
        self.foot=m.body('robot/left_ankle_roll_link').id
        self.up=torch.tensor([0.,0.,1.],device=self.device).expand(candidates+1,3)
        self.swing_qpos=int(m.jnt_qposadr[m.joint('robot/right_hip_pitch_joint').id])

    def choose(self,qpos,qvel,proposed,previous,warmstart=None,minimum_foot_pitch=None,minimum_swing_position=None):
        started=time.monotonic();sim=self.env.sim;m=self.model
        tensor=lambda a:torch.as_tensor(np.asarray(a),dtype=torch.float32,device=self.device)
        q,v,new,old=map(tensor,[qpos,qvel,proposed,previous])
        commands=old[None]+self.alphas[:,None]*(new-old)[None]
        # A damping-only position servo is an additional fallback candidate.
        commands=torch.cat([commands,q[7:][None]],dim=0)
        # Separate Warp worlds share the same world-space physical state. This
        # deliberately bypasses origin-relative StateIO in these planning forks.
        sim.data.qpos[:]=q[None];sim.data.qvel[:]=v[None];sim.data.ctrl[:]=commands
        sim.data.qacc_warmstart[:]=0;sim.forward()
        sim.data.qacc_warmstart[:]=0 if warmstart is None else tensor(warmstart)[None]
        count=self.candidates+1;safe=torch.ones(count,dtype=torch.bool,device=self.device)
        minimum=torch.full((count,3),float('inf'),device=self.device)
        first=None
        for _ in range(self.steps):
            sim.step()
            if first is None:first=torch.cat([sim.data.qpos[:],sim.data.qvel[:]],dim=1).clone()
            sim.forward();values=self.metrics(sim.data);joints=sim.data.qpos[:,7:];rates=sim.data.qvel[:,6:]
            minimum=torch.minimum(minimum,values[:,[7,8,9]])
            safe&=(joints>self.low+.01).all(1)&(joints<self.high-.01).all(1)&(abs(rates)<=self.velocity).all(1)
            safe&=(values[:,7:9]>-.004).all(1)&(values[:,9]>.005)
            if minimum_foot_pitch is not None:
                up=quat_apply(sim.data.xquat[:,self.foot],self.up)
                pitch=torch.atan2(up[:,0],up[:,2]);safe&=pitch>=minimum_foot_pitch
            if minimum_swing_position is not None:safe&=sim.data.qpos[:,self.swing_qpos]>=minimum_swing_position
            safe&=torch.isfinite(first).all(1)&torch.isfinite(values).all(1)
        eligible=torch.nonzero(safe[:self.candidates]).flatten()
        if len(eligible):index=int(eligible[-1]);alpha=float(self.alphas[index])
        elif bool(safe[-1]):index=self.candidates;alpha=-1.
        else:
            self.failure={'minimum_heights':minimum.cpu().numpy().tolist(),'safe':safe.cpu().numpy().tolist(),
                'prediction_steps':self.steps,'candidate_alphas':self.alphas.cpu().numpy().tolist()}
            raise RuntimeError('No motor command passed the independent Warp horizon check')
        report={'alpha':alpha,'safe_candidates':int(safe.sum()),'candidate_index':index,
                'predicted_minimum_heights':minimum[index].cpu().numpy().tolist(),
                'prediction_steps':self.steps,'prediction_seconds':float(m.opt.timestep)*self.steps,
                'elapsed_s':time.monotonic()-started,'executing_state_modified':False}
        return commands[index].cpu().numpy().copy(),first[index].cpu().numpy().copy(),report

    def close(self):self.env.close()
