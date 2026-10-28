"""Experimental physical-motion rewards shared by real and imagined states.

This module changes objectives only when explicitly installed by an experiment.
It never changes the native model, states, actuator limits or acceptance gates.
"""
from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from scipy.spatial import ConvexHull

from supermimic.kinematics.rotations import matrix_from_quat,quat_apply
from supermimic.robots.dimensions import NQ,PROFILE
from supermimic.world_model.heads import AnalyticHeads


@dataclass
class PhysicalRewardCfg:
    foot_weight: float = 100.
    nonfoot_weight: float = 100.
    air_rotation_weight: float = .1
    foot_margin_m: float = -.004
    nonfoot_margin_m: float = .006
    air_height_m: float = .02
    air_gate_width_m: float = .005
    maximum_forward_rate: float = 20.
    hard_joint_weight: float = 0.
    hard_joint_margin_rad: float = .001


class PhysicalGeometry:
    def __init__(self,model,fk,tracked_names,device):
        self.fk=fk;self.device=torch.device(device)
        if fk.body_names!=[model.body(i).name for i in range(1,model.nbody)]:
            raise ValueError('Geometry must use the same single-tree robot as analytic FK')
        if [model.joint(i).name.rsplit('/',1)[-1] for i in range(1,model.njnt)]!=list(PROFILE['joint_names']):
            raise ValueError('Hard joint ranges must match the state joint order')
        self.hard_limits=torch.tensor(model.jnt_range[1:].copy(),dtype=torch.float32,device=device)
        points=[];bodies=[];kinds=[];reference_ids=[]
        tracked_names=list(tracked_names)
        root=int(model.jnt_bodyid[0])
        for g in range(model.ngeom):
            bid=int(model.geom_bodyid[g])
            if model.body_rootid[bid]!=root or not (model.geom_contype[g] or model.geom_conaffinity[g]):continue
            mesh=int(model.geom_dataid[g])
            if mesh<0:raise ValueError('Expected mesh collision hulls')
            first,n=model.mesh_vertadr[mesh],model.mesh_vertnum[mesh]
            vertices=model.mesh_vert[first:first+n].astype(float)
            vertices=vertices[ConvexHull(vertices).vertices]
            rotation=np.zeros(9);mujoco.mju_quat2Mat(rotation,model.geom_quat[g])
            local=vertices@rotation.reshape(3,3).T+model.geom_pos[g]
            foot='foot_' in model.geom(g).name
            kind=(1 if 'left' in model.body(bid).name else 2) if foot else 0
            if foot and model.body(bid).name not in tracked_names:
                raise ValueError('Reference tracking omits a foot collision body')
            reference=tracked_names.index(model.body(bid).name) if foot else -1
            points.extend(local);bodies.extend([bid-1]*len(local));kinds.extend([kind]*len(local));reference_ids.extend([reference]*len(local))
        self.points=torch.tensor(np.asarray(points),dtype=torch.float32,device=device)
        self.bodies=torch.tensor(bodies,dtype=torch.long,device=device)
        self.groups=[torch.tensor(np.flatnonzero(np.array(kinds)==k),device=device) for k in [1,2,0]]
        self.foot_vertices=torch.tensor(np.flatnonzero(np.array(kinds)>0),device=device)
        self.reference_ids=torch.tensor(np.array(reference_ids)[np.array(kinds)>0],device=device)

    def heights(self,pos,quat):
        # Only the world-Z matrix row is needed. Avoid expanded 3D vertex
        # rotations in every imagined step, retaining exact hull heights.
        row=matrix_from_quat(quat)[...,2,:]
        z=(row[...,self.bodies,:]*self.points).sum(-1)+pos[...,self.bodies,2]
        # Indexed minima retain the winning vertices for backward, rather
        # than every hull height at every imagined time step.
        return torch.stack([z[...,g].min(-1).values for g in self.groups],-1)

    def reference_air(self,pos,quat,cfg):
        row=matrix_from_quat(quat)[...,2,:]
        z=(row[...,self.reference_ids,:]*self.points[self.foot_vertices]).sum(-1)+pos[...,self.reference_ids,2]
        return (z.amin(-1)>cfg.air_height_m).to(pos.dtype)

    def joint_excess(self,state,margin=0.):
        q=state[...,7:NQ]
        return (self.hard_limits[:,0]+margin-q).clamp(min=0)+(q-self.hard_limits[:,1]+margin).clamp(min=0)

    def components(self,state,reference_pos,reference_quat,cfg,all_pos=None,all_quat=None):
        if all_pos is None:
            fk=self.fk(state[...,:NQ]);all_pos,all_quat=fk['pos'],fk['quat']
        height=self.heights(all_pos,all_quat)
        foot=(cfg.foot_margin_m-height[...,:2].amin(-1)).clamp(min=0)
        body=(cfg.nonfoot_margin_m-height[...,2]).clamp(min=0)
        clearance=-cfg.foot_weight*foot-cfg.nonfoot_weight*body
        actual_air=torch.sigmoid((height[...,:2].amin(-1)-cfg.air_height_m)/cfg.air_gate_width_m)
        reference_air=self.reference_air(reference_pos,reference_quat,cfg)
        forward=quat_apply(state[...,3:7],state[...,NQ+3:NQ+6])[...,1].clamp(0,cfg.maximum_forward_rate)
        air_rotation=cfg.air_rotation_weight*actual_air*reference_air*forward
        hard_joint=-cfg.hard_joint_weight*self.joint_excess(state,cfg.hard_joint_margin_rad).sum(-1)
        return {'clearance':clearance,'air_rotation':air_rotation,'heights':height,
                'hard_joint':hard_joint,'total':clearance+air_rotation+hard_joint,'reference_air':reference_air}


class PhysicalMotionHeads(AnalyticHeads):
    def __init__(self,base,geometry,cfg):
        self.__dict__.update(base.__dict__)
        self.geometry=geometry;self.physical_cfg=cfg

    def robot_bodies(self,s):
        out=self.fk(s[...,:NQ],s[...,NQ:]);idx=self.tracked_idx
        return {'pos':out['pos'][...,idx,:],'quat':out['quat'][...,idx,:],
            'lin_vel':out['lin_vel'][...,idx,:],'ang_vel':out['ang_vel'][...,idx,:],
            'root_pos':out['pos'][...,self.root_fk_index,:],'root_quat':out['quat'][...,self.root_fk_index,:],
            'all_pos':out['pos'],'all_quat':out['quat']}

    def reward(self,s_next,action,prev_action,ref,bodies=None):
        b=bodies if bodies is not None else self.robot_bodies(s_next)
        terms=super().reward(s_next,action,prev_action,ref,b)
        physical=self.geometry.components(s_next,ref.body_pos_w,ref.body_quat_w,self.physical_cfg,b['all_pos'],b['all_quat'])
        terms['physical_motion']=self.reward_cfg.dt*physical['total']
        terms['total']=terms['total']+terms['physical_motion']
        return terms


def install_physical_reward(trainer,model,cfg):
    """Install matching terms in this experiment's managers and analytic heads."""
    from copy import deepcopy
    from mjlab.managers import RewardManager,RewardTermCfg
    from supermimic.envs.state_io import StateIO
    if abs(trainer.env.step_dt-trainer.wm.heads.reward_cfg.dt)>1e-8:
        raise ValueError('Real and imagined physical-reward time steps differ')
    geometry=PhysicalGeometry(model,trainer.wm.heads.fk,PROFILE['tracked_bodies'],trainer.device)
    trainer.wm.heads=PhysicalMotionHeads(trainer.wm.heads,geometry,cfg)
    io=StateIO(trainer.env)
    def physical_motion(env):
        command=env.command_manager.get_term('motion')
        reference=command.library.bodies(command.frame)
        return geometry.components(io.read(),reference.pos_w,reference.quat_w,cfg)['total']
    # The environment config is unresolved; do not reuse manager-resolved
    # entity indices when constructing the replacement reward manager.
    reward_terms=deepcopy(trainer.env.cfg.rewards)
    reward_terms['joint_limit'].weight=trainer.wm.heads.reward_cfg.w_joint_limit
    reward_terms['physical_motion']=RewardTermCfg(func=physical_motion,weight=1.)
    trainer.env.reward_manager=RewardManager(reward_terms,trainer.env)
    trainer.env.cfg.rewards=reward_terms
    trainer.max_step_reward+=trainer.env.step_dt*cfg.air_rotation_weight*cfg.maximum_forward_rate
    return geometry
