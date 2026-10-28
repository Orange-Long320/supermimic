"""Compare the native frozen-controller input against the actual mjlab reset.

This is an observation/action interface check, not climbing execution. The
environment is created on CPU, receives no policy updates, and is closed.
"""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('New output required')
    from supermimic.robots.paths import select_robot
    select_robot('x2_ultra',check_assets=True)
    import mujoco
    import numpy as np
    import torch
    from supermimic.envs import SuperMimicEnvCfgArgs,make_env
    from supermimic.envs.state_io import StateIO
    from supermimic.robots.dimensions import PROFILE,XI_DIM
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    from supermimic.world_model.heads import analytic_observation,flatten_history_term_major,anchor_relative_obs
    torch.set_num_threads(1);policy,cfg=load_policy_from_checkpoint(a.checkpoint,'cpu')
    env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(a.motion)],num_envs=1,
        action_noise_std=0.,obs_noise=False,reset_noise=False,domain_randomization=False,
        sampling_mode='start',record_transitions=False,anchor_obs=cfg['env'].get('anchor_obs',False)),device='cpu')
    try:
        cmd=env.command_manager.get_term('motion');cmd.forced_frame[:]=cmd.library.clip_start[0]
        obs,_=env.reset();state=StateIO(env).read()
        q,v=cmd.library.ref_qpos_qvel(torch.tensor([0]));expected=torch.cat([q,v],dim=-1)
        m=mujoco.MjModel.from_binary_path(str(a.model));d=mujoco.MjData(m);d.qpos[:]=q[0].numpy();d.qvel[:]=v[0].numpy();mujoco.mj_forward(m,d)
        defaults=torch.tensor([PROFILE['default_joint_pos'][n] for n in PROFILE['joint_names']],dtype=torch.float32)
        clean=analytic_observation(expected,torch.zeros((1,31)),torch.zeros((1,XI_DIM)),defaults)
        native_obs=flatten_history_term_major(torch.stack([clean]*policy.cfg.history))
        # The existing command manager anticipates the first control tick.
        goal=cmd.library.goal_window(torch.tensor([1]),int(cfg['env'].get('future_frames',10)))
        if cfg['env'].get('anchor_obs',False):
            anchor=PROFILE['anchor_body'];body=m.body('robot/'+anchor).id;idx=cmd.library.body_names.index(anchor)
            rel=anchor_relative_obs(torch.tensor(d.xpos[body][None],dtype=torch.float32),torch.tensor(d.xquat[body][None],dtype=torch.float32),
                cmd.library.body_pos_w[1:2,idx],cmd.library.body_quat_w[1:2,idx])
            goal=torch.cat([goal,rel],dim=-1)
        with torch.no_grad():ours=policy(native_obs,goal);theirs=policy(obs['policy'],obs['goal'])
        env.action_manager.process_action(theirs)
        actual_target=env.action_manager.get_term('joint_pos').processed_actions
        scale=torch.tensor([PROFILE['action_scale'][n] for n in PROFILE['joint_names']],dtype=torch.float32)
        target=defaults+theirs[0]*scale
        report={'initial_state_max_abs_error':float(abs(state-expected).max()),'actual_initial_reference_frame':int(cmd.frame[0]),
            'policy_observation_max_abs_error':float(abs(native_obs-obs['policy']).max()),
            'goal_max_abs_error':float(abs(goal-obs['goal']).max()),'action_max_abs_error':float(abs(ours-theirs).max()),
            'servo_target_max_abs_error':float(abs(actual_target[0]-target).max()),'cpu_only':True,'training':False,'complete_climb_verified':False}
        passed=(report['actual_initial_reference_frame']==1 and report['initial_state_max_abs_error']<=2e-5
            and report['policy_observation_max_abs_error']<=2e-5 and report['goal_max_abs_error']<=2e-5
            and report['action_max_abs_error']<=1e-4 and report['servo_target_max_abs_error']<=1e-6)
        report['interface_passed']=passed;a.out.mkdir(parents=True);(a.out/'as-run.py').write_bytes(Path(__file__).read_bytes())
        (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
        if not passed:raise RuntimeError('Native frozen-policy interface does not match the existing environment')
    finally:env.close()


if __name__=='__main__':main()
