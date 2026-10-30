"""Bounded model-based calibration of a new velocity feedback projection.

This deterministic real-data/world-model/BPTT experiment is not physical acceptance.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint', 'dataset', 'reference', 'standing-pose', 'moving-retention', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=128)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--wm-batch', type=int, default=128)
    p.add_argument('--wm-batches', type=int, default=20)
    p.add_argument('--iterations', type=int, default=120)
    p.add_argument('--hours', type=float, default=.5)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--precision-weight', type=float, default=0.)
    a = p.parse_args()
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if Path(sys.prefix).resolve() != Path('/home/2025-Long.xh/HHD/venvs/supermimic').resolve() or not gpu or len(gpu.split(',')) != 1:
        p.error('Use the dedicated server environment and one explicit GPU')
    if not 8 <= a.num_envs <= 1024 or min(a.batch, a.wm_batch, a.wm_batches, a.iterations) < 1 or a.iterations > 120 or not 0 < a.hours <= .5 or (a.lr is not None and a.lr <= 0):
        p.error('Invalid bounded training parameters')
    if a.out.exists():
        p.error('Output directory already exists')
    for path in (a.checkpoint, a.reference, a.standing_pose, a.moving_retention, a.dataset / 'walk.npz'):
        if not path.is_file():
            p.error(f'Missing input: {path}')
    import numpy as np
    import torch
    if not np.isfinite(a.precision_weight) or a.precision_weight < 0:
        p.error('Precision weight must be finite and nonnegative')
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    from supermimic.kinematics.rotations import heading_yaw
    from supermimic.locomotion.config import VelocityCfg
    from supermimic.locomotion.task import measured_velocity
    from supermimic.policy.standing import StandingContext
    from supermimic.world_model.state_spec import StateSpecCfg
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg
    from supermimic_velocity_imagination import VelocityFeedbackImaginationTrainer

    torch.set_num_threads(2)
    torch.manual_seed(29)
    policy, saved = load_policy_from_checkpoint(a.checkpoint, 'cuda:0')
    source = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if not hasattr(policy, 'standing_skill') or policy.standing_skill.mode != 'expert' or policy.standing_skill.feedback_dim != 6:
        raise ValueError('Six-input standing expert required')
    if source.get('standing_reference', {}).get('sha256') != sha(a.reference):
        raise ValueError('Standing reference does not match checkpoint')
    retention = torch.load(a.moving_retention, map_location='cpu', weights_only=False)
    for key, expected in (('source_sha256', sha(a.checkpoint)), ('reference_sha256', sha(a.reference)),
                          ('standing_pose_sha256', sha(a.standing_pose))):
        if retention.get(key) != expected:
            raise ValueError(f'Moving retention {key} does not match initialization')
    for key in ('obs', 'command', 'feedback', 'source_action'):
        if key not in retention:
            raise ValueError(f'Moving retention missing {key}')
    if hasattr(policy, 'velocity_feedback_weight'):
        delattr(policy, 'velocity_feedback_weight')
    policy.requires_grad_(False)
    frozen = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
    lr = float(saved['policy_trainer']['lr']) if a.lr is None else a.lr
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError('Invalid actor learning rate')
    wc = copy.deepcopy(saved['world_model'])
    wc['state_spec'] = StateSpecCfg(**wc['state_spec'])
    wm = WorldModel(WorldModelCfg(**wc), None, 'cuda:0')
    missing, extra = wm.load_state_dict(source['wm'], strict=False)
    if missing or any(not k.startswith('heads.') for k in extra):
        raise ValueError(f'Unexpected world model keys: {missing}, {extra}')
    wm_opt = torch.optim.Adam(wm.dynamics.parameters(), lr=saved['wm_trainer']['lr'])
    env = make_env(SuperMimicEnvCfgArgs(npz_files=[str(a.dataset / 'walk.npz')], num_envs=a.num_envs,
        seed=29, future_frames=saved['env']['future_frames'], episode_length_s=66., action_noise_std=0.,
        obs_noise=False, reset_noise=False, domain_randomization=False, record_transitions=True, replay_capacity=256,
        anchor_obs=saved['env']['anchor_obs'], velocity=VelocityCfg(**saved['velocity'])), device='cuda:0')
    torch.backends.cuda.matmul.fp32_precision = 'ieee'
    torch.backends.cudnn.fp32_precision = 'ieee'
    env.cfg.auto_reset = False
    cmd = env.command_manager.get_term('motion')
    cmd.is_velocity[:] = True; cmd.force_standing_reset = True; cmd.set_manual_command((0., 0., 0.))
    obs, _ = env.reset()
    context = StandingContext(cmd, env)
    replay = env.recorder_manager.get_term('replay').buffer
    ref = np.load(a.reference)
    trainer = VelocityFeedbackImaginationTrainer(policy, wm, cmd.library.default_joint_pos,
        env.scene['robot'].data.soft_joint_pos_limits[0], float(ref['body_pos_w'][0, 0, 2]),
        VelocityCfg(**saved['velocity']), saved['policy']['history'], env.step_dt, lr=lr, horizon=32, precision_weight=a.precision_weight)
    trainer.set_retention(*(retention[k].to('cuda:0') for k in ('obs', 'command', 'feedback', 'source_action')))
    actual_trainable = {name for name, param in policy.named_parameters() if param.requires_grad}
    if actual_trainable != {'velocity_feedback_weight'}:
        raise AssertionError(f'Unexpected trainable parameters: {actual_trainable}')
    E, T = replay.E, replay.T
    ids = torch.arange(E, device='cuda')
    train_env = ids % 4 != 3
    cold = (ids // 4) % 4 == 0
    choices = torch.tensor([[.3,0.,0.],[-.2,0.,0.],[0.,0.,.4],[0.,0.,-.4],[.6,0.,0.]], device='cuda')
    test_commands = torch.cat((torch.tensor([[0.,0.,0.],[.05,0.,0.]], device='cuda'), choices))
    offset = (ids // 2) % 5
    rates = torch.tensor([1.,1.,1.5], device='cuda') * env.step_dt
    goal = cmd.velocity_command.clone()
    meta_command = torch.zeros(E, T, 3, device='cuda')
    history = saved['policy']['history']
    trace = []; actor_updates = falls = timeouts = 0
    started = time.monotonic()
    a.out.mkdir(parents=True, exist_ok=False)

    def verify_frozen():
        now = policy.state_dict()
        for k, v in frozen.items():
            if not torch.equal(now[k].detach().cpu(), v):
                raise AssertionError(f'Frozen tensor changed: {k}')

    @torch.no_grad()
    def check_initial():
        if torch.count_nonzero(policy.velocity_feedback_weight):
            raise AssertionError('New velocity feedback projection is not zero')
        x = obs['policy'][:min(E, 16)]
        feedback = context.features(torch.zeros(len(obs['policy']), 3, device='cuda'))[:len(x)]
        for command in test_commands:
            c = command.expand(len(x), -1)
            before = policy.velocity_action(x, c, feedback, disable_moving_feedback=True)
            after = policy.velocity_action(x, c, feedback)
            if not torch.equal(before, after):
                raise AssertionError('Zero projection changed action')
        return True
    check_initial()
    # Verify the zero-initialized checkpoint can be reconstructed before any update.
    initial_ck = copy.deepcopy(source)
    initial_ck['policy'] = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
    initial_ck['standing_skill'] = dict(initial_ck['standing_skill'], moving_feedback=True)
    initial_path = a.out / 'initial-roundtrip.pt'
    torch.save(initial_ck, initial_path)
    initial_loaded, _ = load_policy_from_checkpoint(initial_path, 'cuda:0')
    with torch.no_grad():
        x = obs['policy'][:min(E, 16)]
        ff = context.features(torch.zeros(E, 3, device='cuda'))[:len(x)]
        for command in test_commands:
            c = command.expand(len(x), -1)
            if not torch.equal(policy.velocity_action(x, c, ff), initial_loaded.velocity_action(x, c, ff)):
                raise AssertionError('Initial checkpoint roundtrip changed actions')
    del initial_loaded, initial_ck
    initial_path.unlink()
    runtime_config = dict(algorithm='supermimic_bptt', actor_lr=lr, precision_weight=a.precision_weight,
        actor_lr_source='checkpoint.config.policy_trainer.lr' if a.lr is None else 'explicit_override',
        imagination_horizon=32, history=history, dt=env.step_dt, wm_window=32,
        wm_batches_per_update=a.wm_batches, wm_batch=a.wm_batch, actor_batch=a.batch,
        wm_lr=saved['wm_trainer']['lr'], wm_grad_clip=saved['wm_trainer']['grad_clip'],
        model_gate_velocity_1=[.05, .05, .15], model_gate_velocity_8=[.10, .10, .15],
        model_gate_velocity_32=[.20, .20, .30],
        model_gate_position_heading_32=[.03, .08], collection_steps_per_iteration=32,
        episode_seconds=66., replay_capacity=T, num_envs=E, obs_noise=False, reset_noise=False,
        action_noise_std=0., domain_randomization=False, exploration_noise=0.,
        moving_retention_samples=len(retention['obs']), moving_retention_sha256=sha(a.moving_retention),
        trainable_policy_parameters=sorted(actual_trainable), control_matmul_precision='ieee',
        source_commit=os.environ.get('SM_SOURCE_COMMIT', 'unspecified'),
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()})
    (a.out / 'runtime_config.json').write_text(json.dumps(runtime_config, indent=2) + '\n')
    (a.out / 'launch.json').write_text(json.dumps(dict(argv=sys.argv, gpu=gpu, interpreter=sys.executable,
        python=sys.version, torch=torch.__version__, cuda_runtime=torch.version.cuda, cuda_device=torch.cuda.get_device_name(0),
        source_sha256=sha(a.checkpoint), reference_sha256=sha(a.reference), standing_pose_sha256=sha(a.standing_pose),
        walk_sha256=sha(a.dataset / 'walk.npz'), retention_sha256=sha(a.moving_retention), entry_sha256=sha(__file__),
        helper_sha256=sha(Path(__file__).with_name('supermimic_velocity_imagination.py')),
        source_commit=os.environ.get('SM_SOURCE_COMMIT', 'unspecified'), actor_lr=lr, precision_weight=a.precision_weight,
        lr_source='checkpoint.config.policy_trainer.lr' if a.lr is None else 'explicit_override',
        zero_projection_equivalent=True, initial_checkpoint_roundtrip=True, frozen_policy_tensors=sorted(frozen), control_matmul_precision='ieee',
        scope='deterministic moving steady-state BPTT; no near-zero counterfactual training or physical acceptance'), indent=2)+'\n')

    def windows(mask, n):
        e,t = replay._sample_from_mask(replay.window_mask(32) & mask[:,None], n)
        tt = (t[None] + torch.arange(33,device='cuda')[:,None]) % T
        ee = e[None].expand(33,-1)
        return replay.s[ee,tt], replay.a_exec[ee[:-1],tt[:-1]], replay.xi[e,t]

    def eligible():
        t = torch.arange(T, device='cuda')[None,:]
        return (replay.valid & ~replay.terminal & train_env[:,None]
                & (((replay.ptr[:,None]-t-1)%T) < T-history-1)
                & (meta_command.square().sum(-1) >= .01))

    def actor_batch():
        e,t = replay._sample_from_mask(eligible(), a.batch)
        tt = (t[None] + torch.arange(history,device='cuda')[:,None]-(history-1)) % T
        ee = e[None].expand(history,-1)
        same = (replay.ep_id[ee,tt] == replay.ep_id[e,t][None]) & replay.valid[ee,tt]
        first = same.float().argmax(0)
        tt = torch.where(same,tt,tt[first,torch.arange(a.batch,device='cuda')][None].expand_as(tt))
        prev = (tt-1)%T
        valid = (replay.ep_id[ee,prev] == replay.ep_id[e,t][None]) & replay.valid[ee,prev] & ~replay.ep_start[ee,tt]
        actions = torch.where(valid[...,None],replay.a[ee,prev],torch.zeros_like(replay.a[ee,prev]))
        return dict(hist_states=replay.s[ee,tt], hist_actions=actions, xi=replay.xi[e,t], command=meta_command[e,t])

    @torch.no_grad()
    def heldout():
        states, actions, xi = windows(~train_env, a.wm_batch)
        pred = states[0]; velocity = {}; pose = {}
        for j in range(32):
            pred = wm.step(pred,actions[j],xi)
            if j+1 in (1,8,32):
                velocity[str(j+1)] = (measured_velocity(pred)-measured_velocity(states[j+1])).square().mean(0).sqrt().tolist()
                dh = heading_yaw(pred[:,3:7])-heading_yaw(states[j+1,:,3:7])
                dh = torch.atan2(torch.sin(dh),torch.cos(dh))
                pose[str(j+1)] = [float((pred[:,:2]-states[j+1,:,:2]).square().sum(-1).mean().sqrt()), float(dh.square().mean().sqrt())]
        return velocity,pose

    def save(iteration):
        verify_frozen()
        ck = copy.deepcopy(source)
        ck['policy'] = {k:v.detach().cpu() for k,v in policy.state_dict().items()}
        ck['standing_skill'] = dict(ck['standing_skill'], moving_feedback=True)
        ck['wm'].update({k:v.detach().cpu() for k,v in wm.state_dict().items()})
        for key in ('standing_ppo_state','transition_ppo_state','physical_stop','control_feedback','standing_feedback',
                    'transition_standing','supermimic_stop','wm_opt','policy_opt','critic_opt','velocity_training'):
            ck.pop(key,None)
        ck['supermimic_velocity_feedback'] = dict(algorithm='supermimic_bptt',iteration=iteration,
            source_sha256=sha(a.checkpoint), reference_sha256=sha(a.reference), standing_pose_sha256=sha(a.standing_pose),
            moving_retention_sha256=sha(a.moving_retention), entry_sha256=sha(__file__), actor_updates=actor_updates,
            actor_state=trainer.save_state(), wm_optimizer=wm_opt.state_dict(), trace=trace, actor_lr=lr,
            config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
            validation='model gate only; physical acceptance pending', scope='moving steady only; no near-zero training',
            control_matmul_precision='ieee', source_commit=os.environ.get('SM_SOURCE_COMMIT', 'unspecified'),
            runtime_config=runtime_config)
        path = a.out / f'candidate_{iteration}.pt'
        torch.save(ck,path)
        reloaded, _ = load_policy_from_checkpoint(path,'cuda:0')
        with torch.no_grad():
            x=retention['obs'].to('cuda:0'); c=retention['command'].to('cuda:0'); f=retention['feedback'].to('cuda:0')
            if not torch.equal(policy.velocity_action(x,c,f),reloaded.velocity_action(x,c,f)):
                raise AssertionError('Checkpoint roundtrip changed actions')
        del reloaded
        torch.save(dict(iteration=iteration, wm=ck['wm'], optimizer=wm_opt.state_dict(),
                        source_sha256=sha(a.checkpoint)), a.out / f'wm_candidate_{iteration}.pt')
        path.with_suffix('.ready').write_text(str(iteration))

    try:
        last_saved = 0
        for iteration in range(1,a.iterations+1):
            with torch.no_grad():
                for _ in range(32):
                    age=env.episode_length_buf*env.step_dt
                    desired=torch.zeros_like(goal)
                    for j,start in enumerate((3.,15.,27.,39.,51.)):
                        active=~cold & (age>=start) & (age<start+7.)
                        desired[active]=choices[(offset[active]+j)%5]
                    goal=goal+(desired-goal).clamp(-rates,rates)
                    cmd.manual_command=goal; cmd._refresh_velocity()
                    feedback=context.features(goal)
                    action=policy.velocity_action(obs['policy'],goal,feedback)
                    ptr=replay.ptr.clone(); before=cmd.state_io.read().clone() if iteration==1 else None
                    meta_command[ids,ptr]=goal
                    obs,_,term,tout,_=env.step(action)
                    term,tout=term.clone(),tout.clone(); done=term|tout
                    falls+=int(term.sum()); timeouts+=int(tout.sum())
                    if before is not None and (~done).any():
                        live=(~done).nonzero().squeeze(-1)
                        if not torch.equal(replay.s[live,ptr[live]],before[live]):
                            raise AssertionError('Replay metadata and state misaligned')
                    if done.any():
                        reset_ids=done.nonzero().squeeze(-1); goal[reset_ids]=0
                        obs,_=env.reset(env_ids=reset_ids); context.reset(reset_ids)
            row=dict(iteration=iteration,seconds=time.monotonic()-started,replay_rows=replay.size,
                     total_falls=falls,total_timeouts=timeouts,model_gate=False,actor_updated=False)
            mask=replay.window_mask(32)
            if iteration>=2 and (mask&train_env[:,None]).any() and (mask&~train_env[:,None]).any():
                wm.requires_grad_(True)
                for _ in range(a.wm_batches):
                    states,actions,xi=windows(train_env,a.wm_batch); pred=states[0]; loss=0.
                    for j in range(32):
                        pred=wm.step(pred,actions[j],xi)
                        loss=loss+wm.spec.error_vector(pred,states[j+1]).square().sum(-1).mean()/32
                    wm_opt.zero_grad(set_to_none=True); loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(wm.dynamics.parameters(),saved['wm_trainer']['grad_clip'])
                    if not torch.isfinite(loss) or not torch.isfinite(norm):
                        raise FloatingPointError('Nonfinite world model fit')
                    wm_opt.step()
                row['wm_loss']=float(loss.detach()); vm,pm=heldout()
                row['heldout_velocity_rmse']=vm; row['heldout_pose_rmse']=pm
                row['model_gate']=(all(x<y for x,y in zip(vm['1'],(.05,.05,.15))) and
                    all(x<y for x,y in zip(vm['8'],(.10,.10,.15))) and
                    all(x<y for x,y in zip(vm['32'],(.20,.20,.30))) and all(x<y for x,y in zip(pm['32'],(.03,.08))))
                row['actor_sampleable']=int(eligible().sum())
                if row['model_gate'] and row['actor_sampleable']:
                    wm.requires_grad_(False)
                    row['actor_result']=trainer.update(actor_batch())
                    row['actor_updated']=bool(row['actor_result']['step_accepted'] and row['actor_result']['projection_change_l2'] > 0)
                    actor_updates+=int(row['actor_updated'])
            row['actor_updates']=actor_updates; row['cuda_peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
            trace.append(row); print(json.dumps(row),flush=True)
            if iteration%20==0:
                save(iteration); last_saved=iteration
            if time.monotonic()-started>=a.hours*3600:
                break
        if iteration!=last_saved:
            save(iteration)
        (a.out/'report.json').write_text(json.dumps(dict(status='completed_candidate',iterations=iteration,
            actor_updates=actor_updates,trace=trace,scope='model gate only; physical validation pending'),indent=2)+'\n')
    finally:
        env.close()


if __name__ == '__main__':
    main()
