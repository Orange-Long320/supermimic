"""Server-only diagnostic of 32-step world-model action gradients against MuJoCo.

No actor updates. Bounded open-loop action perturbations are a diagnostic, not
a deployed policy. Every physical test starts in a fresh simulator.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'actor', 'dataset', 'out'):
        p.add_argument('--'+key, required=True, type=Path)
    p.add_argument('--models', nargs='+', type=Path, required=True)
    p.add_argument('--warmup', type=int, default=150)
    p.add_argument('--epsilon', type=float, default=.02)
    a = p.parse_args()
    if Path(sys.prefix).resolve() != Path('/home/2025-Long.xh/HHD/venvs/supermimic').resolve():
        raise ValueError('Dedicated server interpreter required')
    if len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) != 1 or not 0 < a.epsilon <= .03:
        raise ValueError('One explicit GPU and bounded perturbation required')
    a.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.source/'src'))
    import torch
    from mjlab.utils.torch import configure_torch_backends
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.sim_context import SimContext
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    from supermimic.locomotion.config import VelocityCfg
    from supermimic.locomotion.task import measured_velocity
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg
    from supermimic.world_model.state_spec import StateSpecCfg
    torch.set_num_threads(2)
    configure_torch_backends()
    actor, saved = load_policy_from_checkpoint(a.actor, 'cuda:0')
    actor.requires_grad_(False)
    commands = torch.tensor([[0,0,0],[.3,0,0],[-.2,0,0],[0,0,.4],[0,0,-.4],[.6,0,0]], device='cuda')
    scale = torch.tensor([.5,.5,.7], device='cuda')

    def error(states):
        return ((measured_velocity(states)-commands)/scale).square().sum(-1).mean(0)

    @torch.no_grad()
    def physical(plan=None, start_state=None):
        torch.manual_seed(19)
        env = make_env(SuperMimicEnvCfgArgs(npz_files=[str(a.dataset/'walk.npz')],
            num_envs=6, seed=19, future_frames=saved['env']['future_frames'], episode_length_s=30,
            action_noise_std=0, obs_noise=False, reset_noise=False, domain_randomization=False,
            record_transitions=False, anchor_obs=saved['env']['anchor_obs'],
            velocity=VelocityCfg(**saved['velocity'])), device='cuda:0')
        try:
            env.cfg.auto_reset = False
            cmd = env.command_manager.get_term('motion')
            cmd.is_velocity[:] = True
            cmd.force_standing_reset = True
            cmd.forced_frame[:] = cmd.library.clip_start[0]
            obs, _ = env.reset()
            def set_commands():
                cmd.reference_mode[:] = False
                cmd.fallback_command[:] = commands
                cmd.command_age[:] = 0
                cmd._refresh_velocity()
            for _ in range(a.warmup):
                set_commands()
                act = actor.velocity_action(obs['policy'], commands)
                obs,_,term,tout,_ = env.step(act)
                if (term|tout).any():
                    raise RuntimeError('Warmup failed')
            initial = cmd.state_io.read().clone()
            # Replaying warmup is not guaranteed to reproduce a contact trajectory.
            # Match qpos/qvel explicitly, and zero the solver warmstart in BOTH arms.
            if hasattr(env.sim.data, 'act') and env.sim.data.act[:].numel():
                raise RuntimeError('Actuator internal state needs an explicit snapshot')
            cmd.state_io.write(torch.arange(6,device='cuda'), initial if start_state is None else start_state)
            initial = cmd.state_io.read().clone()
            xi = SimContext(env, '.*(ankle_roll|foot).*', 'torso_link').read().clone()
            states, actions, failures = [], [], []
            for t in range(32):
                set_commands()
                act = actor.velocity_action(obs['policy'], commands) if plan is None else plan[t]
                obs,_,term,tout,_ = env.step(act)
                states.append(cmd.state_io.read().clone())
                actions.append(act.clone())
                failures.append((term|tout).clone())
            return initial, xi, torch.stack(states), torch.stack(actions), bool(torch.stack(failures).any())
        finally:
            env.close()

    initial, xi, actual, actions, failed = physical()
    baseline = error(actual)
    replay_start, replay_xi, replay_states, _, replay_failed = physical(actions,initial)
    replay_delta=float((error(replay_states)-baseline).abs().max())
    if replay_delta>1e-3 or replay_failed!=failed:
        raise RuntimeError(f'Unperturbed physical replay control failed: {replay_delta}')
    report = dict(actor=str(a.actor), actor_sha256=digest(a.actor), argv=sys.argv,
                  gpu=os.environ['CUDA_VISIBLE_DEVICES'], pid=os.getpid(), commands=commands.tolist(),
                  baseline_error=baseline.tolist(), baseline_failure=failed,
                  unperturbed_replay_error_max_delta=replay_delta,
                  scope='Six nominal command cases, deterministic fresh simulators. No policy update. Compare a bounded 32-action plan against the same frozen actor actions; not a full continuous-control acceptance test.', models=[])
    torch.save(dict(initial=initial,xi=xi,states=actual,actions=actions),a.out/'baseline.pt')
    for path in a.models:
        checkpoint = torch.load(path,map_location='cuda:0',weights_only=False)
        config = copy.deepcopy(saved['world_model'])
        config['state_spec'] = StateSpecCfg(**config['state_spec'])
        wm = WorldModel(WorldModelCfg(**config),None,'cuda:0')
        missing, extra = wm.load_state_dict(checkpoint['wm'],strict=False)
        assert not missing and all(k.startswith('heads.') for k in extra)
        wm.requires_grad_(False)
        def predict(plan):
            state=initial
            states=[]
            for t in range(32):
                state=wm.step(state,plan[t],xi)
                states.append(state)
            return torch.stack(states)
        with torch.no_grad():
            predicted_baseline = error(predict(actions))
        plan = torch.nn.Parameter(actions.clone())
        opt = torch.optim.Adam([plan],lr=.005)
        for _ in range(20):
            objective=error(predict(plan)).mean()
            opt.zero_grad(set_to_none=True)
            objective.backward()
            if not torch.isfinite(plan.grad).all():raise FloatingPointError('Invalid plan gradient')
            opt.step()
            with torch.no_grad():
                plan.copy_(torch.maximum(torch.minimum(plan,actions+a.epsilon),actions-a.epsilon))
        with torch.no_grad():
            predicted_final = error(predict(plan))
        start2,xi2,states2,_,failed2=physical(plan.detach(),initial)
        state_delta=float((start2-initial).abs().max())
        context_delta=float((xi2-xi).abs().max())
        if state_delta>1e-3 or context_delta>1e-6:
            raise RuntimeError(f'Unmatched physical restart: {state_delta}, {context_delta}')
        actual_error=error(states2)
        row=dict(model=str(path),sha256=digest(path),restart_state_max_delta=state_delta,
                 restart_context_max_delta=context_delta,failure=failed2,
                 model_error_before=predicted_baseline.tolist(),model_error_after=predicted_final.tolist(),
                 physical_error_before=baseline.tolist(),physical_error_after=actual_error.tolist(),
                 model_improvement=(predicted_baseline-predicted_final).tolist(),
                 physical_improvement=(baseline-actual_error).tolist(),
                 max_action_change=float((plan.detach()-actions).abs().max()))
        torch.save(dict(actions=plan.detach(),physical_states=states2,initial=start2,xi=xi2),
                   a.out/('plan-'+row['sha256'][:12]+'.pt'))
        report['models'].append(row)
        (a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(row),flush=True)
    report['status']='complete'
    (a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
