"""Compare frozen learned dynamics with matched MuJoCo transitions (server only)."""
import argparse, json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'dataset', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    import torch
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.sim_context import SimContext
    from supermimic.locomotion.config import VelocityCfg
    from supermimic.locomotion.task import measured_velocity
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg
    from supermimic.world_model.state_spec import StateSpecCfg
    torch.set_num_threads(2)
    policy, saved = load_policy_from_checkpoint(a.checkpoint, 'cuda:0')
    ck = torch.load(a.checkpoint, map_location='cuda:0', weights_only=False)
    cfg = dict(saved['world_model'])
    cfg['state_spec'] = StateSpecCfg(**cfg['state_spec'])
    wm = WorldModel(WorldModelCfg(**cfg), None, 'cuda:0').eval()
    missing, unexpected = wm.load_state_dict(ck['wm'], strict=False)
    assert not missing and all(k.startswith('heads.') for k in unexpected), (missing, unexpected)
    env = make_env(SuperMimicEnvCfgArgs(npz_files=[str(a.dataset/'walk.npz')], num_envs=8,
        seed=0, future_frames=saved['env']['future_frames'], episode_length_s=72,
        action_noise_std=0, obs_noise=False, reset_noise=False, domain_randomization=False,
        record_transitions=False, anchor_obs=saved['env']['anchor_obs'], velocity=VelocityCfg(**saved['velocity'])), device='cuda:0')
    env.cfg.auto_reset = False
    cmd = env.command_manager.get_term('motion')
    cmd.is_velocity[:] = True
    cmd.force_standing_reset = True
    context = SimContext(env, '.*(ankle_roll|foot).*', 'torso_link')
    schedule = [(0,(0,0,0)), (3,(.3,0,0)), (10,(0,0,0)), (15,(-.2,0,0)),
        (22,(0,0,0)), (27,(0,0,.4)), (34,(0,0,0)), (39,(0,0,-.4)),
        (46,(0,0,0)), (51,(.6,0,0)), (58,(0,0,0))]
    states, actions, phases, dones = [], [], [], []
    try:
        cmd.set_manual_command((0,0,0))
        obs,_ = env.reset()
        xi = context.read().clone()
        with torch.no_grad():
            states.append(cmd.state_io.read())
            for t in range(3300):
                phase, target = next(x for x in reversed(schedule) if t*.02 >= x[0])
                cmd.set_manual_command(target)
                action = policy.velocity_action(obs['policy'], cmd.velocity_command)
                obs,_,term,tout,_ = env.step(action)
                states.append(cmd.state_io.read()); actions.append(action); phases.append(phase)
                dones.append(term|tout)
                if (term|tout).any():
                    break
            s = torch.stack(states); act = torch.stack(actions)
            rows = []
            for phase, target in schedule:
                starts = [i for i in range(0,len(actions)-32,8) if phases[i] == phase and phases[i+31] == phase and i*.02 >= phase+1]
                if not starts:
                    continue
                ids = torch.tensor(starts,device='cuda'); pred=s[ids].reshape(-1,s.shape[-1])
                ctx=xi[None].expand(len(ids),-1,-1).reshape(len(ids)*8,-1)
                row = dict(phase_s=phase,command=target,windows=len(ids))
                for h in range(1,33):
                    pred=wm.step(pred,act[ids+h-1].reshape(len(ids)*8,-1),ctx)
                    if h in (1,8,32):
                        pv=measured_velocity(pred); av=measured_velocity(s[ids+h].reshape_as(pred))
                        row[str(h)]=dict(actual=av.mean(0).tolist(),model=pv.mean(0).tolist(),
                            rmse=(pv-av).square().mean(0).sqrt().tolist())
                rows.append(row)
        result=dict(checkpoint=str(a.checkpoint),steps=len(actions),any_failure=bool(torch.stack(dones).any()),
            scope='Identical initial states, context, and executed action sequence; no optimizer updates.',phases=rows)
        a.out.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)
    finally:
        env.close()

if __name__ == '__main__':
    main()
