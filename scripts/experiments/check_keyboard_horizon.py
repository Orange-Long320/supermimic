"""Exercise bounded actor and critic updates on actual server replay."""
import argparse,json
from dataclasses import is_dataclass
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','dataset','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--control-prior',type=Path)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    import torch
    from keyboard_supervision import install,install_quiet_stop_cost,install_velocity_horizon
    install();install_quiet_stop_cost();install_velocity_horizon(8)
    if a.control_prior:
        from keyboard_supervision import install_control_prior
        install_control_prior(a.control_prior)
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer
    torch.set_num_threads(2)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    cfg=TrainConfig()
    def restore(obj,d):
        for k,v in d.items():
            if not hasattr(obj,k):
                if v is None:continue
                raise ValueError(k)
            if is_dataclass(getattr(obj,k)):restore(getattr(obj,k),v)
            else:setattr(obj,k,tuple(v) if isinstance(getattr(obj,k),tuple) else v)
    restore(cfg,ck['config'])
    cfg.resume=cfg.initialize_from=None
    cfg.data.npz_dir=a.dataset;cfg.data.clips=None;cfg.data.limit=None
    cfg.env.num_envs=32;cfg.env.replay_capacity=64
    cfg.policy_trainer.batch_size=32;cfg.policy_trainer.horizon=32
    cfg.policy_trainer.compile=cfg.wm_trainer.compile=None
    cfg.policy_trainer.critic_epochs=cfg.policy_trainer.critic_minibatches=1
    cfg.run_name=a.out.name;cfg.log_root=a.out.parent
    t=Trainer(cfg)
    try:
        for key,module in [('policy',t.policy),('wm',t.wm),('critic',t.critic)]:module.load_state_dict(ck[key])
        obs,_=t.env.reset()
        with torch.no_grad():
            for _ in range(40):obs,*_=t.env.step(t.act(obs))
        trainer=t.policy_trainer;original_cfg=trainer.cfg;original=trainer._imagine_velocity
        captured={}
        def capture(start):
            ro,align=original(start)
            captured.update(velocity_steps=len(ro.actions),state_count=len(ro.states),horizon_during_update=trainer.cfg.horizon)
            return ro,align
        trainer._imagine_velocity=capture
        before={k:v.clone() for k,v in t.policy.encoders['velocity'].state_dict().items()}
        metrics=trainer._velocity_update(t.buffer)
        if a.control_prior:
            teacher_before={k:v.clone() for k,v in trainer.control_prior.state_dict().items()}
            metrics=trainer._velocity_update(t.buffer)
            assert metrics['control_prior_action_mse']>0,metrics
            assert all(torch.equal(v,teacher_before[k]) for k,v in trainer.control_prior.state_dict().items())
            assert trainer.checkpoint_state()['control_prior']['weight']==10.
        assert captured==dict(velocity_steps=8,state_count=9,horizon_during_update=8),captured
        assert trainer.cfg is original_cfg and trainer.cfg.horizon==32
        assert trainer.checkpoint_state()['imagination_horizon']==8
        assert metrics['alignment']>0 and metrics['critic_loss']>=0
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
        assert any(not torch.equal(v,before[k]) for k,v in t.policy.encoders['velocity'].state_dict().items())
        result=dict(passed=True,**captured,motion_horizon_after_update=trainer.cfg.horizon,metrics=metrics)
        (a.out/'regression.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    finally:
        t.env.close()
        if t.writer:t.writer.close()

if __name__=='__main__':main()
