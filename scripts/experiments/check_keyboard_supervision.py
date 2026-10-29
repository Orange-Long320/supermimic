"""Server regression: exhausted velocity reference must retain tracking supervision."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    import torch
    from dataclasses import is_dataclass
    from keyboard_supervision import install
    install()
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer
    torch.set_num_threads(2)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    cfg=TrainConfig()
    def restore(obj, data):
        for k,v in data.items():
            if not hasattr(obj,k):
                if v is None: continue
                raise ValueError(k)
            if is_dataclass(getattr(obj,k)):restore(getattr(obj,k),v)
            else:setattr(obj,k,tuple(v) if isinstance(getattr(obj,k),tuple) else v)
    restore(cfg,ck['config'])
    cfg.resume=None;cfg.initialize_from=None
    cfg.data.npz_dir=a.dataset;cfg.data.clips=None;cfg.data.limit=None
    cfg.env.num_envs=32;cfg.env.replay_capacity=64
    cfg.policy_trainer.batch_size=32;cfg.policy_trainer.horizon=4
    cfg.policy_trainer.compile=None;cfg.wm_trainer.compile=None
    cfg.policy_trainer.critic_epochs=1;cfg.policy_trainer.critic_minibatches=1
    cfg.run_name=a.out.name;cfg.log_root=a.out.parent
    t=Trainer(cfg)
    try:
        t.policy.load_state_dict(ck['policy']);t.wm.load_state_dict(ck['wm']);t.critic.load_state_dict(ck['critic'])
        obs,_=t.env.reset()
        with torch.no_grad():
            for _ in range(32):obs,*_=t.env.step(t.act(obs))
        # Force the exact failure condition: all velocity replay reference flags
        # are false, as after reference exhaustion. Physical data is untouched.
        t.buffer.command_reference.zero_()
        start=t.buffer.sample_starts(16, t.policy_trainer.cfg.history, env_mask=t.buffer.velocity_env_mask)
        with torch.no_grad():
            _, old_alignment=type(t.policy_trainer).__mro__[1]._imagine_velocity(t.policy_trainer, start)
        assert float(old_alignment)==0.0
        before={k:v.clone() for k,v in t.policy.encoders['velocity'].state_dict().items()}
        result=t.policy_trainer._velocity_update(t.buffer)
        result['original_alignment_with_expired_reference']=float(old_alignment)
        assert result['alignment']>0, result
        assert any(not torch.equal(v,before[k]) for k,v in t.policy.encoders['velocity'].state_dict().items())
        a.out.mkdir(exist_ok=True)
        (a.out/'regression.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    finally:
        t.env.close()
        if t.writer:t.writer.close()


if __name__=='__main__':main()
