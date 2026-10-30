"""Experimental reference-scheduled keyboard viewer; no physical acceptance claim."""
import argparse
from pathlib import Path
import threading
import time

from play_keyboard_policy import COMMANDS, start_controls


class Mailbox:
    def __init__(self):
        self.lock = threading.Lock()
        self.command = (0., 0., 0.)
        self.timestamp = 0.

    def put(self, command):
        with self.lock:
            self.command = tuple(command)
            self.timestamp = time.monotonic()

    def mode(self):
        with self.lock:
            command = self.command if time.monotonic()-self.timestamp < .5 else COMMANDS['stand']
        lookup = {COMMANDS['stand']: 'stand', COMMANDS['forward']: 'walk', COMMANDS['backward']: 'backward',
                  COMMANDS['left']: 'turn_left_inplace', COMMANDS['right']: 'turn_right_inplace', COMMANDS['jog']: 'jog'}
        return lookup[command]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--viewer-port', type=int, default=18080)
    p.add_argument('--keyboard-port', type=int, default=18081)
    p.add_argument('--loop-profile', choices=['baseline', 'sustained'], default='baseline')
    p.add_argument('--straight-heading', action='store_true')
    a = p.parse_args()
    import os
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(a.gpu):
        raise ValueError('Select a single physical GPU explicitly')
    import numpy as np
    import torch
    import viser
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.viewer import ViserPlayViewer
    from supermimic.control.skill_tracker import SkillTracker
    from mjlab.envs import ManagerBasedRlEnv
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    torch.set_num_threads(2)
    policy, cfg = load_policy_from_checkpoint(a.checkpoint, 'cuda:0')
    stand = a.dataset / 'walk_stop.npz'
    with np.load(stand) as z:
        initial = len(z['joint_pos'])-1
    env_cfg = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=[str(stand)], num_envs=1, seed=0,
        future_frames=cfg['env']['future_frames'], episode_length_s=1000., action_noise_std=0., obs_noise=False,
        reset_noise=False, domain_randomization=False, record_transitions=False, sampling_mode='start', anchor_obs=True))
    env_cfg.auto_reset = False
    del env_cfg.terminations['clip_end']
    env = ManagerBasedRlEnv(env_cfg, device='cuda:0')
    cmd = env.command_manager.get_term('motion')
    cmd.forced_frame[:] = initial
    mailbox = Mailbox()
    controls = start_controls(mailbox, a.keyboard_port, a.viewer_port)
    server = viser.ViserServer(host='127.0.0.1', port=a.viewer_port, label='X2 skill phase - experimental')
    class Controller:
        tracker = None
        def reset(self, dones=None):
            # Viewer calls reset(dones) after every step and reset() only for a
            # manual full reset. False done flags must not clear the scheduler.
            if dones is None:
                self.tracker = None
        @torch.no_grad()
        def __call__(self, obs):
            if self.tracker is None:
                self.tracker = SkillTracker(env, a.dataset, cfg, loop_profile=a.loop_profile, straight_heading=a.straight_heading)
            goal = self.tracker.goal(mailbox.mode())
            action = policy(obs['policy'], goal)
            self.tracker.advance()
            return action
    print(f'Experimental keyboard viewer http://127.0.0.1:{a.keyboard_port}', flush=True)
    try:
        ViserPlayViewer(RslRlVecEnvWrapper(env), Controller(), viser_server=server).run()
    finally:
        controls.shutdown(); controls.server_close(); server.stop(); env.close()


if __name__ == '__main__':
    main()
