"""Check analytic proprioceptive history against captured native policy inputs.

This checks recorded transitions only, not world-model accuracy or motion success.
Run through scripts/robot.py for the matching robot dimensions.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from supermimic.robots.dimensions import PROFILE, XI_DIM
from supermimic.world_model.heads import OBS_TERMS, analytic_observation, flatten_history_term_major


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    args = parser.parse_args()
    observation = np.load(args.trace/'baseline-observations.npz')
    trajectory = np.load(args.trace/'baseline-trajectory.npz')
    recorded = torch.tensor(observation['policy'])
    width = sum(d for _, d in OBS_TERMS)
    if recorded.shape[1] % width:
        raise ValueError('Policy history width does not match robot observation terms')
    history_length = recorded.shape[1] // width
    history = []; offset = 0
    for _, dimension in OBS_TERMS:
        history.append(recorded[0, offset:offset+history_length*dimension].reshape(history_length, 1, dimension))
        offset += history_length*dimension
    history = torch.cat(history, dim=-1)
    q0 = torch.tensor([PROFILE['default_joint_pos'][n] for n in PROFILE['joint_names']])
    errors = {name: [] for name, _ in OBS_TERMS}
    history_errors = []
    for t in range(1, len(recorded)):
        state = torch.tensor(np.r_[trajectory['qpos'][t-1], trajectory['qvel'][t-1]])[None]
        previous_action = torch.tensor(trajectory['actions'][t-1])[None]
        current = analytic_observation(state, previous_action, torch.zeros(1, XI_DIM), q0)
        history = torch.cat([history[1:], current[None]], dim=0)
        expected = flatten_history_term_major(history)[0]
        history_errors.append(float((expected-recorded[t]).abs().max()))
        offset = 0
        for name, dimension in OBS_TERMS:
            start = offset+(history_length-1)*dimension
            errors[name].append(float((expected[start:start+dimension]-recorded[t, start:start+dimension]).abs().max()))
            offset += history_length*dimension
    report = {'frames': len(recorded), 'history_length': history_length,
        'maximum_history_error': max(history_errors),
        'maximum_current_term_errors': {k: max(v) for k, v in errors.items()},
        'reference_frame_sequence': observation['reference_frame'].tolist(),
        'passed_at_3e_5': max(history_errors) < 3e-5,
        'scope': 'Noise-free native state/action to policy proprioceptive history; goals and learned predictions not checked'}
    output = args.trace/'feedback-audit.json'
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
