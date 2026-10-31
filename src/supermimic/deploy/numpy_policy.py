"""Exact FP32 11500 architecture on NumPy BLAS; no quantization or actuation."""
import hashlib
from pathlib import Path

import numpy as np
from scipy.special import expit


SOURCE_POLICY_SHA = '4424f231fc1896da098b2fca366e8522df6791596638b2b9257ba682520bbe4c'


def parameter_shapes():
    shapes = {'encoders.motion.scale': (729,), 'decoder.scale': (990,)}
    for prefix, sizes in [('encoders.motion.mlp.net.', (729, 2048, 1024, 512, 256, 32)),
                          ('decoder.mlp.net.', (1022, 2048, 1024, 512, 256, 31))]:
        for i, (before, after) in enumerate(zip(sizes[:-1], sizes[1:])):
            shapes[prefix+str(2*i)+'.weight'] = (after, before)
            shapes[prefix+str(2*i)+'.bias'] = (after,)
    return shapes


class NumpyPolicy:
    def __init__(self, path, *, expected_sha256):
        path = Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError('NumPy policy archive hash mismatch')
        with np.load(path, allow_pickle=False) as data:
            shapes = parameter_shapes()
            if set(data.files) != set(shapes) | {'source_policy_sha256'} or str(data['source_policy_sha256'].item()) != SOURCE_POLICY_SHA:
                raise ValueError('Wrong policy source or parameter set')
            self.weights = {}
            for key, shape in shapes.items():
                value = data[key]
                if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
                    raise ValueError('Invalid policy parameter: '+key)
                value = np.ascontiguousarray(value); value.flags.writeable = False
                self.weights[key] = value

    def mlp(self, x, prefix):
        for i in (0, 2, 4, 6, 8):
            x = self.weights[prefix+str(i)+'.weight']@x+self.weights[prefix+str(i)+'.bias']
            if i != 8: x *= expit(x)  # SiLU, including its negative branch.
        return x

    def infer(self, observation, goal):
        observation = np.asarray(observation, dtype=np.float32); goal = np.asarray(goal, dtype=np.float32)
        if observation.shape != (990,) or goal.shape != (729,) or not np.isfinite(np.r_[observation, goal]).all():
            raise ValueError('Invalid policy input')
        z = np.tanh(self.mlp(goal*self.weights['encoders.motion.scale'], 'encoders.motion.mlp.net.'))
        action = self.mlp(np.r_[z, observation*self.weights['decoder.scale']], 'decoder.mlp.net.')
        if action.shape != (31,) or not np.isfinite(action).all(): raise ValueError('Invalid policy output')
        return action
