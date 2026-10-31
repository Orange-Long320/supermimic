"""Load an exported policy without importing the training or simulation stack."""
class TorchScriptPolicy:
    def __init__(self, path):
        import torch
        self.torch = torch
        self.model = torch.jit.load(str(path), map_location='cpu').eval()

    def infer(self, observation, goal):
        torch = self.torch
        with torch.inference_mode():
            action = self.model(torch.tensor([observation], dtype=torch.float32),
                                torch.tensor([goal], dtype=torch.float32))
        if action.ndim != 2 or action.shape[0] != 1:
            raise ValueError('Unexpected exported policy output shape')
        return tuple(float(x) for x in action[0])


class BufferedTorchPolicy:
    """Single-owner CPU inference with frozen FP32 graph and reusable inputs.

    No quantization, action clipping, weights update, or actuator output. This
    object is not thread-safe; the policy worker must be its sole owner.
    """
    def __init__(self, path, *, observation_dim=990, goal_dim=729, action_dim=31):
        import numpy as np
        import torch
        self.np, self.torch = np, torch
        self.action_dim = action_dim
        self.model = torch.jit.freeze(torch.jit.load(str(path), map_location='cpu').eval())
        self.observation = torch.empty((1, observation_dim), dtype=torch.float32)
        self.goal = torch.empty((1, goal_dim), dtype=torch.float32)
        self.observation_array = self.observation.numpy()[0]
        self.goal_array = self.goal.numpy()[0]

    def infer(self, observation, goal):
        np, torch = self.np, self.torch
        obs, target = np.asarray(observation), np.asarray(goal)
        if (obs.shape != self.observation_array.shape or target.shape != self.goal_array.shape
                or not np.isfinite(obs).all() or not np.isfinite(target).all()):
            raise ValueError('Invalid buffered policy input')
        np.copyto(self.observation_array, obs)
        np.copyto(self.goal_array, target)
        with torch.inference_mode():
            result = self.model(self.observation, self.goal)
        if result.shape != (1, self.action_dim) or not bool(torch.isfinite(result).all()):
            raise ValueError('Invalid buffered policy output')
        # Copy: the caller can retain history independent of the next invocation.
        return result[0].numpy().copy()
