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
