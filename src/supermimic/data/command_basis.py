"""Linear preparation-command basis shared with offline command proposals."""
import torch


def preparation_command_basis(times, preparation):
    """Weights for initial + three preparation knots, with the native .12s fade."""
    f = (times / preparation).clamp(0, 1) * 3
    index = f.long().clamp(max=2)
    fraction = f - index
    weights = (torch.nn.functional.one_hot(index, 4) * (1 - fraction[..., None])
               + torch.nn.functional.one_hot(index + 1, 4) * fraction[..., None])
    return weights * (1 - (times - preparation) / .12).clamp(0, 1)[..., None]
