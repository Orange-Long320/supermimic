"""Linear preparation-command basis shared with offline command proposals."""
import torch


def early_contact_command_basis(times):
    """Short native command pulses: initial unloading and early swing support.

    These change servo commands only. They expire by .16 s and make no claim
    about contact feasibility; all proposals require native physics validation.
    """
    return torch.stack([(1-times.abs()/.04).clamp(0, 1),
                        (1-(times-.08).abs()/.08).clamp(0, 1)], -1)


def preparation_command_basis(times, preparation):
    """Weights for initial + three preparation knots, with the native .12s fade."""
    f = (times / preparation).clamp(0, 1) * 3
    index = f.long().clamp(max=2)
    fraction = f - index
    weights = (torch.nn.functional.one_hot(index, 4) * (1 - fraction[..., None])
               + torch.nn.functional.one_hot(index + 1, 4) * fraction[..., None])
    return weights * (1 - (times - preparation) / .12).clamp(0, 1)[..., None]
