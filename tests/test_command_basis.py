import torch
from supermimic.data.command_basis import preparation_command_basis


def test_preparation_knots_and_release():
    times = torch.tensor([0., .2, .4, .6, .66, .72], dtype=torch.float64)
    expected = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                             [0, 0, 0, 1], [0, 0, 0, .5], [0, 0, 0, 0]], dtype=torch.float64)
    torch.testing.assert_close(preparation_command_basis(times, .6), expected)


def test_basis_matches_native_interpolation_with_batched_clocks():
    times = torch.linspace(0, 1.4, 71, dtype=torch.float64)[:, None]
    prep = torch.tensor([.52, .7, 1.1], dtype=torch.float64)[None, :]
    knots = torch.tensor([.03, -.2, .1, .25], dtype=torch.float64)
    f = (times / prep).clamp(0, 1) * 3
    index = f.long().clamp(max=2); fraction = f - index
    native = (knots[index] * (1-fraction) + knots[index+1] * fraction)
    native *= (1-(times-prep)/.12).clamp(0, 1)
    torch.testing.assert_close(preparation_command_basis(times, prep) @ knots, native)
