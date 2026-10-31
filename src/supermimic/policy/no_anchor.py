"""Weights-only initialization for native goals without the 9 anchor features.

This equals the source network evaluated with zero anchor inputs. It is an
initialization for world-model/imagination training, not a validated controller.
"""


def remove_anchor_weights(state, *, future_dim, feature_dim=None):
    """Keep every parameter except the anchor columns and their scale buffer."""
    result = dict(state)
    if feature_dim is None:
        scale, weight = 'encoders.motion.scale', 'encoders.motion.mlp.net.0.weight'
        begin = 0
    else:
        scale, weight = 'goal_scale', 'mlp.net.0.weight'
        begin = feature_dim
    if state[scale].ndim != 1 or state[scale].shape[0] != future_dim + 9:
        raise ValueError('Expected future frames plus exactly nine anchor features')
    w = state[weight]
    if w.ndim != 2 or w.shape[1] < begin + future_dim + 9:
        raise ValueError('Unexpected first-layer goal layout')
    if feature_dim is None and w.shape[1] != future_dim + 9:
        raise ValueError('Unexpected policy input width')
    result[scale] = state[scale][:future_dim].clone()
    # Critic features precede the goal; xi and privileged root error follow it.
    import torch
    result[weight] = torch.cat((w[:, :begin+future_dim], w[:, begin+future_dim+9:]), dim=1)
    return result
