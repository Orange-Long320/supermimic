"""Joint-selective early recovery without changing actuator or state limits."""
import torch


def blend_joint_recovery(main_action,recovery_action,residual,global_fraction,early_fraction,indices):
    """Use pure recovery feedback early; restore the existing late residual.

    With no early takeover, the original global blend is returned exactly.
    All inputs and outputs are motor commands, never simulated states.
    """
    result=torch.lerp(main_action,recovery_action+residual,global_fraction[:,None])
    early=torch.lerp(main_action[:,indices],recovery_action[:,indices],early_fraction[:,None])
    early=early+residual[:,indices]*global_fraction[:,None]
    result[:,indices]=torch.where((early_fraction>global_fraction)[:,None],early,result[:,indices])
    return result
