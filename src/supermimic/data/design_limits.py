"""Device-side physical limit statistics for batched design rollouts."""
import torch


def joint_limit_statistics(qpos,qvel,actuator_force,soft_lower,soft_upper,hard_lower,hard_upper,velocity_limit,effort_limit):
    """Match the original NumPy float64 soft/hard/speed/effort reductions."""
    position=qpos[:,7:].to(torch.float64);velocity=qvel[:,6:].to(torch.float64)
    force=actuator_force.to(torch.float64)
    soft=torch.maximum(soft_lower-position,position-soft_upper).clamp(min=0).amax(dim=1)
    hard=torch.maximum(hard_lower-position,position-hard_upper).clamp(min=0).amax(dim=1)
    speed=(velocity.abs()/velocity_limit).amax(dim=1)
    effort=(force.abs()/effort_limit).amax(dim=1)
    return torch.stack([soft,hard,speed,effort],dim=1)
