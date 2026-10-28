"""Physical limit statistics and constraint priorities for design rollouts."""
import math
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


def core_valid(report):
    limits=report['maxima']
    return (all(math.isfinite(float(x)) for x in [report['cost'],limits['minimum_nonfoot_height'],limits['speed'],limits['effort']])
            and limits['minimum_nonfoot_height']>.005 and limits['speed']<=1 and limits['effort']<=1.00001)


def candidate_priority(report,takeoff=False,preserve_core=False):
    complete=report['physical_candidate_passed'] or (takeoff and report['takeoff_candidate_passed'])
    cost=float(report['cost']);complete=complete and math.isfinite(cost);cost=cost if math.isfinite(cost) else math.inf
    return not complete,not core_valid(report) if preserve_core else False,cost


def contact_repair_eligible(report,minimum_apex):
    """Retain the measured rising single-leg launch while repairing contact."""
    apex=report.get('objective_event',{}).get('ballistic_apex_estimate_m')
    bounds=report['maxima']
    return (apex is not None and math.isfinite(float(apex)) and apex>=minimum_apex
            and core_valid(report) and report['single_left_support_at_launch']
            and report['consecutive_ascending_air_frames']>=3
            and all(math.isfinite(float(bounds[k])) for k in ['hard','foot_penetration']))


def contact_repair_priority(report,minimum_apex):
    """Restore hard/contact feasibility before trading deficits for motion score.

    This ranks proposals only. The original physical acceptance gates stay in
    the rollout evaluator, including its independent full-motion requirements.
    """
    cost=float(report['cost']);cost=cost if math.isfinite(cost) else math.inf
    complete=(report['physical_candidate_passed'] or report['takeoff_candidate_passed']) and math.isfinite(cost)
    eligible=contact_repair_eligible(report,minimum_apex)
    hard=float(report['maxima']['hard']);foot=float(report['maxima']['foot_penetration'])
    hard=hard if math.isfinite(hard) else math.inf
    foot=foot if math.isfinite(foot) else math.inf
    return (not complete,not eligible,hard>=1e-4,max(hard-1e-4,0),
            foot>=.005,max(foot-.005,0),cost)
