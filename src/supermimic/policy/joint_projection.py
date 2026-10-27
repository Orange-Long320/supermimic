"""Experimental physics-rate motor-command projection; not a safety proof."""
import torch


def project_joint_commands(target,position,velocity,acceleration,previous_torque,
                           stiffness,damping,effort,effective_inertia,lower,upper,
                           margin=.04,frequency=35.):
    """Project servo torque into approximate acceleration-barrier bounds.

    Only position commands change. Motor gains, effort limits, and simulated
    states remain untouched. Observed acceleration estimates the disturbance;
    the diagonal sensitivity approximation requires full physical validation.
    """
    nominal=(stiffness*(target-position)-damping*velocity).clamp(-effort,effort)
    disturbance=acceleration-previous_torque/effective_inertia
    minimum=frequency**2*(lower+margin-position)-2*frequency*velocity
    maximum=frequency**2*(upper-margin-position)-2*frequency*velocity
    torque_low=effective_inertia*(minimum-disturbance)
    torque_high=effective_inertia*(maximum-disturbance)
    infeasible=(torque_low>effort)|(torque_high < -effort)
    torque=torch.maximum(torch.minimum(nominal,torque_high),torque_low).clamp(-effort,effort)
    command=position+(torque+damping*velocity)/stiffness
    return command,(torque-nominal).abs()>1e-5,infeasible
