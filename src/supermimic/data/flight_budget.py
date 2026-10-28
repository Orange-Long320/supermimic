"""Vertical flight estimates for already airborne physical states."""
import math


def rising_ballistic_apex(com_z: float, vz: float, downward_gravity: float) -> float:
    """Estimate remaining apex under uniform gravity, without actuator credit.

    Descending states have no remaining rise. This is a search diagnostic;
    contact-free flight and a successful landing still require simulation.
    """
    if not all(math.isfinite(x) for x in (com_z, vz, downward_gravity)) or downward_gravity <= 0:
        raise ValueError("Expected finite height/velocity and positive downward gravity")
    return com_z + max(vz, 0.0) ** 2 / (2 * downward_gravity)
