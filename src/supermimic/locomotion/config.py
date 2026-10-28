"""Conservative first-stage commands; values are a training proposal, not measured limits."""

from dataclasses import dataclass


@dataclass
class VelocityCfg:
    enabled: bool = False
    fraction: float = 0.5
    forward: tuple[float, float] = (0.0, 0.8)
    lateral: tuple[float, float] = (0.0, 0.0)
    yaw: tuple[float, float] = (-0.6, 0.6)
    resample_seconds: float = 4.0
    standing_probability: float = 0.25
    reference_probability: float = 0.5
    reference_tolerance: float = 0.05
    alignment_weight: float = 0.1
    stop_relabel_fraction: float = 0.0  # imagine stop from real states, independent of recorded goal
    min_height_fraction: float = 0.6
    max_tilt: float = 0.7  # 1 - upright root z-axis
    encoder_hidden: tuple[int, ...] = (256, 256)

    def validate(self):
        import math

        if not 0 < self.fraction < 1:
            raise ValueError("Joint training needs both motion and velocity environments")
        for bounds in (self.forward, self.lateral, self.yaw):
            if len(bounds) != 2 or not all(math.isfinite(v) for v in bounds) or bounds[0] > bounds[1]:
                raise ValueError("Velocity ranges must be finite ordered pairs")
            if not bounds[0] <= 0 <= bounds[1]:
                raise ValueError("Velocity ranges must include zero for stopping")
        for p in (self.standing_probability, self.reference_probability, self.stop_relabel_fraction):
            if not 0 <= p <= 1:
                raise ValueError("Velocity probabilities must be in [0, 1]")
        if self.resample_seconds <= 0 or not 0 < self.min_height_fraction < 1 or not 0 < self.max_tilt < 2:
            raise ValueError("Invalid resampling period or fall thresholds")
        if not math.isfinite(self.reference_tolerance) or self.reference_tolerance < 0:
            raise ValueError("Reference tolerance must be finite and nonnegative")

    @property
    def command_scale(self):
        return tuple(max(abs(lo), abs(hi), 0.1) for lo, hi in (self.forward, self.lateral, self.yaw))
