"""Explicit fixed box geometry for scene/reference diagnostics.

This does not install a trainable terrain-aware task. Placement must be supplied
after reference alignment; dimensions use the user's approach-facing convention.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FixedBox:
    front_x: float
    center_y: float = 0.0
    depth: float = 0.66
    width: float = 0.75
    height: float = 0.45
    friction: float = 1.0

    def __post_init__(self):
        values = (self.front_x, self.center_y, self.depth, self.width, self.height, self.friction)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Box parameters must be finite")
        if min(self.depth, self.width, self.height, self.friction) <= 0:
            raise ValueError("Box dimensions and friction must be positive")

    @property
    def center(self):
        return (self.front_x + self.depth / 2, self.center_y, self.height / 2)

    @property
    def half_size(self):
        return (self.depth / 2, self.width / 2, self.height / 2)

    def add_to_spec(self, spec):
        import mujoco

        if any(g.name == "box45_collision" for g in spec.geoms):
            raise ValueError("Box geometry already exists")
        spec.worldbody.add_geom(
            name="box45_collision", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=self.center, size=self.half_size,
            contype=1, conaffinity=1, condim=3,
            friction=(self.friction, 0.005, 0.0001),
            rgba=(0.35, 0.45, 0.6, 1.0),
        )

    def top_contains(self, x, y, margin=0.0):
        """Foot point containment; complete sole hulls need their own audit."""
        if not math.isfinite(margin) or margin < 0:
            raise ValueError("Top margin must be finite and nonnegative")
        return (
            self.front_x + margin <= x <= self.front_x + self.depth - margin
            and abs(y - self.center_y) <= self.width / 2 - margin
        )
