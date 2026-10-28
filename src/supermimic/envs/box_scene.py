"""Fixed box geometry and shared simulator/imagination scene features."""
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


BOX_FEATURE_DIM = 9


def box_features(root_pos, root_quat, box: FixedBox):
    """Centre offset, box +x axis in root heading frame, and half sizes (metres).

    Positions are relative to the environment origin. Sharing this function with
    imagination prevents the policy and dynamics from losing box-relative xy/yaw.
    """
    import torch
    from supermimic.kinematics.rotations import heading_quat, quat_apply_inverse

    center = root_pos.new_tensor(box.center)
    axis = torch.zeros_like(root_pos)
    axis[..., 0] = 1
    heading = heading_quat(root_quat)
    return torch.cat((quat_apply_inverse(heading, center - root_pos),
                      quat_apply_inverse(heading, axis),
                      root_pos.new_tensor(box.half_size).expand_as(root_pos)), dim=-1)
