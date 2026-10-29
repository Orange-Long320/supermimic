"""X2 identity using the unchanged UniLab Manager environment factory."""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env

registry.register_env_config("X2UltraMotionTracking", ManagerBasedRlEnvCfg)
for backend in ("mujoco", "mjwarp"):
    registry.register_env(
        "X2UltraMotionTracking", make_manager_based_rl_env, sim_backend=backend
    )
