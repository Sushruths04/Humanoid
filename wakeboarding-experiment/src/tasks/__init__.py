"""Gym registration for the wakeboard task."""
from __future__ import annotations

try:
    import gymnasium as gym
    from .wakeboard_start_cfg import WakeboardStartEnv, WakeboardStartEnvCfg

    gym.register(
        id="Humanoid-G1-Wakeboard-Start-v0",
        entry_point="wakeboarding_experiment.src.tasks.wakeboard_start_cfg:WakeboardStartEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": WakeboardStartEnvCfg},
    )
except Exception:
    # CPU / no Isaac Lab — registration skipped.
    pass
