"""Gymnasium task registration for the thesis environments."""

from __future__ import annotations

LANGUAGE_TASK_ID = "Humanoid-G1-Language-PickPlace-v0"


def register_tasks() -> bool:
    """Register custom tasks when Gymnasium is available."""

    try:
        import gymnasium as gym
    except Exception:
        return False

    try:
        gym.spec(LANGUAGE_TASK_ID)
        return True
    except Exception:
        pass

    from .g1_language_pickplace_cfg import LanguageConditionedG1EnvCfg

    gym.register(
        id=LANGUAGE_TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": LanguageConditionedG1EnvCfg},
        disable_env_checker=True,
    )
    return True


register_tasks()

__all__ = ["LANGUAGE_TASK_ID", "register_tasks"]

