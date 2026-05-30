"""Gymnasium task registration for the thesis environments."""

from __future__ import annotations

LANGUAGE_TASK_ID = "Humanoid-G1-Language-PickPlace-v0"
CUSTOM_TASK_ID = "Humanoid-G1-Custom-MarkerNav-v0"


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

    gym.register(
        id=LANGUAGE_TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": "my_humanoid_project.tasks.g1_language_pickplace_cfg:LanguageConditionedG1EnvCfg",
            "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        },
        disable_env_checker=True,
    )

    gym.register(
        id=CUSTOM_TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": "my_humanoid_project.tasks.g1_language_pickplace_cfg:LanguageConditionedG1CustomTaskCfg",
            "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        },
        disable_env_checker=True,
    )
    return True


register_tasks()

__all__ = ["LANGUAGE_TASK_ID", "CUSTOM_TASK_ID", "register_tasks"]

