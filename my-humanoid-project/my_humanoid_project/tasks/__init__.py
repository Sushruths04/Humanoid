"""Gymnasium task registration for the thesis environments."""

from __future__ import annotations

LANGUAGE_TASK_ID = "Humanoid-G1-Language-PickPlace-v0"
CUSTOM_TASK_ID = "Humanoid-G1-Custom-MarkerNav-v0"
VISION_VLA_TASK_ID = "Humanoid-G1-Vision-VLA-v0"
ROBUST_TASK_ID = "Humanoid-G1-Robust-VLA-v0"
COMMAND_NAV_TASK_ID = "Humanoid-G1-CommandNav-v0"
LANG_NAV_TASK_ID = "Humanoid-G1-LangNav-v0"
OBSTACLE_NAV_TASK_ID = "Humanoid-G1-ObstacleNav-v0"


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

    print(f"DEBUG: Registering {LANGUAGE_TASK_ID}...")
    gym.register(
        id=LANGUAGE_TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": "my_humanoid_project.tasks.g1_language_pickplace_cfg:LanguageConditionedG1EnvCfg",
            "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        },
        disable_env_checker=True,
    )

    print(f"DEBUG: Registering {CUSTOM_TASK_ID}...")
    gym.register(
        id=CUSTOM_TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": "my_humanoid_project.tasks.g1_language_pickplace_cfg:LanguageConditionedG1CustomTaskCfg",
            "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        },
        disable_env_checker=True,
    )

    print(f"DEBUG: Registering {VISION_VLA_TASK_ID}...")
    gym.register(
        id=VISION_VLA_TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": "my_humanoid_project.tasks.g1_vla_vision_cfg:G1VisionVLAEnvCfg",
            "rsl_rl_cfg_entry_point": "my_humanoid_project.tasks.g1_vla_vision_cfg:G1VisionVLACnnRunnerCfg",
        },
        disable_env_checker=True,
    )

    print(f"DEBUG: Registering {ROBUST_TASK_ID}...")
    gym.register(
        id=ROBUST_TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": "my_humanoid_project.tasks.g1_language_pickplace_cfg:LanguageConditionedG1RobustTaskCfg",
            "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        },
        disable_env_checker=True,
    )


    print(f"DEBUG: Registering {COMMAND_NAV_TASK_ID}...")
    gym.register(
        id=COMMAND_NAV_TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": "my_humanoid_project.tasks.g1_command_nav_cfg:CommandConditionedG1NavCfg",
            "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        },
        disable_env_checker=True,
    )


    print(f"DEBUG: Registering {LANG_NAV_TASK_ID}...")
    gym.register(
        id=LANG_NAV_TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": "my_humanoid_project.tasks.g1_lang_nav_cfg:LanguageConditionedG1NavCfg",
            "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        },
        disable_env_checker=True,
    )


    print(f"DEBUG: Registering {OBSTACLE_NAV_TASK_ID}...")
    gym.register(
        id=OBSTACLE_NAV_TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": "my_humanoid_project.tasks.g1_obstacle_nav_cfg:ObstacleG1NavCfg",
            "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        },
        disable_env_checker=True,
    )

    print("DEBUG: Registration complete.")
    return True


register_tasks()

__all__ = ["LANGUAGE_TASK_ID", "CUSTOM_TASK_ID", "VISION_VLA_TASK_ID", "ROBUST_TASK_ID", "COMMAND_NAV_TASK_ID", "register_tasks"]
