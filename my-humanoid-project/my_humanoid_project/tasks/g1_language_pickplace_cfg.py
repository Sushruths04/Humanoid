"""Language-conditioned Unitree G1 environment configuration.

This module is intentionally importable on CPU machines without Isaac Sim. When
Isaac Lab is installed, `LanguageConditionedG1EnvCfg` subclasses the stock G1
flat locomotion config and adds a fixed command embedding observation term.
"""

from __future__ import annotations

from my_humanoid_project.language_commands import LANGUAGE_EMBEDDING_DIM, embedding_for_text

BASE_TASK_ID = "Isaac-Velocity-Flat-G1-v0"
LANGUAGE_TASK_ID = "Humanoid-G1-Language-PickPlace-v0" # Keeping the name for continuity in scripts

try:
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.utils import configclass
    from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
    from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg

    ISAACLAB_AVAILABLE = True
except Exception:
    ObsTerm = None
    configclass = None
    G1FlatEnvCfg = object
    G1RoughEnvCfg = object
    ISAACLAB_AVAILABLE = False


def language_command_embedding(env, dim: int = LANGUAGE_EMBEDDING_DIM):
    """Return the current command embedding for every environment instance."""

    import torch

    device = getattr(env, "device", "cpu")
    num_envs = int(getattr(env, "num_envs", 1))
    # In a real VLA, this would change. Here we use a fixed embedding for the "smoke" run.
    text = getattr(env, "language_command_text", "walk forward")
    vec = torch.tensor(embedding_for_text(text, dim), dtype=torch.float32, device=device)
    return vec.unsqueeze(0).repeat(num_envs, 1)


if ISAACLAB_AVAILABLE:

    @configclass
    class LanguageConditionedG1EnvCfg(G1FlatEnvCfg):
        """G1 locomotion task with language conditioning in the policy observation."""

        def __post_init__(self):
            super().__post_init__()
            
            # Append language command to policy observations
            self.observations.policy.language_command = ObsTerm(func=language_command_embedding)
            
            self.language_task_id = LANGUAGE_TASK_ID
            self.base_task_id = BASE_TASK_ID
            self.language_embedding_dim = LANGUAGE_EMBEDDING_DIM


    @configclass
    class LanguageConditionedG1CustomTaskCfg(LanguageConditionedG1EnvCfg):
        """G1 task with language conditioning and visual markers (Red vs Blue)."""

        def __post_init__(self):
            super().__post_init__()

            # Add markers to scene
            from isaaclab.assets import AssetBaseCfg
            import isaaclab.sim as sim_utils

            # Red marker
            self.scene.red_marker = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/RedMarker",
                init_state=AssetBaseCfg.InitialStateCfg(pos=[2.0, 1.0, 0.0]),
                spawn=sim_utils.SphereCfg(
                    radius=0.2,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            )

            # Blue marker
            self.scene.blue_marker = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/BlueMarker",
                init_state=AssetBaseCfg.InitialStateCfg(pos=[2.0, -1.0, 0.0]),
                spawn=sim_utils.SphereCfg(
                    radius=0.2,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                ),
            )

    @configclass
    class LanguageConditionedG1RobustTaskCfg(G1RoughEnvCfg):
        """G1 task with language conditioning, visual markers, and ROUGH terrain robustness."""

        def __post_init__(self):
            super().__post_init__()

            # 1. Add Language Command to Observations
            self.observations.policy.language_command = ObsTerm(func=language_command_embedding)

            # 2. Add Visual Markers
            from isaaclab.assets import AssetBaseCfg
            import isaaclab.sim as sim_utils

            self.scene.red_marker = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/RedMarker",
                init_state=AssetBaseCfg.InitialStateCfg(pos=[2.0, 1.0, 1.0]),
                spawn=sim_utils.SphereCfg(
                    radius=0.2,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            )
            self.scene.blue_marker = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/BlueMarker",
                init_state=AssetBaseCfg.InitialStateCfg(pos=[2.0, -1.0, 1.0]),
                spawn=sim_utils.SphereCfg(
                    radius=0.2,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                ),
            )

            # 3. Add Domain Randomization (Sim-to-Real Robustness)
            if hasattr(self.events, 'randomize_joint_parameters'):
                self.events.randomize_joint_parameters.params["stiffness_distribution_params"] = (0.75, 1.25)
                self.events.randomize_joint_parameters.params["damping_distribution_params"] = (0.75, 1.25)

            self.language_task_id = "Humanoid-G1-Robust-VLA-v0"

else:

    class LanguageConditionedG1EnvCfg:
        """CPU placeholder used for import/syntax checks before GPU setup."""

        language_task_id = LANGUAGE_TASK_ID
        base_task_id = BASE_TASK_ID
        language_embedding_dim = LANGUAGE_EMBEDDING_DIM
