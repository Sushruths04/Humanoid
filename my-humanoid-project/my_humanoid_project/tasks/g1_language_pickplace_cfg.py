"""Language-conditioned Unitree G1 pick-place environment configuration.

This module is intentionally importable on CPU machines without Isaac Sim. When
Isaac Lab is installed, `LanguageConditionedG1EnvCfg` subclasses the stock G1
loco-manipulation config and adds a fixed command embedding observation term.
"""

from __future__ import annotations

from my_humanoid_project.language_commands import LANGUAGE_EMBEDDING_DIM, embedding_for_text

BASE_TASK_ID = "Isaac-PickPlace-Locomanipulation-G1-Abs-v0"
LANGUAGE_TASK_ID = "Humanoid-G1-Language-PickPlace-v0"

try:
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.utils import configclass
    from isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_env_cfg import (
        LocomanipulationG1EnvCfg,
        ObservationsCfg,
    )

    ISAACLAB_AVAILABLE = True
except Exception:
    ObsTerm = None
    configclass = None
    LocomanipulationG1EnvCfg = object
    ObservationsCfg = object
    ISAACLAB_AVAILABLE = False


def language_command_embedding(env, dim: int = LANGUAGE_EMBEDDING_DIM):
    """Return the current command embedding for every environment instance."""

    import torch

    device = getattr(env, "device", "cpu")
    num_envs = int(getattr(env, "num_envs", 1))
    text = getattr(env, "language_command_text", "pick up the red cube")
    vec = torch.tensor(embedding_for_text(text, dim), dtype=torch.float32, device=device)
    return vec.unsqueeze(0).repeat(num_envs, 1)


if ISAACLAB_AVAILABLE:

    @configclass
    class LanguageObservationsCfg(ObservationsCfg):
        """Stock G1 observations with a command embedding appended to policy obs."""

        @configclass
        class PolicyCfg(ObservationsCfg.PolicyCfg):
            language_command = ObsTerm(func=language_command_embedding)

        policy: PolicyCfg = PolicyCfg()


    @configclass
    class LanguageConditionedG1EnvCfg(LocomanipulationG1EnvCfg):
        """G1 pick-place task with language conditioning in the policy observation."""

        observations: LanguageObservationsCfg = LanguageObservationsCfg()

        def __post_init__(self):
            super().__post_init__()
            self.language_task_id = LANGUAGE_TASK_ID
            self.base_task_id = BASE_TASK_ID
            self.language_embedding_dim = LANGUAGE_EMBEDDING_DIM

else:

    class LanguageConditionedG1EnvCfg:
        """CPU placeholder used for import/syntax checks before GPU setup."""

        language_task_id = LANGUAGE_TASK_ID
        base_task_id = BASE_TASK_ID
        language_embedding_dim = LANGUAGE_EMBEDDING_DIM

