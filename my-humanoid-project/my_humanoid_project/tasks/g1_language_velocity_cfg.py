"""Language-conditioned G1 with REAL command following (language "on").

Subclasses the stock G1 flat velocity-locomotion env and:
  1. samples a discrete language command per env on reset (VEL_COMMANDS),
  2. exposes its hash embedding as a policy observation,
  3. adds a reward that tracks the command's target (vx, vy, yaw) — the actual "on" switch.

CPU-import-guarded like the rest of my_humanoid_project. See ../../PLANS/LANGUAGE_ON_PLAN.md.
VERIFY ON GPU: base-frame velocity accessors + obs/reward manager wiring for the installed
Isaac Lab version.
"""
from __future__ import annotations

import torch

from my_humanoid_project.language_velocity_commands import (
    NUM_COMMANDS,
    command_embeddings,
    command_targets,
    command_track_reward,
    sample_command_ids,
)

LANGUAGE_VELOCITY_TASK_ID = "Humanoid-G1-Language-Velocity-v0"

try:
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import RewardTermCfg as RewTerm
    from isaaclab.utils import configclass
    from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg

    ISAACLAB_AVAILABLE = True
except Exception:
    ISAACLAB_AVAILABLE = False
    def configclass(cls):  # type: ignore
        return cls


# ---- observation + reward callables (need env buffers set in __post_init__/reset) ----
def language_embedding_obs(env):
    """Per-env command embedding (N, LANGUAGE_EMBEDDING_DIM)."""
    return env._cmd_embeddings_table[env._cmd_ids]


def language_command_track(env):
    """Reward tracking the commanded base velocity (the 'on' signal)."""
    d = env.scene["robot"].data
    return command_track_reward(
        d.root_lin_vel_b, d.root_ang_vel_b, env._cmd_ids, env._cmd_targets, sigma=0.5
    )


if ISAACLAB_AVAILABLE:

    @configclass
    class G1LanguageVelocityEnvCfg(G1FlatEnvCfg):
        """G1 flat locomotion + randomized language command + command-tracking reward."""

        def __post_init__(self):
            super().__post_init__()
            # 1) add the command embedding to the policy observation
            self.observations.policy.language_command = ObsTerm(func=language_embedding_obs)
            # 2) add the command-tracking reward (weight tuned in the runner/yaml)
            self.rewards.language_command_track = RewTerm(func=language_command_track, weight=1.5)
            self.language_velocity_task_id = LANGUAGE_VELOCITY_TASK_ID

    def install_command_buffers(env):
        """Call once after env creation: build the embedding/target tables + cmd buffer.
        Hook env.reset to resample commands per episode."""
        env._cmd_embeddings_table = command_embeddings(env.device)   # (NUM_COMMANDS, D)
        env._cmd_targets = command_targets(env.device)               # (NUM_COMMANDS, 3)
        env._cmd_ids = sample_command_ids(env.num_envs, env.device)

        _orig_reset_idx = env._reset_idx

        def _reset_idx(env_ids):
            _orig_reset_idx(env_ids)
            env._cmd_ids[env_ids] = sample_command_ids(len(env_ids), env.device)

        env._reset_idx = _reset_idx   # VERIFY: _reset_idx is the per-env reset hook
        return env
