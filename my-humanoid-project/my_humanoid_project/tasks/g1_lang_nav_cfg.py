"""Language-conditioned G1 navigation (P1).

Like CommandNav, but the command is a natural-language instruction encoded by a
FROZEN text encoder (offline, cached). The policy observation carries the text
embedding of the commanded targets command (not a one-hot), over 3 markers.
Reuses the verified P0 reset/steer/reward helpers; only the observation differs.
"""

from __future__ import annotations

import json
import os
import sys

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

NUM_MARKERS = 3
LANG_NAV_TASK_ID = "Humanoid-G1-LangNav-v0"

try:
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import RewardTermCfg as RewTerm
    from isaaclab.utils import configclass
    from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg

    from .g1_command_nav_cfg import (
        RADIUS_RANGE, _commanded_target_xy, _ensure_buffers, _robot_xy,
        nav_command_reward, reset_nav_command, steer_velocity_to_target,
    )

    ISAACLAB_AVAILABLE = True
except Exception:
    EventTerm = ObsTerm = RewTerm = configclass = None
    G1FlatEnvCfg = object
    ISAACLAB_AVAILABLE = False


def _lang_cache_path() -> str:
    import programs

    return os.path.join(os.path.dirname(programs.__file__), "common", "cache", "nav_command_embeddings.json")


def lang_command_obs(env, num_markers: int = NUM_MARKERS):
    """Observation: frozen text embedding of the commanded target + relative vector."""
    _ensure_buffers(env, num_markers)
    if not hasattr(env, "_lang_embeddings"):
        data = json.loads(open(_lang_cache_path()).read())
        env._lang_embeddings = torch.tensor(data["embeddings"], dtype=torch.float32, device=env.device)
    emb = env._lang_embeddings[env._nav_target_ids]
    rel = _commanded_target_xy(env) - _robot_xy(env)
    return torch.cat([emb, rel], dim=-1)


if ISAACLAB_AVAILABLE:

    @configclass
    class LanguageConditionedG1NavCfg(G1FlatEnvCfg):
        """G1 navigation conditioned on a natural-language command embedding."""

        def __post_init__(self):
            super().__post_init__()

            self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
            if hasattr(self.commands.base_velocity, "heading_command"):
                self.commands.base_velocity.heading_command = False

            self.events.sample_nav_command = EventTerm(
                func=reset_nav_command, mode="reset",
                params={"num_markers": NUM_MARKERS, "radius_range": RADIUS_RANGE},
            )
            self.events.steer_velocity = EventTerm(
                func=steer_velocity_to_target, mode="interval", interval_range_s=(0.0, 0.0),
                params={"num_markers": NUM_MARKERS, "speed": 1.0, "yaw_gain": 0.5, "max_yaw_rate": 1.0},
            )
            self.observations.policy.lang_command = ObsTerm(
                func=lang_command_obs, params={"num_markers": NUM_MARKERS}
            )
            self.rewards.nav_command = RewTerm(
                func=nav_command_reward, weight=1.0,
                params={"num_markers": NUM_MARKERS, "reach_radius": 0.5,
                        "progress_scale": 1.0, "wrong_penalty_scale": 1.0, "reach_bonus": 10.0},
            )
            self.lang_nav_task_id = LANG_NAV_TASK_ID

else:

    class LanguageConditionedG1NavCfg:
        """CPU placeholder for import/syntax checks before GPU setup."""

        lang_nav_task_id = LANG_NAV_TASK_ID
