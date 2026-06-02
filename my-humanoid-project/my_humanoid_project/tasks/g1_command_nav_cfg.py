"""Real command-conditioned G1 navigation task (P0).

Unlike the earlier decorative marker config, this task actually conditions on a
command:
  - a reset event samples a commanded target id + randomized marker positions
    per episode (programs.common.commands),
  - an observation term exposes the command (one-hot) plus the relative vector
    to the COMMANDED target,
  - a reward term rewards progress toward the commanded target and penalizes
    drifting toward the wrong one (programs.common.rewards).

Importable on CPU (no Isaac Sim): the Isaac Lab classes are only defined when
Isaac Lab is available; the helper functions import the pure logic lazily.
"""

from __future__ import annotations

import os
import sys

import torch

# Make the repo-root `programs` package importable from inside this package.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

NUM_MARKERS = 2
RADIUS_RANGE = (2.0, 5.0)
COMMAND_NAV_TASK_ID = "Humanoid-G1-CommandNav-v0"

try:
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import RewardTermCfg as RewTerm
    from isaaclab.utils import configclass
    from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg

    ISAACLAB_AVAILABLE = True
except Exception:
    EventTerm = ObsTerm = RewTerm = configclass = None
    G1FlatEnvCfg = object
    ISAACLAB_AVAILABLE = False


def _robot_xy(env, env_ids=None):
    """Robot planar position relative to its per-env origin."""
    robot = env.scene["robot"]
    if env_ids is None:
        return robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return robot.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]


def _ensure_buffers(env, num_markers):
    if not hasattr(env, "_nav_target_ids"):
        n, dev = env.num_envs, env.device
        env._nav_target_ids = torch.zeros(n, dtype=torch.long, device=dev)
        env._nav_markers_xy = torch.zeros(n, num_markers, 2, device=dev)
        env._nav_prev_xy = torch.zeros(n, 2, device=dev)


def reset_nav_command(env, env_ids, num_markers: int = NUM_MARKERS, radius_range=RADIUS_RANGE):
    """Reset event: resample commanded target id + marker positions per env."""
    from programs.common.commands import sample_marker_positions, sample_target_ids

    _ensure_buffers(env, num_markers)
    k = len(env_ids)
    env._nav_target_ids[env_ids] = sample_target_ids(k, num_markers, device=env.device)
    env._nav_markers_xy[env_ids] = sample_marker_positions(k, num_markers, radius_range, device=env.device)
    env._nav_prev_xy[env_ids] = _robot_xy(env, env_ids)


def nav_command_obs(env, num_markers: int = NUM_MARKERS):
    """Observation: one-hot command + relative vector to the commanded target."""
    from programs.common.commands import target_id_to_onehot

    _ensure_buffers(env, num_markers)
    onehot = target_id_to_onehot(env._nav_target_ids, num_markers)
    idx = torch.arange(env.num_envs, device=env.device)
    target_xy = env._nav_markers_xy[idx, env._nav_target_ids]
    rel = target_xy - _robot_xy(env)
    return torch.cat([onehot, rel], dim=-1)


def nav_command_reward(
    env,
    num_markers: int = NUM_MARKERS,
    reach_radius: float = 0.5,
    progress_scale: float = 1.0,
    wrong_penalty_scale: float = 1.0,
    reach_bonus: float = 10.0,
):
    """Reward: progress toward the commanded target (pure fn), updates prev xy."""
    from programs.common.rewards import commanded_target_reward

    _ensure_buffers(env, num_markers)
    xy = _robot_xy(env)
    reward = commanded_target_reward(
        xy, env._nav_prev_xy, env._nav_markers_xy, env._nav_target_ids,
        reach_radius=reach_radius, progress_scale=progress_scale,
        wrong_penalty_scale=wrong_penalty_scale, reach_bonus=reach_bonus,
    )
    env._nav_prev_xy = xy.clone()
    return reward


if ISAACLAB_AVAILABLE:

    @configclass
    class CommandConditionedG1NavCfg(G1FlatEnvCfg):
        """G1 flat locomotion + genuine command-conditioned navigation."""

        def __post_init__(self):
            super().__post_init__()

            self.events.sample_nav_command = EventTerm(
                func=reset_nav_command,
                mode="reset",
                params={"num_markers": NUM_MARKERS, "radius_range": RADIUS_RANGE},
            )
            self.observations.policy.nav_command = ObsTerm(
                func=nav_command_obs, params={"num_markers": NUM_MARKERS}
            )
            self.rewards.nav_command = RewTerm(
                func=nav_command_reward,
                weight=1.0,
                params={
                    "num_markers": NUM_MARKERS,
                    "reach_radius": 0.5,
                    "progress_scale": 1.0,
                    "wrong_penalty_scale": 1.0,
                    "reach_bonus": 10.0,
                },
            )
            self.command_nav_task_id = COMMAND_NAV_TASK_ID

else:

    class CommandConditionedG1NavCfg:
        """CPU placeholder for import/syntax checks before GPU setup."""

        command_nav_task_id = COMMAND_NAV_TASK_ID
