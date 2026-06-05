"""Real command-conditioned G1 navigation task (P0).

The task conditions on a command and actually drives navigation:
  - a reset event samples a commanded target id + randomized marker positions,
  - a per-step event steers G1 base-velocity command toward the commanded
    target (reuses G1 locomotion), so velocity-tracking and navigation align,
  - an observation term exposes the command (one-hot + relative target vector),
  - a reward term rewards progress toward the commanded target.

Importable on CPU (no Isaac Sim): Isaac Lab classes only defined when available;
helper functions import the pure logic lazily.
"""

from __future__ import annotations

import os
import sys

import torch

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
    robot = env.scene["robot"]
    if env_ids is None:
        return robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return robot.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]


def _robot_yaw(env):
    from isaaclab.utils.math import euler_xyz_from_quat

    _, _, yaw = euler_xyz_from_quat(env.scene["robot"].data.root_quat_w)
    return yaw


def _ensure_buffers(env, num_markers):
    if not hasattr(env, "_nav_target_ids"):
        n, dev = env.num_envs, env.device
        env._nav_target_ids = torch.zeros(n, dtype=torch.long, device=dev)
        env._nav_markers_xy = torch.zeros(n, num_markers, 2, device=dev)
        env._nav_prev_xy = torch.zeros(n, 2, device=dev)


def _commanded_target_xy(env):
    idx = torch.arange(env.num_envs, device=env.device)
    return env._nav_markers_xy[idx, env._nav_target_ids]


def reset_nav_command(env, env_ids, num_markers: int = NUM_MARKERS, radius_range=RADIUS_RANGE):
    """Reset event: resample commanded target id + marker positions per env."""
    from programs.common.commands import sample_marker_positions, sample_target_ids

    _ensure_buffers(env, num_markers)
    k = len(env_ids)
    env._nav_target_ids[env_ids] = sample_target_ids(k, num_markers, device=env.device)
    env._nav_markers_xy[env_ids] = sample_marker_positions(k, num_markers, radius_range, device=env.device)
    env._nav_prev_xy[env_ids] = _robot_xy(env, env_ids)


def steer_velocity_to_target(
    env, env_ids, num_markers: int = NUM_MARKERS,
    speed: float = 1.0, yaw_gain: float = 0.5, max_yaw_rate: float = 1.0,
):
    """Per-step event: overwrite base-velocity command to steer toward target."""
    from programs.common.commands import velocity_command_to_target

    _ensure_buffers(env, num_markers)
    cmd = velocity_command_to_target(
        _robot_xy(env), _robot_yaw(env), _commanded_target_xy(env),
        speed=speed, yaw_gain=yaw_gain, max_yaw_rate=max_yaw_rate,
    )
    term = env.command_manager.get_term("base_velocity")
    term.vel_command_b[:] = cmd


def nav_command_obs(env, num_markers: int = NUM_MARKERS):
    """Observation: one-hot command + relative vector to the commanded target."""
    from programs.common.commands import target_id_to_onehot

    _ensure_buffers(env, num_markers)
    onehot = target_id_to_onehot(env._nav_target_ids, num_markers)
    rel = _commanded_target_xy(env) - _robot_xy(env)
    return torch.cat([onehot, rel], dim=-1)


def nav_upright_reward(env):
    """Per-step reward for staying upright — reduces fall rate."""
    from programs.common.rewards import upright_reward
    return upright_reward(env.scene["robot"].data.root_quat_w)


def nav_command_reward(
    env, num_markers: int = NUM_MARKERS, reach_radius: float = 0.5,
    progress_scale: float = 1.0, wrong_penalty_scale: float = 1.0, reach_bonus: float = 10.0,
):
    """Reward: progress toward the commanded target (pure fn); updates prev xy."""
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
        """G1 locomotion repurposed for genuine command-conditioned navigation."""

        def __post_init__(self):
            super().__post_init__()

            # Stop the base task from issuing random velocity commands; we steer
            # the command toward the commanded target every step instead.
            self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
            if hasattr(self.commands.base_velocity, "heading_command"):
                self.commands.base_velocity.heading_command = False

            self.events.sample_nav_command = EventTerm(
                func=reset_nav_command, mode="reset",
                params={"num_markers": NUM_MARKERS, "radius_range": RADIUS_RANGE},
            )
            self.events.steer_velocity = EventTerm(
                func=steer_velocity_to_target, mode="interval",
                interval_range_s=(0.0, 0.0),
                params={"num_markers": NUM_MARKERS, "speed": 1.0, "yaw_gain": 0.5, "max_yaw_rate": 1.0},
            )
            self.observations.policy.nav_command = ObsTerm(
                func=nav_command_obs, params={"num_markers": NUM_MARKERS}
            )
            self.rewards.nav_command = RewTerm(
                func=nav_command_reward, weight=1.0,
                params={"num_markers": NUM_MARKERS, "reach_radius": 0.5,
                        "progress_scale": 1.0, "wrong_penalty_scale": 1.0, "reach_bonus": 10.0},
            )
            self.rewards.upright = RewTerm(
                func=nav_upright_reward, weight=0.5,
                params={},
            )
            self.command_nav_task_id = COMMAND_NAV_TASK_ID

else:

    class CommandConditionedG1NavCfg:
        """CPU placeholder for import/syntax checks before GPU setup."""

        command_nav_task_id = COMMAND_NAV_TASK_ID
