"""Obstacle-aware command-conditioned G1 navigation (P1 CP1.3).

Builds on CommandNav: in addition to a commanded marker, K obstacles are sampled
in the path each episode. The base-velocity command is steered toward the target
while repelling from nearby obstacles (potential field), the policy observes
relative obstacle positions, and a collision penalty discourages getting close.

State-based pass: obstacles are penalty regions (positions in a buffer) rather
than physical/visual bodies; physical+visual obstacles arrive with the vision
phase. Importable on CPU; Isaac Lab classes defined only when available.
"""

from __future__ import annotations

import os
import sys

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

NUM_MARKERS = 2
NUM_OBSTACLES = 3
RADIUS_RANGE = (2.5, 5.0)
OBSTACLE_RADIUS_RANGE = (1.0, 3.0)
COLLISION_RADIUS = 0.5
OBSTACLE_NAV_TASK_ID = "Humanoid-G1-ObstacleNav-v0"

try:
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import RewardTermCfg as RewTerm
    from isaaclab.utils import configclass
    from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg

    from .g1_command_nav_cfg import _commanded_target_xy, _ensure_buffers, _robot_xy, _robot_yaw, nav_command_reward

    ISAACLAB_AVAILABLE = True
except Exception:
    EventTerm = ObsTerm = RewTerm = configclass = None
    G1FlatEnvCfg = object
    ISAACLAB_AVAILABLE = False


def _ensure_obstacles(env, num_obstacles):
    if not hasattr(env, "_nav_obstacles_xy"):
        env._nav_obstacles_xy = torch.zeros(env.num_envs, num_obstacles, 2, device=env.device)


def reset_obstacle_nav(env, env_ids, num_markers: int = NUM_MARKERS, num_obstacles: int = NUM_OBSTACLES,
                       radius_range=RADIUS_RANGE, obstacle_radius_range=OBSTACLE_RADIUS_RANGE):
    from programs.common.commands import sample_marker_positions, sample_target_ids

    _ensure_buffers(env, num_markers)
    _ensure_obstacles(env, num_obstacles)
    k = len(env_ids)
    env._nav_target_ids[env_ids] = sample_target_ids(k, num_markers, device=env.device)
    env._nav_markers_xy[env_ids] = sample_marker_positions(k, num_markers, radius_range, device=env.device)
    env._nav_obstacles_xy[env_ids] = sample_marker_positions(k, num_obstacles, obstacle_radius_range, device=env.device)
    env._nav_prev_xy[env_ids] = _robot_xy(env, env_ids)


def steer_avoiding(env, env_ids, num_markers: int = NUM_MARKERS, speed: float = 1.0, yaw_gain: float = 0.5,
                   max_yaw_rate: float = 1.0, avoid_radius: float = 1.5, avoid_gain: float = 2.0):
    from programs.common.commands import velocity_command_to_target_avoiding

    _ensure_buffers(env, num_markers)
    cmd = velocity_command_to_target_avoiding(
        _robot_xy(env), _robot_yaw(env), _commanded_target_xy(env), env._nav_obstacles_xy,
        speed=speed, yaw_gain=yaw_gain, max_yaw_rate=max_yaw_rate, avoid_radius=avoid_radius, avoid_gain=avoid_gain,
    )
    env.command_manager.get_term("base_velocity").vel_command_b[:] = cmd


def obstacle_nav_obs(env, num_markers: int = NUM_MARKERS, num_obstacles: int = NUM_OBSTACLES):
    from programs.common.commands import target_id_to_onehot

    _ensure_buffers(env, num_markers)
    _ensure_obstacles(env, num_obstacles)
    onehot = target_id_to_onehot(env._nav_target_ids, num_markers)
    xy = _robot_xy(env)
    rel_target = _commanded_target_xy(env) - xy
    rel_obs = (env._nav_obstacles_xy - xy.unsqueeze(1)).reshape(env.num_envs, -1)
    return torch.cat([onehot, rel_target, rel_obs], dim=-1)


def obstacle_collision_reward(env, num_obstacles: int = NUM_OBSTACLES,
                              collision_radius: float = COLLISION_RADIUS, penalty_scale: float = 1.0):
    from programs.common.rewards import collision_penalty

    _ensure_obstacles(env, num_obstacles)
    return collision_penalty(_robot_xy(env), env._nav_obstacles_xy,
                             collision_radius=collision_radius, penalty_scale=penalty_scale)


if ISAACLAB_AVAILABLE:

    @configclass
    class ObstacleG1NavCfg(G1FlatEnvCfg):
        """Command-conditioned G1 navigation with obstacle avoidance."""

        def __post_init__(self):
            super().__post_init__()

            self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
            if hasattr(self.commands.base_velocity, "heading_command"):
                self.commands.base_velocity.heading_command = False

            self.events.sample_obstacle_nav = EventTerm(
                func=reset_obstacle_nav, mode="reset",
                params={"num_markers": NUM_MARKERS, "num_obstacles": NUM_OBSTACLES,
                        "radius_range": RADIUS_RANGE, "obstacle_radius_range": OBSTACLE_RADIUS_RANGE},
            )
            self.events.steer_velocity = EventTerm(
                func=steer_avoiding, mode="interval", interval_range_s=(0.0, 0.0),
                params={"num_markers": NUM_MARKERS, "speed": 1.0, "yaw_gain": 0.5, "max_yaw_rate": 1.0,
                        "avoid_radius": 1.5, "avoid_gain": 2.0},
            )
            self.observations.policy.obstacle_nav = ObsTerm(
                func=obstacle_nav_obs, params={"num_markers": NUM_MARKERS, "num_obstacles": NUM_OBSTACLES}
            )
            self.rewards.nav_command = RewTerm(
                func=nav_command_reward, weight=1.0,
                params={"num_markers": NUM_MARKERS, "reach_radius": 0.5, "progress_scale": 1.0,
                        "wrong_penalty_scale": 1.0, "reach_bonus": 10.0},
            )
            self.rewards.collision = RewTerm(
                func=obstacle_collision_reward, weight=1.0,
                params={"num_obstacles": NUM_OBSTACLES, "collision_radius": COLLISION_RADIUS, "penalty_scale": 1.0},
            )
            self.obstacle_nav_task_id = OBSTACLE_NAV_TASK_ID

else:

    class ObstacleG1NavCfg:
        """CPU placeholder for import/syntax checks before GPU setup."""

        obstacle_nav_task_id = OBSTACLE_NAV_TASK_ID
