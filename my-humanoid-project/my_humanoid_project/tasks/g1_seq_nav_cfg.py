"""Sequential multi-goal G1 navigation (P1 CP1.4).

The robot must visit an ordered sequence of markers ("go to red, then blue").
A per-step event tracks the current subgoal, advances it when reached, and
steers toward the active subgoal; the reward gives progress + a bonus per
completed subgoal + a final-reach bonus. Importable on CPU.
"""

from __future__ import annotations

import os
import sys

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

NUM_MARKERS = 3
NUM_SUBGOALS = 2
RADIUS_RANGE = (2.0, 5.0)
REACH_RADIUS = 0.5
SEQ_NAV_TASK_ID = "Humanoid-G1-SeqNav-v0"

try:
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import RewardTermCfg as RewTerm
    from isaaclab.utils import configclass
    from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg

    from .g1_command_nav_cfg import _robot_xy, _robot_yaw

    ISAACLAB_AVAILABLE = True
except Exception:
    EventTerm = ObsTerm = RewTerm = configclass = None
    G1FlatEnvCfg = object
    ISAACLAB_AVAILABLE = False


def _ensure_seq_buffers(env, num_markers, num_subgoals):
    if not hasattr(env, "_seq_phase"):
        n, dev = env.num_envs, env.device
        env._seq_markers_xy = torch.zeros(n, num_markers, 2, device=dev)
        env._seq_targets = torch.zeros(n, num_subgoals, dtype=torch.long, device=dev)
        env._seq_phase = torch.zeros(n, dtype=torch.long, device=dev)
        env._seq_prev_xy = torch.zeros(n, 2, device=dev)
        env._seq_advanced = torch.zeros(n, dtype=torch.bool, device=dev)


def _current_target_xy(env):
    idx = torch.arange(env.num_envs, device=env.device)
    marker_id = env._seq_targets[idx, env._seq_phase]
    return env._seq_markers_xy[idx, marker_id]


def reset_seq_nav(env, env_ids, num_markers: int = NUM_MARKERS, num_subgoals: int = NUM_SUBGOALS,
                  radius_range=RADIUS_RANGE):
    from programs.common.commands import sample_marker_positions
    from programs.common.sequence import sample_subgoal_sequence

    _ensure_seq_buffers(env, num_markers, num_subgoals)
    k = len(env_ids)
    env._seq_markers_xy[env_ids] = sample_marker_positions(k, num_markers, radius_range, device=env.device)
    env._seq_targets[env_ids] = sample_subgoal_sequence(k, num_markers, num_subgoals, device=env.device)
    env._seq_phase[env_ids] = 0
    env._seq_prev_xy[env_ids] = _robot_xy(env, env_ids)
    env._seq_advanced[env_ids] = False


def seq_step(env, env_ids, num_markers: int = NUM_MARKERS, num_subgoals: int = NUM_SUBGOALS,
             reach_radius: float = REACH_RADIUS, speed: float = 1.0, yaw_gain: float = 0.5,
             max_yaw_rate: float = 1.0):
    from programs.common.commands import velocity_command_to_target
    from programs.common.sequence import advance_subgoal

    _ensure_seq_buffers(env, num_markers, num_subgoals)
    xy = _robot_xy(env)
    dist = (xy - _current_target_xy(env)).norm(dim=-1)
    new_phase, advanced = advance_subgoal(env._seq_phase, dist, reach_radius, num_subgoals)
    env._seq_phase = new_phase
    env._seq_advanced = advanced
    if bool(advanced.any()):
        env._seq_prev_xy[advanced] = xy[advanced]
    cmd = velocity_command_to_target(xy, _robot_yaw(env), _current_target_xy(env),
                                     speed=speed, yaw_gain=yaw_gain, max_yaw_rate=max_yaw_rate)
    env.command_manager.get_term("base_velocity").vel_command_b[:] = cmd


def seq_obs(env, num_markers: int = NUM_MARKERS, num_subgoals: int = NUM_SUBGOALS):
    from programs.common.commands import target_id_to_onehot

    _ensure_seq_buffers(env, num_markers, num_subgoals)
    idx = torch.arange(env.num_envs, device=env.device)
    cur_marker = env._seq_targets[idx, env._seq_phase]
    onehot = target_id_to_onehot(cur_marker, num_markers)
    denom = max(num_subgoals - 1, 1)
    phase_norm = (env._seq_phase.float() / denom).unsqueeze(-1)
    rel = _current_target_xy(env) - _robot_xy(env)
    return torch.cat([onehot, phase_norm, rel], dim=-1)


def seq_reward(env, num_markers: int = NUM_MARKERS, num_subgoals: int = NUM_SUBGOALS,
               reach_radius: float = REACH_RADIUS, progress_scale: float = 1.0,
               advance_bonus: float = 10.0, final_bonus: float = 10.0):
    _ensure_seq_buffers(env, num_markers, num_subgoals)
    xy = _robot_xy(env)
    target = _current_target_xy(env)
    dist = (xy - target).norm(dim=-1)
    prev_dist = (env._seq_prev_xy - target).norm(dim=-1)
    progress = (prev_dist - dist) * progress_scale
    advanced_bonus = env._seq_advanced.float() * advance_bonus
    is_final = env._seq_phase >= (num_subgoals - 1)
    final_reach = (is_final & (dist < reach_radius)).float() * final_bonus
    env._seq_prev_xy = xy.clone()
    return progress + advanced_bonus + final_reach


if ISAACLAB_AVAILABLE:

    @configclass
    class SequentialG1NavCfg(G1FlatEnvCfg):
        """G1 navigation through an ordered sequence of markers."""

        def __post_init__(self):
            super().__post_init__()

            self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
            if hasattr(self.commands.base_velocity, "heading_command"):
                self.commands.base_velocity.heading_command = False

            self.events.sample_seq_nav = EventTerm(
                func=reset_seq_nav, mode="reset",
                params={"num_markers": NUM_MARKERS, "num_subgoals": NUM_SUBGOALS, "radius_range": RADIUS_RANGE},
            )
            self.events.seq_step = EventTerm(
                func=seq_step, mode="interval", interval_range_s=(0.0, 0.0),
                params={"num_markers": NUM_MARKERS, "num_subgoals": NUM_SUBGOALS, "reach_radius": REACH_RADIUS,
                        "speed": 1.0, "yaw_gain": 0.5, "max_yaw_rate": 1.0},
            )
            self.observations.policy.seq_nav = ObsTerm(
                func=seq_obs, params={"num_markers": NUM_MARKERS, "num_subgoals": NUM_SUBGOALS}
            )
            self.rewards.seq_nav = RewTerm(
                func=seq_reward, weight=1.0,
                params={"num_markers": NUM_MARKERS, "num_subgoals": NUM_SUBGOALS, "reach_radius": REACH_RADIUS,
                        "progress_scale": 1.0, "advance_bonus": 10.0, "final_bonus": 10.0},
            )
            self.seq_nav_task_id = SEQ_NAV_TASK_ID

else:

    class SequentialG1NavCfg:
        """CPU placeholder for import/syntax checks before GPU setup."""

        seq_nav_task_id = SEQ_NAV_TASK_ID
