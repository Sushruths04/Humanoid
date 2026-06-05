"""Sequential multi-goal G1 navigation (P1 CP1.4).

The robot must visit an ordered sequence of markers ("go to red, then blue").

Design: this is built on the PROVEN command-nav core (CommandNav reaches 94.5%,
ObstacleNav 85.9%). We reuse its steering, observation, and reward verbatim, and
add ONLY a phase-advance event: when the current subgoal is reached, re-point the
commanded target (`_nav_target_ids`) to the next subgoal. So to the policy this
is exactly command-nav with a target that hops forward through the sequence.

Importable on CPU (no Isaac Sim): Isaac Lab classes only defined when available.
"""

from __future__ import annotations

import os
import sys

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse the proven command-nav machinery (all module-level -> CPU-import-safe).
from .g1_command_nav_cfg import (
    _commanded_target_xy,
    _ensure_buffers,
    _robot_xy,
    nav_command_obs,
    nav_command_reward,
    steer_velocity_to_target,
)

NUM_MARKERS = 3
NUM_SUBGOALS = 2
# Closer targets than command-nav: the robot must reach TWO in sequence, so a
# nearer first subgoal lets the policy bootstrap the reach -> bonus -> progress
# loop before the episode times out (avoids the stand-still local optimum).
RADIUS_RANGE = (1.0, 2.5)
REACH_RADIUS = 0.5
SEQ_NAV_TASK_ID = "Humanoid-G1-SeqNav-v0"

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


def _ensure_seq_buffers(env, num_subgoals):
    if not hasattr(env, "_seq_phase"):
        n, dev = env.num_envs, env.device
        env._seq_targets = torch.zeros(n, num_subgoals, dtype=torch.long, device=dev)
        env._seq_phase = torch.zeros(n, dtype=torch.long, device=dev)


def reset_seq_nav(env, env_ids, num_markers: int = NUM_MARKERS, num_subgoals: int = NUM_SUBGOALS,
                  radius_range=RADIUS_RANGE):
    """Reset: sample markers + an ordered subgoal sequence; point command at subgoal 0."""
    from programs.common.commands import sample_marker_positions
    from programs.common.sequence import sample_subgoal_sequence

    _ensure_buffers(env, num_markers)          # creates _nav_target_ids/_nav_markers_xy/_nav_prev_xy
    _ensure_seq_buffers(env, num_subgoals)
    k = len(env_ids)
    env._nav_markers_xy[env_ids] = sample_marker_positions(k, num_markers, radius_range, device=env.device)
    env._seq_targets[env_ids] = sample_subgoal_sequence(k, num_markers, num_subgoals, device=env.device)
    env._seq_phase[env_ids] = 0
    env._nav_target_ids[env_ids] = env._seq_targets[env_ids, 0]
    env._nav_prev_xy[env_ids] = _robot_xy(env, env_ids)


def seq_advance(env, env_ids, num_markers: int = NUM_MARKERS, num_subgoals: int = NUM_SUBGOALS,
                reach_radius: float = REACH_RADIUS):
    """Per-step (BEFORE steer): when the current subgoal is reached, advance the
    phase and re-point the commanded target to the next subgoal."""
    from programs.common.sequence import advance_subgoal

    _ensure_seq_buffers(env, num_subgoals)
    xy = _robot_xy(env)
    dist = (xy - _commanded_target_xy(env)).norm(dim=-1)
    new_phase, advanced = advance_subgoal(env._seq_phase, dist, reach_radius, num_subgoals)
    env._seq_phase = new_phase
    if bool(advanced.any()):
        idx = torch.arange(env.num_envs, device=env.device)
        next_ids = env._seq_targets[idx, env._seq_phase]
        env._nav_target_ids[advanced] = next_ids[advanced]
        # Reset progress baseline so the target hop doesn't score a false penalty.
        env._nav_prev_xy[advanced] = xy[advanced]


if ISAACLAB_AVAILABLE:

    @configclass
    class SequentialG1NavCfg(G1FlatEnvCfg):
        """G1 navigation through an ordered sequence of markers (proven core + hop)."""

        def __post_init__(self):
            super().__post_init__()

            self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
            if hasattr(self.commands.base_velocity, "heading_command"):
                self.commands.base_velocity.heading_command = False

            self.events.sample_seq_nav = EventTerm(
                func=reset_seq_nav, mode="reset",
                params={"num_markers": NUM_MARKERS, "num_subgoals": NUM_SUBGOALS, "radius_range": RADIUS_RANGE},
            )
            # IMPORTANT: seq_advance is registered BEFORE steer so the steering
            # command uses the freshly-advanced target within the same step.
            self.events.seq_advance = EventTerm(
                func=seq_advance, mode="interval", interval_range_s=(0.0, 0.0),
                params={"num_markers": NUM_MARKERS, "num_subgoals": NUM_SUBGOALS, "reach_radius": REACH_RADIUS},
            )
            self.events.steer_velocity = EventTerm(
                func=steer_velocity_to_target, mode="interval", interval_range_s=(0.0, 0.0),
                params={"num_markers": NUM_MARKERS, "speed": 1.0, "yaw_gain": 0.5, "max_yaw_rate": 1.0},
            )
            self.observations.policy.nav_command = ObsTerm(
                func=nav_command_obs, params={"num_markers": NUM_MARKERS}
            )
            # Reuse the proven command-nav reward (progress + reach bonus toward the
            # current subgoal). wrong-marker penalty disabled: the "wrong" markers
            # here include the *next* subgoal, which we don't want to penalise.
            self.rewards.nav_command = RewTerm(
                func=nav_command_reward, weight=1.0,
                params={"num_markers": NUM_MARKERS, "reach_radius": REACH_RADIUS,
                        "progress_scale": 2.0, "wrong_penalty_scale": 1.0, "reach_bonus": 10.0},
            )
            self.seq_nav_task_id = SEQ_NAV_TASK_ID

else:

    class SequentialG1NavCfg:
        """CPU placeholder for import/syntax checks before GPU setup."""

        seq_nav_task_id = SEQ_NAV_TASK_ID
