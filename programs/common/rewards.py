"""Pure, simulator-agnostic reward functions for command-conditioned navigation.

These operate on plain torch tensors so they can be unit-tested on CPU without
Isaac Lab. The Isaac Lab reward-term wrappers extract tensors from the env and
call into these functions.
"""

from __future__ import annotations

import torch


def commanded_target_reward(
    robot_xy: torch.Tensor,
    prev_robot_xy: torch.Tensor,
    markers_xy: torch.Tensor,
    target_id: torch.Tensor,
    reach_radius: float = 0.5,
    progress_scale: float = 1.0,
    wrong_penalty_scale: float = 1.0,
    reach_bonus: float = 10.0,
) -> torch.Tensor:
    """Reward shaped by the COMMANDED marker.

    Terms:
      + progress toward the commanded marker (closing distance)
      - penalty for closing distance toward any non-commanded marker
      + bonus when within reach_radius of the commanded marker

    Shapes: robot_xy/prev_robot_xy (N, 2); markers_xy (N, M, 2);
    target_id (N,) long. Returns (N,).
    """
    n = robot_xy.shape[0]
    env_idx = torch.arange(n, device=robot_xy.device)

    # Distance to every marker, now and previously: (N, M)
    dists = torch.linalg.norm(robot_xy.unsqueeze(1) - markers_xy, dim=-1)
    prev_dists = torch.linalg.norm(prev_robot_xy.unsqueeze(1) - markers_xy, dim=-1)
    approach = prev_dists - dists  # >0 means getting closer to that marker

    # Progress toward the commanded marker.
    target_approach = approach[env_idx, target_id]
    progress_term = target_approach * progress_scale

    # Penalty for approaching any wrong marker.
    is_wrong = torch.ones_like(approach, dtype=torch.bool)
    is_wrong[env_idx, target_id] = False
    wrong_approach = torch.where(is_wrong, approach, torch.zeros_like(approach))
    penalty_term = wrong_approach.clamp(min=0.0).sum(dim=-1) * wrong_penalty_scale

    # Bonus for being within reach of the commanded marker.
    target_dist = dists[env_idx, target_id]
    reached = (target_dist < reach_radius).to(robot_xy.dtype)
    bonus_term = reached * reach_bonus

    return progress_term - penalty_term + bonus_term
