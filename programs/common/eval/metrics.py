"""Pure aggregate metrics for navigation evaluation.

Simulator-agnostic: takes per-episode outcome tensors and returns summary
numbers. Reused by both the humanoid and manipulation eval harnesses.
"""

from __future__ import annotations

import torch


def compute_episode_metrics(
    reached: torch.Tensor,
    fell: torch.Tensor,
    final_distance: torch.Tensor,
    episode_length: torch.Tensor,
) -> dict:
    """Aggregate per-episode outcomes (each shape (N,)) into summary metrics."""
    return {
        "num_episodes": int(reached.shape[0]),
        "success_rate": float(reached.float().mean()),
        "fall_rate": float(fell.float().mean()),
        "mean_final_distance": float(final_distance.float().mean()),
        "mean_episode_length": float(episode_length.float().mean()),
    }


def success_rate_by_command(
    reached: torch.Tensor,
    command_ids: torch.Tensor,
    num_commands: int,
) -> torch.Tensor:
    """Per-command success rate, shape (num_commands,). NaN where no episodes."""
    reached = reached.float()
    rates = torch.full((num_commands,), float("nan"))
    for c in range(num_commands):
        mask = command_ids == c
        if bool(mask.any()):
            rates[c] = reached[mask].mean()
    return rates


def sequence_eval_metrics(reach_steps: torch.Tensor, num_subgoals: int) -> dict:
    """Sequential-navigation metrics for the SeqNav task.

    reach_steps: (N, num_subgoals) long tensor. Entry k is the first timestep at
    which subgoal k marker was reached, or -1 if never reached within the
    episode. A "full sequence" requires every subgoal reached AND visited in the
    commanded order (reach steps non-decreasing along the subgoal axis).

    Returns num_episodes, full_sequence_success, ordering_accuracy,
    first_subgoal_rate. ordering_accuracy is computed over episodes that reached
    ALL subgoals (denominator); it is NaN if no episode reached the full set.
    """
    reach_steps = reach_steps.long()
    n = int(reach_steps.shape[0])
    reached = reach_steps >= 0
    reached_all = reached.all(dim=1)
    in_order = (reach_steps[:, 1:] >= reach_steps[:, :-1]).all(dim=1)
    full = reached_all & in_order

    denom = int(reached_all.sum())
    ordering = float((reached_all & in_order).sum()) / denom if denom > 0 else float("nan")
    return {
        "num_episodes": n,
        "num_subgoals": int(num_subgoals),
        "full_sequence_success": float(full.float().mean()),
        "ordering_accuracy": ordering,
        "first_subgoal_rate": float(reached[:, 0].float().mean()),
    }
