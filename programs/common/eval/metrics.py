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
