"""Sequential multi-goal navigation logic (P1 CP1.4), simulator-agnostic.

Helpers to sample an ordered subgoal sequence and to advance the current phase
when a subgoal is reached. CPU-testable; the Isaac Lab task wires these into a
reset event, a per-step phase-advance event, and a per-subgoal reward.
"""

from __future__ import annotations

import torch


def advance_subgoal(
    phase: torch.Tensor,
    dist_to_current: torch.Tensor,
    reach_radius: float,
    num_subgoals: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance phase when the current subgoal is reached (except on the final one).

    Returns (new_phase, advanced_mask). phase/dist_to_current are (N,).
    """
    reached = dist_to_current < reach_radius
    is_final = phase >= (num_subgoals - 1)
    advanced = reached & (~is_final)
    new_phase = torch.clamp(phase + advanced.long(), max=num_subgoals - 1)
    return new_phase, advanced


def sample_subgoal_sequence(
    num_envs: int,
    num_markers: int,
    num_subgoals: int,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample (num_envs, num_subgoals) ordered distinct marker ids per env.

    Each row is the first num_subgoals entries of a random permutation of the
    markers, so subgoals within a sequence are distinct and ordered.
    """
    scores = torch.rand((num_envs, num_markers), device=device, generator=generator)
    order = scores.argsort(dim=-1)  # random permutation per row
    return order[:, :num_subgoals].long()
