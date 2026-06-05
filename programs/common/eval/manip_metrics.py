"""Pure aggregate metrics for manipulation evaluation.

Simulator-agnostic: takes per-episode outcome tensors and returns summary
numbers. Reused by T0 and later T-track eval harnesses.
"""

from __future__ import annotations

import torch


def compute_manip_metrics(
    grasped: torch.Tensor,
    placed: torch.Tensor,
    dropped: torch.Tensor,
    task_success: torch.Tensor,
    steps_to_success: torch.Tensor,
) -> dict:
    """Aggregate per-episode manipulation outcomes (each shape (N,)) into summary.

    grasped: bool tensor — robot achieved a stable grasp at some point
    placed: bool tensor — robot placed object at target at some point
    dropped: bool tensor — robot dropped object after grasping (fell below grasp height)
    task_success: bool tensor — full task completion (typically grasped + placed + held)
    steps_to_success: long tensor — step at which task_success first triggered, -1 if never
    """
    n = int(grasped.shape[0])
    successful = task_success.bool()
    steps = steps_to_success.float()
    mean_steps = float(steps[successful].mean()) if bool(successful.any()) else float("nan")
    return {
        "num_episodes": n,
        "grasp_success": float(grasped.float().mean()),
        "place_success": float(placed.float().mean()),
        "task_success": float(task_success.float().mean()),
        "object_drop_rate": float(dropped.float().mean()),
        "mean_steps_to_success": mean_steps,
    }


def grasp_then_place_success(
    grasped: torch.Tensor,
    placed: torch.Tensor,
) -> torch.Tensor:
    """Task success = grasped AND placed, shape (N,) bool."""
    return grasped.bool() & placed.bool()


def object_drop_rate_from_heights(
    grasp_height: torch.Tensor,
    min_height_after_grasp: torch.Tensor,
    drop_threshold: float = 0.05,
) -> torch.Tensor:
    """Detect drops: after grasping, object fell > drop_threshold below grasp height.

    grasp_height: (N,) height at grasp moment (meters)
    min_height_after_grasp: (N,) minimum object height recorded after grasp
    Returns (N,) bool — True if a drop occurred.
    """
    return (grasp_height - min_height_after_grasp) > drop_threshold
