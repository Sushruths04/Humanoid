"""Pure command/target sampling for command-conditioned navigation.

CPU-testable helpers used by the Isaac Lab task to (re)sample a navigation
command and randomize marker layout at episode reset. Keeping them simulator-
agnostic lets us unit-test the randomization without Isaac Sim.
"""

from __future__ import annotations

import math

import torch


def sample_target_ids(
    num_envs: int,
    num_markers: int,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one commanded marker id per env, uniform in [0, num_markers)."""
    return torch.randint(
        low=0,
        high=num_markers,
        size=(num_envs,),
        device=device,
        generator=generator,
        dtype=torch.long,
    )


def sample_marker_positions(
    num_envs: int,
    num_markers: int,
    radius_range: tuple[float, float] = (2.0, 5.0),
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample (num_envs, num_markers, 2) XY marker positions in an annulus.

    Angles are uniform in [0, 2pi); radii uniform in [r_min, r_max].
    """
    r_min, r_max = radius_range
    angles = torch.rand((num_envs, num_markers), device=device, generator=generator) * (2.0 * math.pi)
    radii = torch.rand((num_envs, num_markers), device=device, generator=generator) * (r_max - r_min) + r_min
    x = radii * torch.cos(angles)
    y = radii * torch.sin(angles)
    return torch.stack([x, y], dim=-1)


def target_id_to_onehot(target_id: torch.Tensor, num_markers: int) -> torch.Tensor:
    """One-hot encode commanded marker ids as (N, num_markers) float for obs."""
    return torch.nn.functional.one_hot(target_id, num_classes=num_markers).float()
