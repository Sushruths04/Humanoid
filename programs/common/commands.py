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


def velocity_command_to_target(
    robot_xy: torch.Tensor,
    robot_yaw: torch.Tensor,
    target_xy: torch.Tensor,
    speed: float = 1.0,
    yaw_gain: float = 1.0,
    max_yaw_rate: float = 1.0,
) -> torch.Tensor:
    """Base-frame velocity command (vx, vy, wz) that steers toward target_xy.

    Turn toward the target (wz proportional to heading error, clamped) and walk
    forward only in proportion to how much the robot already faces it. Returns
    (N, 3). robot_xy/target_xy are (N, 2) world-frame; robot_yaw is (N,).
    """
    delta = target_xy - robot_xy
    desired_heading = torch.atan2(delta[..., 1], delta[..., 0])
    # Shortest signed angle error, wrapped to [-pi, pi].
    diff = desired_heading - robot_yaw
    yaw_err = torch.atan2(torch.sin(diff), torch.cos(diff))
    vx = speed * torch.clamp(torch.cos(yaw_err), min=0.0, max=1.0)
    vy = torch.zeros_like(vx)
    wz = torch.clamp(yaw_gain * yaw_err, min=-max_yaw_rate, max=max_yaw_rate)
    return torch.stack([vx, vy, wz], dim=-1)


def velocity_command_to_target_avoiding(
    robot_xy: torch.Tensor,
    robot_yaw: torch.Tensor,
    target_xy: torch.Tensor,
    obstacles_xy: torch.Tensor,
    speed: float = 1.0,
    yaw_gain: float = 1.0,
    max_yaw_rate: float = 1.0,
    avoid_radius: float = 1.5,
    avoid_gain: float = 2.0,
) -> torch.Tensor:
    """Steer toward target while repelling from nearby obstacles (potential field).

    obstacles_xy is (N, K, 2) world-frame. Attraction is a unit vector to the
    target; each obstacle within avoid_radius adds a repulsion pushing the robot
    away, scaled by closeness. The net direction is converted to a base-frame
    (vx, vy, wz) command with the same turn-then-walk law. Returns (N, 3).
    """
    to_target = target_xy - robot_xy
    attract = to_target / (to_target.norm(dim=-1, keepdim=True) + 1e-6)

    to_robot = robot_xy.unsqueeze(1) - obstacles_xy            # (N, K, 2)
    dist = to_robot.norm(dim=-1)                               # (N, K)
    weight = torch.clamp((avoid_radius - dist) / avoid_radius, min=0.0)  # (N, K)
    repel_dir = to_robot / (dist.unsqueeze(-1) + 1e-6)
    repel = (repel_dir * weight.unsqueeze(-1) * avoid_gain).sum(dim=1)   # (N, 2)

    net = attract + repel
    desired_heading = torch.atan2(net[..., 1], net[..., 0])
    diff = desired_heading - robot_yaw
    yaw_err = torch.atan2(torch.sin(diff), torch.cos(diff))
    vx = speed * torch.clamp(torch.cos(yaw_err), min=0.0, max=1.0)
    vy = torch.zeros_like(vx)
    wz = torch.clamp(yaw_gain * yaw_err, min=-max_yaw_rate, max=max_yaw_rate)
    return torch.stack([vx, vy, wz], dim=-1)
