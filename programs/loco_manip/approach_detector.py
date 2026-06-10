"""Approach detector — computes distance and bearing to target object.

Tells the state machine and nav policy how far/which direction the object is.
"""
from __future__ import annotations
import torch
import math


class ApproachDetector:
    """Computes distance and bearing from robot to target object.

    Args:
        reach_dist: Distance (m) at which the robot can attempt a grasp (default 0.5m)
    """

    def __init__(self, reach_dist: float = 0.5):
        self.reach_dist = reach_dist

    def distance(self, robot_xy: torch.Tensor, obj_xy: torch.Tensor) -> torch.Tensor:
        """Euclidean distance from robot to object. Shape: (N,)"""
        return torch.norm(obj_xy - robot_xy, dim=-1)

    def bearing(
        self,
        robot_xy: torch.Tensor,  # (N, 2)
        obj_xy: torch.Tensor,    # (N, 2)
        yaw: torch.Tensor,       # (N,) robot yaw in radians
    ) -> torch.Tensor:
        """Signed angle from robot heading to object direction. Shape: (N,)

        Positive = object is to the left (counter-clockwise).
        Zero     = object is directly ahead.
        """
        delta = obj_xy - robot_xy
        world_angle = torch.atan2(delta[:, 1], delta[:, 0])
        bearing = world_angle - yaw
        bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
        return bearing

    def within_reach(self, robot_xy: torch.Tensor, obj_xy: torch.Tensor) -> bool:
        """True when the robot is close enough to attempt grasping."""
        return self.distance(robot_xy, obj_xy).item() < self.reach_dist

    def approach_obs(
        self,
        robot_xy: torch.Tensor,
        obj_xy: torch.Tensor,
        yaw: torch.Tensor,
    ) -> torch.Tensor:
        """4-dim observation: [dist, bearing_sin, bearing_cos, within_reach]. Shape: (N, 4)"""
        dist  = self.distance(robot_xy, obj_xy).unsqueeze(-1)
        b     = self.bearing(robot_xy, obj_xy, yaw)
        b_sin = torch.sin(b).unsqueeze(-1)
        b_cos = torch.cos(b).unsqueeze(-1)
        reach = (dist < self.reach_dist).float()
        return torch.cat([dist, b_sin, b_cos, reach], dim=-1)
