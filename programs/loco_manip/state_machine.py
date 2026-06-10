"""Loco-manipulation state machine.

Controls which policy is active at each moment:
  WALK     → nav policy (model_499.pt) drives the robot toward the table
  APPROACH → nav policy slows down, robot orients toward object
  GRASP    → arm policy takes over, legs hold position
  PLACE    → arm policy places object at target location
  DONE     → episode complete

Usage:
    sm = LocoManipSM()
    sm.update(robot_xy, obj_xy, obj_height)
    if sm.state == State.WALK:
        actions = nav_policy(obs)
    elif sm.state == State.GRASP:
        actions = arm_policy(obs)
"""
from __future__ import annotations
from enum import Enum, auto
import torch


class State(Enum):
    WALK     = auto()  # Walking toward the table
    APPROACH = auto()  # Within approach distance, slowing down
    GRASP    = auto()  # Arms active, picking up object
    PLACE    = auto()  # Arms placing object at target
    DONE     = auto()  # Task complete


class LocoManipSM:
    """State machine for staged loco-manipulation.

    Args:
        approach_dist: Distance (metres) that triggers WALK→APPROACH (default 1.5m)
        grasp_dist:    Distance (metres) that triggers APPROACH→GRASP (default 0.5m)
        lift_height:   Object z-height (metres) that triggers GRASP→PLACE (default 0.2m)
    """

    def __init__(
        self,
        approach_dist: float = 1.5,
        grasp_dist: float = 0.5,
        lift_height: float = 0.2,
    ):
        self.approach_dist = approach_dist
        self.grasp_dist    = grasp_dist
        self.lift_height   = lift_height
        self.state         = State.WALK

    def reset(self) -> None:
        self.state = State.WALK

    def update(
        self,
        robot_xy: torch.Tensor,    # (N, 2) robot xy position
        obj_xy:   torch.Tensor,    # (N, 2) object xy position
        obj_height: torch.Tensor,  # (N,)   object z position above ground
    ) -> State:
        """Advance state based on current world state. Returns the new state."""
        dist = torch.norm(obj_xy - robot_xy, dim=-1).item()

        if self.state == State.WALK:
            if dist < self.approach_dist:
                self.state = State.APPROACH

        elif self.state == State.APPROACH:
            if dist < self.grasp_dist:
                self.state = State.GRASP

        elif self.state == State.GRASP:
            if obj_height.item() > self.lift_height:
                self.state = State.PLACE

        elif self.state == State.PLACE:
            pass  # place completion added when we have a target location

        return self.state

    @property
    def nav_active(self) -> bool:
        return self.state in (State.WALK, State.APPROACH)

    @property
    def arm_active(self) -> bool:
        return self.state in (State.GRASP, State.PLACE)
