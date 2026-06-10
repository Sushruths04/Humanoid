"""Tests for loco-manipulation state machine.

Run: pytest programs/loco_manip/tests/test_state_machine.py -v
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))

import torch
from programs.loco_manip.state_machine import LocoManipSM, State


def test_starts_in_walk():
    sm = LocoManipSM()
    assert sm.state == State.WALK


def test_transitions_to_approach_when_close():
    sm = LocoManipSM(approach_dist=1.5, grasp_dist=0.5)
    robot_xy = torch.tensor([[0.0, 0.0]])
    obj_xy   = torch.tensor([[1.0, 0.0]])
    sm.update(robot_xy=robot_xy, obj_xy=obj_xy, obj_height=torch.tensor([0.0]))
    assert sm.state == State.APPROACH


def test_stays_walk_when_far():
    sm = LocoManipSM(approach_dist=1.5, grasp_dist=0.5)
    robot_xy = torch.tensor([[0.0, 0.0]])
    obj_xy   = torch.tensor([[5.0, 0.0]])
    sm.update(robot_xy=robot_xy, obj_xy=obj_xy, obj_height=torch.tensor([0.0]))
    assert sm.state == State.WALK


def test_transitions_to_grasp_when_very_close():
    sm = LocoManipSM(approach_dist=1.5, grasp_dist=0.5)
    sm.state = State.APPROACH
    robot_xy = torch.tensor([[0.0, 0.0]])
    obj_xy   = torch.tensor([[0.3, 0.0]])
    sm.update(robot_xy=robot_xy, obj_xy=obj_xy, obj_height=torch.tensor([0.0]))
    assert sm.state == State.GRASP


def test_transitions_to_place_when_object_lifted():
    sm = LocoManipSM()
    sm.state = State.GRASP
    robot_xy = torch.tensor([[0.0, 0.0]])
    obj_xy   = torch.tensor([[0.2, 0.0]])
    sm.update(robot_xy=robot_xy, obj_xy=obj_xy, obj_height=torch.tensor([0.3]))
    assert sm.state == State.PLACE


def test_reset_goes_back_to_walk():
    sm = LocoManipSM()
    sm.state = State.GRASP
    sm.reset()
    assert sm.state == State.WALK
