"""Tests for approach detector.

Run: pytest programs/loco_manip/tests/test_approach_detector.py -v
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))
import torch, math
from programs.loco_manip.approach_detector import ApproachDetector


def test_distance_correct():
    det = ApproachDetector()
    robot = torch.tensor([[0.0, 0.0]])
    obj   = torch.tensor([[3.0, 4.0]])
    assert abs(det.distance(robot, obj).item() - 5.0) < 1e-4


def test_bearing_facing_right():
    det = ApproachDetector()
    robot = torch.tensor([[0.0, 0.0]])
    obj   = torch.tensor([[1.0, 0.0]])
    bearing = det.bearing(robot, obj, yaw=torch.tensor([0.0]))
    assert abs(bearing.item()) < 1e-4


def test_bearing_object_to_left():
    det = ApproachDetector()
    robot = torch.tensor([[0.0, 0.0]])
    obj   = torch.tensor([[0.0, 1.0]])  # object at +y = 90° left
    bearing = det.bearing(robot, obj, yaw=torch.tensor([0.0]))
    assert abs(bearing.item() - math.pi / 2) < 1e-3


def test_within_reach():
    det = ApproachDetector(reach_dist=0.5)
    robot = torch.tensor([[0.0, 0.0]])
    obj   = torch.tensor([[0.4, 0.0]])
    assert det.within_reach(robot, obj) is True


def test_not_within_reach():
    det = ApproachDetector(reach_dist=0.5)
    robot = torch.tensor([[0.0, 0.0]])
    obj   = torch.tensor([[0.6, 0.0]])
    assert det.within_reach(robot, obj) is False
