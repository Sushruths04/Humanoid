import torch
from programs.common.eval.manip_metrics import (
    compute_manip_metrics,
    grasp_then_place_success,
    object_drop_rate_from_heights,
)


def _basic_batch():
    # 4 episodes: all combos of grasp/place
    grasped = torch.tensor([True, True, False, True])
    placed = torch.tensor([True, False, False, True])
    dropped = torch.tensor([False, True, False, False])
    task_success = grasp_then_place_success(grasped, placed)
    steps = torch.tensor([50, -1, -1, 80], dtype=torch.long)
    return grasped, placed, dropped, task_success, steps


def test_compute_manip_metrics_shape():
    g, p, d, t, s = _basic_batch()
    m = compute_manip_metrics(g, p, d, t, s)
    assert m["num_episodes"] == 4
    for key in ("grasp_success", "place_success", "task_success", "object_drop_rate"):
        assert 0.0 <= m[key] <= 1.0, key


def test_grasp_success_rate():
    g, p, d, t, s = _basic_batch()
    m = compute_manip_metrics(g, p, d, t, s)
    assert abs(m["grasp_success"] - 0.75) < 1e-5  # 3/4 grasped


def test_place_success_rate():
    g, p, d, t, s = _basic_batch()
    m = compute_manip_metrics(g, p, d, t, s)
    assert abs(m["place_success"] - 0.5) < 1e-5   # 2/4 placed


def test_task_success_requires_both():
    g, p, d, t, s = _basic_batch()
    m = compute_manip_metrics(g, p, d, t, s)
    assert abs(m["task_success"] - 0.5) < 1e-5    # eps 0 and 3: grasped+placed


def test_object_drop_rate():
    g, p, d, t, s = _basic_batch()
    m = compute_manip_metrics(g, p, d, t, s)
    assert abs(m["object_drop_rate"] - 0.25) < 1e-5  # 1/4 dropped


def test_mean_steps_to_success_only_over_successes():
    g, p, d, t, s = _basic_batch()
    m = compute_manip_metrics(g, p, d, t, s)
    # Successes at steps [50, 80] -> mean = 65
    assert abs(m["mean_steps_to_success"] - 65.0) < 1e-3


def test_mean_steps_nan_when_no_successes():
    g = torch.tensor([False, False])
    p = torch.tensor([False, False])
    d = torch.tensor([False, False])
    t = torch.tensor([False, False])
    s = torch.tensor([-1, -1], dtype=torch.long)
    m = compute_manip_metrics(g, p, d, t, s)
    import math
    assert math.isnan(m["mean_steps_to_success"])


def test_grasp_then_place_success_both_required():
    grasped = torch.tensor([True, True, False, True])
    placed = torch.tensor([True, False, True, True])
    result = grasp_then_place_success(grasped, placed)
    expected = torch.tensor([True, False, False, True])
    assert (result == expected).all()


def test_object_drop_rate_from_heights_threshold():
    grasp_h = torch.tensor([0.5, 0.5, 0.5])
    min_h = torch.tensor([0.44, 0.46, 0.30])  # drops: 0.06, 0.04, 0.20
    drops = object_drop_rate_from_heights(grasp_h, min_h, drop_threshold=0.05)
    assert bool(drops[0])   # 0.06 > 0.05 -> dropped
    assert not bool(drops[1])  # 0.04 <= 0.05 -> not dropped
    assert bool(drops[2])   # 0.20 > 0.05 -> dropped


def test_object_drop_no_drops_when_height_maintained():
    grasp_h = torch.tensor([0.5, 0.5])
    min_h = torch.tensor([0.50, 0.48])  # 0.00 and 0.02 drop -> both < threshold
    drops = object_drop_rate_from_heights(grasp_h, min_h, drop_threshold=0.05)
    assert not bool(drops[0])
    assert not bool(drops[1])
