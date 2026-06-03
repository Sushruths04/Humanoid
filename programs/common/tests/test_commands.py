import torch
from programs.common.commands import sample_target_ids


def test_sampled_target_ids_have_correct_shape_and_range():
    ids = sample_target_ids(num_envs=64, num_markers=3)
    assert ids.shape == (64,)
    assert ids.dtype == torch.long
    assert int(ids.min()) >= 0
    assert int(ids.max()) <= 2


def test_sampling_is_reproducible_with_a_generator():
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    a = sample_target_ids(num_envs=32, num_markers=4, generator=g1)
    b = sample_target_ids(num_envs=32, num_markers=4, generator=g2)
    assert torch.equal(a, b)


def test_all_marker_ids_appear_across_many_envs():
    ids = sample_target_ids(num_envs=2000, num_markers=3, generator=torch.Generator().manual_seed(1))
    present = set(int(x) for x in ids.unique())
    assert present == {0, 1, 2}


from programs.common.commands import sample_marker_positions


def test_marker_positions_shape_and_radius_bounds():
    pos = sample_marker_positions(
        num_envs=128, num_markers=3, radius_range=(2.0, 5.0),
        generator=torch.Generator().manual_seed(0),
    )
    assert pos.shape == (128, 3, 2)
    radii = torch.linalg.norm(pos, dim=-1)
    assert float(radii.min()) >= 2.0 - 1e-5
    assert float(radii.max()) <= 5.0 + 1e-5


def test_marker_positions_reproducible_with_generator():
    a = sample_marker_positions(64, 4, generator=torch.Generator().manual_seed(7))
    b = sample_marker_positions(64, 4, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


from programs.common.commands import target_id_to_onehot


def test_target_id_to_onehot_is_valid_one_hot():
    target_id = torch.tensor([0, 2, 1], dtype=torch.long)
    oh = target_id_to_onehot(target_id, num_markers=3)
    assert oh.shape == (3, 3)
    assert torch.allclose(oh.sum(dim=-1), torch.ones(3))
    assert torch.equal(oh.argmax(dim=-1), target_id)


from programs.common.commands import velocity_command_to_target


def test_velocity_command_straight_ahead():
    cmd = velocity_command_to_target(
        robot_xy=torch.tensor([[0.0, 0.0]]),
        robot_yaw=torch.tensor([0.0]),
        target_xy=torch.tensor([[5.0, 0.0]]),
        speed=1.0, yaw_gain=1.0, max_yaw_rate=2.0,
    )
    assert cmd.shape == (1, 3)
    assert abs(float(cmd[0, 0]) - 1.0) < 1e-5   # full forward
    assert abs(float(cmd[0, 2])) < 1e-5         # no turn needed


def test_velocity_command_target_to_left_turns_left():
    cmd = velocity_command_to_target(
        robot_xy=torch.tensor([[0.0, 0.0]]),
        robot_yaw=torch.tensor([0.0]),
        target_xy=torch.tensor([[0.0, 5.0]]),
        speed=1.0, yaw_gain=1.0, max_yaw_rate=2.0,
    )
    assert abs(float(cmd[0, 0])) < 1e-5         # not facing target -> no forward
    assert float(cmd[0, 2]) > 0.5              # positive yaw rate (turn left)


def test_velocity_command_target_behind_no_forward_max_turn():
    cmd = velocity_command_to_target(
        robot_xy=torch.tensor([[0.0, 0.0]]),
        robot_yaw=torch.tensor([0.0]),
        target_xy=torch.tensor([[-5.0, 0.0]]),
        speed=1.0, yaw_gain=1.0, max_yaw_rate=2.0,
    )
    assert abs(float(cmd[0, 0])) < 1e-5         # facing away -> no forward
    assert abs(float(cmd[0, 2])) > 1.9         # near max yaw rate to turn around


from programs.common.commands import velocity_command_to_target_avoiding


def test_avoiding_steers_straight_when_no_obstacles_near():
    cmd = velocity_command_to_target_avoiding(
        robot_xy=torch.tensor([[0.0, 0.0]]),
        robot_yaw=torch.tensor([0.0]),
        target_xy=torch.tensor([[5.0, 0.0]]),
        obstacles_xy=torch.tensor([[[0.0, 9.0]]]),  # far away
        speed=1.0, yaw_gain=1.0, max_yaw_rate=2.0,
        avoid_radius=1.5, avoid_gain=2.0,
    )
    assert cmd.shape == (1, 3)
    assert abs(float(cmd[0, 0]) - 1.0) < 1e-4
    assert abs(float(cmd[0, 2])) < 1e-4


def test_avoiding_turns_away_from_obstacle_directly_ahead():
    cmd = velocity_command_to_target_avoiding(
        robot_xy=torch.tensor([[0.0, 0.0]]),
        robot_yaw=torch.tensor([0.0]),
        target_xy=torch.tensor([[5.0, 0.0]]),
        obstacles_xy=torch.tensor([[[1.0, 0.05]]]),  # close, just left of center
        speed=1.0, yaw_gain=1.0, max_yaw_rate=2.0,
        avoid_radius=1.5, avoid_gain=3.0,
    )
    assert abs(float(cmd[0, 2])) > 0.1
