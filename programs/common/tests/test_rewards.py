import torch
from programs.common.rewards import commanded_target_reward


def test_moving_toward_commanded_target_gives_positive_reward():
    # 1 env, 2 markers: marker0 at (2,0), marker1 at (-2,0). Commanded = marker0.
    markers_xy = torch.tensor([[[2.0, 0.0], [-2.0, 0.0]]])  # (N=1, M=2, 2)
    target_id = torch.tensor([0], dtype=torch.long)          # commanded marker 0
    prev_robot_xy = torch.tensor([[0.0, 0.0]])               # started at origin
    robot_xy = torch.tensor([[1.0, 0.0]])                    # moved 1m toward marker0

    reward = commanded_target_reward(
        robot_xy, prev_robot_xy, markers_xy, target_id,
        reach_radius=0.5, progress_scale=1.0, wrong_penalty_scale=1.0, reach_bonus=10.0,
    )

    assert reward.shape == (1,)
    assert reward.item() > 0.0


def test_approaching_wrong_marker_is_penalized():
    # target0 far away (10m); wrong marker1 near. Isolate the penalty with
    # progress_scale=0 so only wrong-marker approach can move the reward.
    markers_xy = torch.tensor([[[0.0, 10.0], [2.0, 0.0]]])
    target_id = torch.tensor([0], dtype=torch.long)
    prev_robot_xy = torch.tensor([[0.0, 0.0]])
    robot_xy = torch.tensor([[1.0, 0.0]])  # moved 1m toward wrong marker1

    reward = commanded_target_reward(
        robot_xy, prev_robot_xy, markers_xy, target_id,
        reach_radius=0.5, progress_scale=0.0, wrong_penalty_scale=1.0, reach_bonus=0.0,
    )

    assert reward.item() < 0.0


def test_reaching_commanded_target_adds_bonus():
    # Robot sits on the commanded marker; isolate the bonus.
    markers_xy = torch.tensor([[[0.0, 0.0], [5.0, 5.0]]])
    target_id = torch.tensor([0], dtype=torch.long)
    prev_robot_xy = torch.tensor([[0.0, 0.0]])
    robot_xy = torch.tensor([[0.0, 0.0]])  # within reach_radius of target0

    reward = commanded_target_reward(
        robot_xy, prev_robot_xy, markers_xy, target_id,
        reach_radius=0.5, progress_scale=0.0, wrong_penalty_scale=0.0, reach_bonus=10.0,
    )

    assert reward.item() == 10.0


def test_reward_depends_on_which_target_is_commanded():
    # DoD probe: identical motion, different command -> different reward.
    markers_xy = torch.tensor([[[2.0, 0.0], [-2.0, 0.0]]])
    prev_robot_xy = torch.tensor([[0.0, 0.0]])
    robot_xy = torch.tensor([[1.0, 0.0]])  # moved toward marker0, away from marker1

    kwargs = dict(reach_radius=0.5, progress_scale=1.0, wrong_penalty_scale=1.0, reach_bonus=10.0)
    r_cmd0 = commanded_target_reward(robot_xy, prev_robot_xy, markers_xy, torch.tensor([0]), **kwargs)
    r_cmd1 = commanded_target_reward(robot_xy, prev_robot_xy, markers_xy, torch.tensor([1]), **kwargs)

    assert r_cmd0.item() > 0.0    # commanded marker0: rewarded
    assert r_cmd1.item() < 0.0    # commanded marker1: penalized
    assert r_cmd0.item() > r_cmd1.item()


def test_batched_envs_are_independent():
    # Two envs with different commands evaluated together.
    markers_xy = torch.tensor([
        [[2.0, 0.0], [-2.0, 0.0]],
        [[2.0, 0.0], [-2.0, 0.0]],
    ])
    prev_robot_xy = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    robot_xy = torch.tensor([[1.0, 0.0], [1.0, 0.0]])  # both moved toward marker0
    target_id = torch.tensor([0, 1], dtype=torch.long)  # env0 commands m0, env1 commands m1

    reward = commanded_target_reward(
        robot_xy, prev_robot_xy, markers_xy, target_id,
        reach_radius=0.5, progress_scale=1.0, wrong_penalty_scale=1.0, reach_bonus=10.0,
    )

    assert reward.shape == (2,)
    assert reward[0].item() > 0.0   # env0 moved toward its commanded marker
    assert reward[1].item() < 0.0   # env1 moved away from its commanded marker


from programs.common.rewards import collision_penalty


def test_collision_penalty_zero_when_far():
    p = collision_penalty(
        robot_xy=torch.tensor([[0.0, 0.0]]),
        obstacles_xy=torch.tensor([[[5.0, 5.0]]]),
        collision_radius=0.4, penalty_scale=1.0,
    )
    assert p.shape == (1,)
    assert abs(float(p[0])) < 1e-6


def test_collision_penalty_max_at_contact():
    p = collision_penalty(
        robot_xy=torch.tensor([[0.0, 0.0]]),
        obstacles_xy=torch.tensor([[[0.0, 0.0]]]),
        collision_radius=0.4, penalty_scale=2.0,
    )
    assert abs(float(p[0]) - (-2.0)) < 1e-6   # full intrusion -> -penalty_scale


def test_collision_penalty_batched_and_multi_obstacle():
    p = collision_penalty(
        robot_xy=torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        obstacles_xy=torch.tensor([
            [[0.2, 0.0], [5.0, 5.0]],   # one close, one far
            [[5.0, 5.0], [5.0, 5.0]],   # both far
        ]),
        collision_radius=0.4, penalty_scale=1.0,
    )
    assert p.shape == (2,)
    assert float(p[0]) < 0.0     # env0 has a close obstacle
    assert abs(float(p[1])) < 1e-6   # env1 clear
