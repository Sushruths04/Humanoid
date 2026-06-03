import torch
from programs.common.sequence import advance_subgoal


def test_advance_when_reached_nonfinal():
    phase = torch.tensor([0], dtype=torch.long)
    dist = torch.tensor([0.3])
    new_phase, advanced = advance_subgoal(phase, dist, reach_radius=0.5, num_subgoals=2)
    assert int(new_phase[0]) == 1
    assert bool(advanced[0]) is True


def test_no_advance_on_final_subgoal():
    phase = torch.tensor([1], dtype=torch.long)   # final of 2
    dist = torch.tensor([0.1])
    new_phase, advanced = advance_subgoal(phase, dist, reach_radius=0.5, num_subgoals=2)
    assert int(new_phase[0]) == 1
    assert bool(advanced[0]) is False


def test_no_advance_when_far():
    phase = torch.tensor([0], dtype=torch.long)
    dist = torch.tensor([1.0])
    new_phase, advanced = advance_subgoal(phase, dist, reach_radius=0.5, num_subgoals=2)
    assert int(new_phase[0]) == 0
    assert bool(advanced[0]) is False


def test_batched():
    phase = torch.tensor([0, 1, 0], dtype=torch.long)
    dist = torch.tensor([0.2, 0.2, 1.0])
    new_phase, advanced = advance_subgoal(phase, dist, reach_radius=0.5, num_subgoals=3)
    assert [int(x) for x in new_phase] == [1, 2, 0]
    assert [bool(x) for x in advanced] == [True, True, False]


from programs.common.sequence import sample_subgoal_sequence


def test_subgoal_sequence_shape_range_distinct():
    seq = sample_subgoal_sequence(64, num_markers=3, num_subgoals=2, generator=torch.Generator().manual_seed(0))
    assert seq.shape == (64, 2)
    assert int(seq.min()) >= 0 and int(seq.max()) <= 2
    for row in seq:
        assert len(set(int(x) for x in row)) == 2   # distinct within a sequence
