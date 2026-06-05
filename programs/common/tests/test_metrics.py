import torch
from programs.common.eval.metrics import compute_episode_metrics


def test_compute_episode_metrics_basic():
    reached = torch.tensor([True, True, False, False])
    fell = torch.tensor([False, False, True, False])
    final_distance = torch.tensor([0.1, 0.2, 3.0, 1.0])
    episode_length = torch.tensor([100, 120, 80, 200])

    m = compute_episode_metrics(reached, fell, final_distance, episode_length)

    assert m["num_episodes"] == 4
    assert m["success_rate"] == 0.5
    assert m["fall_rate"] == 0.25
    assert abs(m["mean_final_distance"] - 1.075) < 1e-6
    assert abs(m["mean_episode_length"] - 125.0) < 1e-6


from programs.common.eval.metrics import success_rate_by_command


def test_success_rate_by_command():
    reached = torch.tensor([True, False, True, True])
    command_ids = torch.tensor([0, 0, 1, 1])
    rates = success_rate_by_command(reached, command_ids, num_commands=2)
    assert abs(float(rates[0]) - 0.5) < 1e-6   # cmd0: 1/2
    assert abs(float(rates[1]) - 1.0) < 1e-6   # cmd1: 2/2


def test_write_results_markdown(tmp_path):
    from programs.common.eval.report import write_results_markdown
    metrics = {
        "num_episodes": 4, "success_rate": 0.5, "fall_rate": 0.25,
        "mean_final_distance": 1.075, "mean_episode_length": 125.0,
    }
    out = tmp_path / "p0_baseline.md"
    write_results_markdown(metrics, str(out), title="P0 Baseline")
    text = out.read_text()
    assert "P0 Baseline" in text
    assert "success_rate" in text
    assert "0.5" in text


def test_sequence_eval_metrics_full_and_order():
    import torch
    from programs.common.eval.metrics import sequence_eval_metrics

    # 4 episodes, 2 subgoals; -1 = subgoal never reached, else first-reach step.
    reach = torch.tensor([
        [10, 25],   # both reached, correct order -> full success
        [30, 12],   # both reached, WRONG order  -> not full, ordering fail
        [15, -1],   # only first reached
        [-1, -1],   # none reached
    ])
    m = sequence_eval_metrics(reach, num_subgoals=2)
    assert m["num_episodes"] == 4
    assert abs(m["full_sequence_success"] - 0.25) < 1e-6   # only ep0
    assert abs(m["first_subgoal_rate"] - 0.75) < 1e-6      # ep0,1,2
    assert abs(m["ordering_accuracy"] - 0.5) < 1e-6        # of {ep0,ep1} reaching both, ep0 ordered
