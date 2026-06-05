---
tags: [failure, eval, attributeerror, seqnav, evaluator]
---

# Eval Crash — Missing Buffer

## Symptom

```
AttributeError: 'ManagerBasedRLEnv' object has no attribute '_nav_target_ids'
```

Crash happens when running `evaluate.py` on the SeqNav task.

## Root Cause

`evaluate.py` was written for CommandNav and ObstacleNav. It directly accesses `base._nav_target_ids` (the single commanded target) and `base._nav_markers_xy` to compute which target to measure distance to.

SeqNav's first implementation used different buffers (`_seq_markers_xy`, `_seq_targets`, `_seq_phase`) — none of the `_nav_*` buffers existed on it.

Even after SeqNav was rebuilt on the CommandNav core (so `_nav_*` buffers DO exist), the standard evaluator still gives wrong metrics — it measures "did you reach *a* commanded target?" which makes no sense for a sequential task where the commanded target *changes* during the episode.

## The Fix

Wrote `programs/common/eval/evaluate_seq.py` — a sequence-aware evaluator:

1. **Snapshots** the first-episode layout: `_seq_targets` (ordered subgoal marker ids) + `_nav_markers_xy` (marker positions) after reset
2. **Tracks** the first timestep each subgoal was reached (or -1 if never) using a `reach_steps[N, num_subgoals]` tensor
3. **Computes** `sequence_eval_metrics(reach_steps, num_subgoals)` → full_sequence_success, ordering_accuracy, first_subgoal_rate

The metric `sequence_eval_metrics` was written TDD (test first, red-green confirmed) with this test case:
```python
reach = torch.tensor([
    [10, 25],  # both reached in order → full success
    [30, 12],  # both reached, WRONG order → fail
    [15, -1],  # only first reached
    [-1, -1],  # none reached
])
# expected: full_sequence=0.25, first_subgoal=0.75, ordering=0.5
```

## Lesson

> **Match the evaluator to the task.** If you build a new task type, build a new evaluator. Single-target metrics can't score multi-step sequential tasks.

Also: when a new type of task is added (sequential, hierarchical, multi-agent, etc.) the evaluation harness needs to be extended. Don't assume an existing evaluator generalizes.

## Related

- [[P1.4 - SeqNav]]
- [[SeqNav Stand-Still Local Optimum]]
- [[Evaluation Harness]]
- [programs/common/eval/evaluate_seq.py](../../programs/common/eval/evaluate_seq.py)
- [[00 - Failure Index]]
