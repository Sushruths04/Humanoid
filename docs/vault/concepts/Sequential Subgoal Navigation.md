---
tags: [concepts, sequential, subgoal, navigation, curriculum]
---

# Sequential Subgoal Navigation

## What It Is

The robot must visit a sequence of markers **in order**: first go to marker A, then go to marker B. The order matters — reaching B before A doesn't count as a full sequence.

This is harder than single-target navigation because:
1. The policy must reach the first goal within the episode (otherwise it never learns the sequence)
2. The policy must stay aligned to the current goal (not the ultimate destination)
3. Success is measured as a full ordered sequence, not just "reached any marker"

---

## How It's Implemented (the proven architecture)

SeqNav is built **on top of CommandNav** — it reuses the exact same steering, observation, and reward. The only addition is a **phase-advance event**:

```python
def seq_advance(env, env_ids, num_markers=3, num_subgoals=2, reach_radius=0.5):
    """When current subgoal is reached, hop _nav_target_ids to the next one."""
    dist = (robot_xy - commanded_target_xy(env)).norm(dim=-1)
    new_phase, advanced = advance_subgoal(env._seq_phase, dist, reach_radius, num_subgoals)
    env._seq_phase = new_phase
    if advanced.any():
        next_ids = env._seq_targets[arange, env._seq_phase]
        env._nav_target_ids[advanced] = next_ids[advanced]
        env._nav_prev_xy[advanced] = robot_xy[advanced]  # reset progress baseline
```

**Registered BEFORE the steer event** so the steer uses the freshly-advanced target in the same step.

To the policy, this looks like **CommandNav with a target that hops**. The same one-hot + relative-vector observation just points at whichever subgoal is currently active.

---

## Final Parameters (after the bootstrap-fix debugging)

| Parameter | Value | Why |
|---|---|---|
| NUM_MARKERS | 3 | 3 possible marker positions |
| NUM_SUBGOALS | 2 | visit 2 of 3 in order |
| RADIUS_RANGE | **(1.0, 2.5) m** | close targets to enable bootstrap |
| REACH_RADIUS | 0.5 m | within half a meter = reached |
| progress_scale | 2.0 | stronger nav signal vs. locomotion |

> **Why 1.0–2.5 m instead of the 2.0–5.0 m used for other tasks?** The bootstrap fix. See [[SeqNav Stand-Still Local Optimum]] for the full story.

---

## The Bootstrap Problem (the hardest part)

With 2 sequential subgoals 2–5 m apart and a typical episode length, the probability of *randomly* reaching the first subgoal early enough to start learning is very low. The policy finds it easier to stand still and track a near-zero velocity command for locomotion reward.

Closer targets (1–2.5 m) mean the policy reaches subgoal-0 within the first few hundred training iterations, triggering the reach bonus, which creates a strong gradient to learn the full sequence.

**Curriculum principle:** always make the FIRST success easy. You can increase difficulty later once the basic behavior is learned.

---

## Metrics for Sequential Navigation

Single-target success rate (from `evaluate.py`) is wrong for sequential tasks — it measures reaching *a* target, not whether you visited them *in order*. Use `evaluate_seq.py` with `sequence_eval_metrics`:

| Metric | What it measures |
|---|---|
| `full_sequence_success` | fraction of episodes where ALL subgoals reached AND in correct order |
| `ordering_accuracy` | of episodes reaching all subgoals, fraction in the correct order |
| `first_subgoal_rate` | fraction reaching subgoal-0 (a weak lower bound) |

Results (SeqNav final): **80.9% full-sequence, 94.5% ordering, 97.7% first-subgoal**.

---

## Related

- [[P1.4 - SeqNav]]
- [[Command-Conditioned Navigation]]
- [[SeqNav Stand-Still Local Optimum]]
- [[Evaluation Harness]]
- [programs/common/sequence.py](../../programs/common/sequence.py)
- [programs/common/eval/evaluate_seq.py](../../programs/common/eval/evaluate_seq.py)
