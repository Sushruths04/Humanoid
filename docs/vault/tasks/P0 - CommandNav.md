---
tags: [task, p0, commandnav, result, baseline]
---

# P0 — CommandNav (Humanoid-G1-CommandNav-v0)

## Summary

The baseline task that fixed the [[Decorative Navigation Defect]] in the original repo. The robot must walk to a **commanded colored marker** (red, blue) in a randomized arena.

**Result: 94.5% success** (per-command: red 95.8%, blue 93.4%). Fall rate 28% → **stability fix in progress** (see below).

[Full result doc](../../results/p0_baseline.md)

## Stability Fix (P0 follow-up — training now)

Fall rate was 28.1% — robot often topples after reaching the target.

**Fix:** added `upright_reward` (weight=0.5) as a new `RewTerm`:
- Pure function in `programs/common/rewards.py`: `1 - 2*(x² + y²)` from quaternion, clipped at 0
- Returns 1.0 when perfectly upright, 0.0 when horizontal
- TDD: 4 tests, all green

Retrained with 4096 envs, 500 iters. Results pending.

---

## What Was Built

- `programs/common/commands.py` — `velocity_command_to_target`, `sample_target_ids`, `sample_marker_positions`, `target_id_to_onehot`
- `programs/common/rewards.py` — `commanded_target_reward`, `collision_penalty`
- `programs/common/eval/` — full eval harness (`metrics.py`, `report.py`, `evaluate.py`)
- `my-humanoid-project/my_humanoid_project/tasks/g1_command_nav_cfg.py` — the task config
- TDD: 17 CPU unit tests, all pass before any GPU run

---

## Key Parameters

| Parameter | Value |
|---|---|
| NUM_MARKERS | 2 (red, blue) |
| RADIUS_RANGE | (2.0, 5.0) m |
| REACH_RADIUS | 0.5 m |
| steer speed | 1.0 m/s |
| steer yaw_gain | 0.5 |
| progress_scale | 1.0 |
| wrong_penalty_scale | 1.0 |
| reach_bonus | 10.0 |
| reward weight | 1.0 |
| num_envs (train) | 4096 |
| max_iterations | 500 |

---

## Architecture

See [[Command-Conditioned Navigation]] for the full pattern.

Reset event: `reset_nav_command` samples a random target id + marker positions.  
Step event: `steer_velocity_to_target` writes the steering command every step.  
Obs: one-hot target id (2-dim) + relative vector to target (2-dim) = 4-dim nav obs.  
Reward: `commanded_target_reward` (progress + wrong-marker penalty + reach bonus).

---

## How to Retrain from Scratch

```bash
bash programs/scripts/train_eval_nav.sh Humanoid-G1-CommandNav-v0 4096 500 256
```

Checkpoint: `huggingface.co/mitvho09/humanoid-g1-nav` → `checkpoints/g1_commandnav/model_499.pt`

---

## Related

- [[Command-Conditioned Navigation]]
- [[Velocity-Command Steering Law]]
- [[Reward Shaping & Progress Rewards]]
- [[P1.3 - ObstacleNav]] — layers obstacles on top of this
- [[Decorative Navigation Defect]] — the failure this fixes
