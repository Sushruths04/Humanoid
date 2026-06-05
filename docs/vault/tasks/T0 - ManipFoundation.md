---
tags: [task, t0, manipulation, libero, franka, t-track]
---

# T0 — Manipulation Foundation (T-Track)

## Summary

The tabletop manipulation track baseline. Establishes a Franka/LIBERO env + manipulation eval harness + a behaviour-cloning baseline. Runs in parallel to the P-track humanoid nav work; shares the `programs/common/` spine.

**Status: CPU harness built. Env integration pending (requires LIBERO install in container).**

[T-track plan](../../TABLETOP_MANIPULATION_PLAN.md)

---

## What Was Built (CPU-verified)

- `programs/common/eval/manip_metrics.py` — pure tensor metrics (10 TDD tests, all green):
  - `compute_manip_metrics` — aggregates grasp/place/drop/task-success
  - `grasp_then_place_success` — boolean AND of grasp + place tensors
  - `object_drop_rate_from_heights` — detects drops from height trajectory
- `programs/t0_manip_foundation/evaluate_manip.py` — eval harness skeleton:
  - CLI: `--task`, `--checkpoint`, `--num-envs`, `--out`
  - Writes `docs/results/t0_manip.md`
  - `_build_env()` and `_load_policy()` stubs to wire in LIBERO
- `programs/t0_manip_foundation/README.md` — setup + DoD + metric table

---

## Metrics Defined

| Metric | Description |
|---|---|
| `grasp_success` | Robot lifted object > threshold above table |
| `place_success` | Object placed within `place_radius` of target |
| `task_success` | Grasped AND placed AND not dropped |
| `object_drop_rate` | Dropped object after grasping (height drop > threshold) |
| `mean_steps_to_success` | Steps to first task_success (over successes only; NaN if none) |

---

## Checkpoints

**CPT0.1 — Env up:** LIBERO installed; `python -c "from libero.libero import benchmark"` succeeds; one episode renders headless mp4.

**CPT0.2 — Eval harness:** `evaluate_manip.py --task libero_spatial --num-envs 64` prints metric dict + writes `docs/results/t0_manip.md`.

**CPT0.3 — BC baseline:** Behaviour-clone on small demo set (LeRobot dataset); BC policy beats random on ≥1 task.

---

## Remaining Steps

1. SSH to GPU machine, install LIBERO: `docker exec -it isaac-lab-base bash -c "pip install libero-benchmark"`
2. Wire `_build_env()` in `evaluate_manip.py` to create LIBERO env
3. Collect demos via scripted expert or LeRobot dataset → push to HF
4. Train BC baseline (ACT or Diffusion Policy via LeRobot)
5. Run eval, verify task_success > 0

---

## Key Parameters (planned)

| Parameter | Value |
|---|---|
| Env suite | `libero_spatial` (start), then `libero_object` |
| Num train envs | 64 |
| Grasp height threshold | 0.05 m |
| Place radius | 0.05 m |
| Drop threshold | 0.05 m |
| BC epochs | 100 (smoke baseline) |

---

## Related

- [[P0 - CommandNav]] — shares `programs/common/` eval spine
- [[P2 - World Model]] — T2 will reuse Dreamer-mini for manipulation
- [[Frozen Text Encoder for Language Tasks]] — T1 language-conditioned manip reuses it
