---
tags: [project, results]
---

# Results Summary (every number + its source)

> Rule: a result only counts if there's an artifact on disk. Source files listed.

## Verified results
| Phase | Metric | Value | Source file |
|---|---|---|---|
| GR00T | train loss | 0.0855 | `results/gr00t_eval_smoke/summary.txt` |
| GR00T | action MSE / MAE | **25.87 / 3.01** | `results/gr00t_eval_smoke/summary.txt` |
| G1 baseline | throughput | ~14k steps/s (L40S) | `FINAL_RESULTS.md` (run-derived) |
| MarkerNav | mean reward | **28.9** @ 8192 envs, 114k steps/s | `FINAL_RESULTS.md` |
| **Robust** | **mean reward** | **22.82** | `logs/g1_robust/train.log` ✅ |
| **Robust** | **mean episode length** | **981.18 / 1000** | `logs/g1_robust/train.log` ✅ |
| Vision | pipeline reaches PPO, scaled to 2048 envs | qualitative | `state/30_vision_vla.done`, `STEP-30-vision-vla.md` |

## Saved artifacts on disk
- `checkpoints/g1_robust/model_latest.pt` — 3.27 MB (real RSL-RL policy) ✅
- `logs/g1_robust/train.log` — 2.5 MB telemetry ✅
- `results/gr00t_eval_smoke/`, `results/gr00t_demo/` — eval/demo summaries ✅
- GR00T checkpoint on HF: `mitvho09/GR00T-Humanoid`

## NOT present (gaps)
- ❌ No vision checkpoint / vision training log in the repo.
- ❌ No MarkerNav / baseline checkpoint or log saved locally (numbers come from `FINAL_RESULTS.md`, not a log file here) — treat as **reported, not independently re-verifiable** from this checkout.
- ❌ No success-rate / task-completion metric for the language or vision tasks.

## The honest headline
> **Strongest claim you can fully back with on-disk evidence:** a **G1 humanoid trained with PPO on rough terrain with ±25% joint domain randomization stays upright ~98% of the episode (981/1000) at mean reward 22.8.**

Everything else is either reported-only or verified-as-plumbing. See [[Open_Questions_and_Next_Steps]].

Related: [[Phase2.5_Sim2Real_Robust]] · [[Phase1_GR00T]]
