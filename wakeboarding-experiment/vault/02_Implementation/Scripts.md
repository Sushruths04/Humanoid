---
tags: [implementation, scripts, live]
---

# Scripts (LIVE — update when code changes)

> Per [[DOC_PROTOCOL]]: every new/changed script gets a row here in the same change.
> **Legend:** 🟩 built + CPU-verified · 🟨 built, untested on GPU · ⬜ TODO.
> **All code below was written 2026-06-19; pure-PyTorch parts CPU-tested. Nothing run on GPU yet.**

| Script | Purpose | Key args / env-vars | Outputs | Status |
|---|---|---|---|---|
| `scripts/00_smoke.sh` | build env + reach PPO | `NUM_ENVS MAX_ITERS` | `runs/wakeboard_smoke/` | 🟨 untested-GPU |
| `scripts/10_train_stage1.sh` | Stage I discovery + curriculum | `NUM_ENVS MAX_ITERS` | `ckpt_2x_stage1*` | 🟨 untested-GPU |
| `scripts/20_train_stage2.sh` | Stage II deployable + AMP + DR | `STAGE1_CKPT` | `ckpt_30_stage2_deploy` | 🟨 untested-GPU |
| `scripts/30_eval.sh` | eval a checkpoint | `CKPT V_PULL_KMH` | `results/eval_*.json` | 🟨 untested-GPU |
| `scripts/31_eval_speed_sweep.sh` | success vs pull speed (Table A) | `CKPT` | `results/sweep_*.json` | 🟨 untested-GPU |
| `scripts/40_record_video.sh` | mp4 rollout (rendering kit) | `CKPT` | mp4 | 🟨 untested-GPU |
| `scripts/99_collect_results.sh` | aggregate JSON → tables | — | [[Results_Live]] | 🟩 CPU-ok |
| `train.py` | RSL-RL train loop + curriculum hook | `--config` | checkpoints | 🟨 untested-GPU |
| `eval.py` | rollout + metrics JSON | `--checkpoint` | results JSON | 🟨 untested-GPU |
| `modal_app.py` | **Modal** serverless runner (L40S) | `--action` | Volume `/ckpts` | 🟨 untested-GPU |

## Core modules (`src/`)
| Module | Responsibility | Status |
|---|---|---|
| `rope_model.py` | pull models A/B (force cap, anchor) | 🟩 **CPU-verified** |
| `curriculum.py` | pull-speed 10→30 auto-curriculum | 🟩 **CPU-verified** |
| `amp/discriminator.py` + `reference_motion.py` | AMP style reward + keyframe refs | 🟩 **CPU-verified** |
| `rewards/wakeboard_rewards.py` | all reward terms ([[Reward_Design]]) | 🟨 needs GPU env buffers |
| `board.py` | board asset + foot binding | 🟨 untested-GPU (VERIFY G1 link names) |
| `tasks/wakeboard_start_cfg.py` | env: G1+board+rope, obs, terminations | 🟨 untested-GPU (most `# VERIFY` markers here) |

## What "untested-GPU" means
The framework-agnostic logic (rope physics, curriculum, AMP) is CPU-verified. The Isaac-Lab-coupled code (env cfg, board binding, reward state accessors) is written against the documented manager-based API with `# VERIFY` markers, and needs a **GPU smoke pass** (`00_smoke.sh`) to fix version-specific API calls (G1 joint/link names, external-force API, `projected_gravity_b`). Budget ~½–1 day of GPU debugging for first green smoke.
