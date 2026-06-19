---
tags: [implementation, scripts, live]
---

# Scripts (LIVE — update when code changes)

> Per [[DOC_PROTOCOL]]: every new/changed script gets a row here in the same change. Empty now — fill as you build (`PLAN.md` §13–§14).

| Script | Purpose | Key args / env-vars | Inputs | Outputs | Status |
|---|---|---|---|---|---|
| `scripts/00_smoke.sh` | build env + reach PPO | `NUM_ENVS=16 MAX_ITERS=2` | configs/smoke.yaml | `ckpt_00_smoke`, log | ⬜ TODO |
| `scripts/10_train_stage1.sh` | Stage I discovery | `V_PULL`, stage1.yaml | — | `ckpt_2x_stage1*` | ⬜ TODO |
| `scripts/20_train_stage2.sh` | Stage II deployable + AMP | stage2.yaml | Stage-I traj | `ckpt_30_stage2_deploy` | ⬜ TODO |
| `scripts/30_eval.sh` | eval a checkpoint | `CKPT` | checkpoint | `results/eval_*.json` | ⬜ TODO |
| `scripts/31_eval_speed_sweep.sh` | success vs pull speed | speeds 20/25/30/35 | checkpoint | `results/speed_sweep.json` | ⬜ TODO |
| `scripts/40_record_video.sh` | mp4 rollout (rendering kit) | `--enable_cameras` | checkpoint | mp4 | ⬜ TODO |
| `scripts/99_collect_results.sh` | aggregate JSON → tables | — | results/*.json | [[Results_Live]] | ⬜ TODO |

## Core modules (`src/`)
| Module | Responsibility | Status |
|---|---|---|
| `tasks/wakeboard_start_cfg.py` | env: G1+board+rope, obs, terminations | ⬜ TODO |
| `rewards/wakeboard_rewards.py` | all reward terms ([[Reward_Design]]) | ⬜ TODO |
| `rope_model.py` | pull models A/B | ⬜ TODO |
| `board.py` | board asset + foot binding | ⬜ TODO |
| `curriculum.py` | pull-speed + stage curricula | ⬜ TODO |
| `amp/` | discriminator + reference-motion loader | ⬜ TODO |
