---
tags: [reference, parameters, hyperparameters, cheat-sheet]
---

# All Parameters Cheat-Sheet

## Training (all tasks)

| Parameter | Value | Notes |
|---|---|---|
| num_envs | 4096 | measured peak VRAM: ~4.6 GB on L4 |
| max_iterations | 500 | ~20 min on L4 |
| iter_time | ~2.4 s | measured |
| checkpoint frequency | 50 iters | model_0.pt, model_50.pt, ..., model_499.pt |
| eval envs | 256 | for evaluation rollout |

## Navigation Task Parameters

| Task | NUM_MARKERS | RADIUS_RANGE | REACH_RADIUS | speed | yaw_gain | max_yaw_rate |
|---|---|---|---|---|---|---|
| CommandNav | 2 | (2.0, 5.0) m | 0.5 m | 1.0 | 0.5 | 1.0 |
| LangNav | 3 | (2.0, 5.0) m | 0.5 m | 1.0 | 0.5 | 1.0 |
| ObstacleNav | 3 | (2.0, 5.0) m | 0.5 m | 1.0 | 0.5 | 1.0 |
| SeqNav | 3 | **(1.0, 2.5) m** | 0.5 m | 1.0 | 0.5 | 1.0 |

## Reward Parameters

| Task | weight | progress_scale | wrong_penalty_scale | reach_bonus |
|---|---|---|---|---|
| CommandNav | 1.0 | 1.0 | 1.0 | 10.0 |
| LangNav | 1.0 | 1.0 | 1.0 | 10.0 |
| ObstacleNav (nav) | 1.0 | 1.0 | 1.0 | 10.0 |
| ObstacleNav (collision) | 1.0 | — | — | — |
| SeqNav | 1.0 | **2.0** | **1.0** | 10.0 |

## Obstacle Parameters (ObstacleNav)

| Parameter | Value |
|---|---|
| NUM_OBSTACLES | 3 |
| avoid_radius (potential field) | 1.5 m |
| avoid_gain | 2.0 |
| collision_radius | 0.4 m |
| penalty_scale | 1.0 |

## Stability Reward (P0 follow-up)

| Parameter | Value | Notes |
|---|---|---|
| upright_reward weight | **0.5** (v1) / **2.0** (v2) | env var `COMMANDNAV_UPRIGHT_WEIGHT` |
| formula | `(1 - 2*(x²+y²)).clamp(0)` | from quaternion [w,x,y,z] |
| max value | 1.0 (perfectly upright) | 0.0 when horizontal, clipped 0 when inverted |

## World Model (Dreamer-mini)

| Parameter | Value |
|---|---|
| deter (GRU hidden) | 64 |
| stoch (latent dim) | 16 |
| hidden (MLP) | 64 |
| Toy training loss | 2.7 → 0.11 |
| Isaac WM deter | 128 |
| Isaac WM stoch | 32 |
| Isaac WM training steps | 2000 |

## VRAM Reference

| Task type | Required VRAM | Recommended GPU |
|---|---|---|
| State-based nav RL (P0–P1) | ~5 GB | T4/16GB ✅ |
| Vision RL (P3, cameras) | 24–40 GB | L4/L40S |
| Cosmos Predict inference | 24–40 GB | L40S |
| Cosmos Predict post-train | 80 GB | A100-80 |

## World Model — Isaac Rollout Results

| Metric | Value |
|---|---|
| Rollout episodes | 200 (CommandNav, P0-stable policy) |
| obs_dim / act_dim | 4 / 37 |
| Training steps | 2000 |
| Initial / final loss | 0.7625 / 0.0109 |
| Imagined reward | 0.133 (finite ✅) |

## T0 Manipulation (LIBERO)

| Parameter | Value |
|---|---|
| Env | `libero_spatial:0` (pick bowl → place on plate) |
| Policy | MLPBCPolicy: obs=12, act=7, hidden=256 |
| Obs format | joint_pos(7) + eef_pos(3) + gripper_qpos(2) |
| Training demos | 50 × ~100 steps = 5018 transitions |
| BC epochs | 200 |
| MUJOCO_GL | egl (server, no display) |
| Python env | conda `libero_env` (Python 3.9, torch 2.7+cu118) |

## T1 GR00T N1.7 (LIBERO)

| Parameter | Value |
|---|---|
| Base model | `nvidia/GR00T-N1.7-3B` (3B params, VLA) |
| LIBERO checkpoint | `nvidia/GR00T-N1.7-LIBERO/libero_spatial` |
| Embodiment tag | `LIBERO_PANDA` |
| Obs state | eef_xyz(3) + euler_rpy(3) + gripper_qpos(2) = 8 dims |
| Obs video | `image` + `wrist_image` — (B=1,T=1,H,W,3) uint8 |
| Action chunk | 8 steps per inference call |
| Eval VRAM | ≥16 GB (T4 15 GB may OOM) |
| Fine-tune VRAM | ≥40 GB (L40S / A100) |
| Fine-tune dataset | `IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot` |
| Expected task_success | ~97.7% (NVIDIA paper, 10 tasks) |

## Results (all measured)

| Checkpoint | Metric | Value |
|---|---|---|
| P0 CommandNav | success_rate | **94.5%** |
| P0 CommandNav | per-command | [95.8%, 93.4%] |
| P0-stable CommandNav | success_rate | **92.6%** |
| P0-stable CommandNav | fall_rate | **7.8%** ✅ (was 28.1%) |
| P1.2 LangNav | per-command | **98.8%** |
| P1.3 ObstacleNav | success_rate | **85.9%** |
| P1.3 ObstacleNav | per-command | [83.7%, 88.0%] |
| P1.4 SeqNav | full_sequence_success | **80.9%** |
| P1.4 SeqNav | ordering_accuracy | **94.5%** |
| P1.4 SeqNav | first_subgoal_rate | 97.7% |
| T0 BC (libero_spatial:0) | task_success | **50.0%** |
| T0 BC (libero_spatial:0) | grasp_success | 70.0% |
| T1 GR00T N1.7 (libero_spatial, 10 tasks) | mean_task_success | **97.0%** |

## Related

- [[00 - START HERE]]
- [[PPO with RSL-RL]]
- [[Reward Shaping & Progress Rewards]]
- [docs/GPU_VRAM_REQUIREMENTS.md](../../docs/GPU_VRAM_REQUIREMENTS.md)
