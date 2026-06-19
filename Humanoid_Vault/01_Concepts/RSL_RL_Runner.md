---
tags: [concept, rl, trainer]
---

# RSL-RL Runner

## What it is
**RSL-RL** is the lightweight, GPU-vectorized **on-policy ([[PPO_for_Locomotion|PPO]])** trainer Isaac Lab uses for legged robots. You configure it via `RslRlOnPolicyRunnerCfg` (and here `RslRlCNNModelCfg` for the vision policy).

## What the config controls (from this project)
- **`obs_groups`** — which observation groups feed actor vs critic. The vision task uses `{"actor": ["policy", "images"], "critic": ["policy", "images"]}` — i.e. proprioception **and** pixels to both.
- **`actor` / `critic` models** — MLP (`hidden_dims=[256,256]`, ELU) for state; **CNN** (Nature-CNN) when pixels are present.
- **`share_cnn_encoders = True`** — actor & critic share one visual encoder (cheaper, common in pixel PPO).
- **`num_steps_per_env`, `max_iterations`, `save_interval`** — rollout length, training length, checkpoint cadence (50).
- **`experiment_name`** — names the log/checkpoint dir (`g1_vla_vision_cnn`, `g1_robust`, …).

## Distribution
Gaussian policy with learned/`log`-parameterized std (`init_std=1.0`) — continuous joint-position/velocity actions.

## Why it matters
RSL-RL is *the* reason 8k-env PPO runs fast. Understanding its config is understanding how every result in this project was produced. The `.pt` checkpoints (e.g. `g1_robust/model_latest.pt`, 3.27 MB) are RSL-RL policy/optimizer states.

Related: [[PPO_for_Locomotion]] · [[Isaac_Lab_and_Isaac_Sim]] · [[Architecture_Task_Hierarchy]]
