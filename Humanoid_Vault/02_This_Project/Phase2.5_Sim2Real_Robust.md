---
tags: [project, phase, sim2real, robustness]
---

# Phase 2.5 — Sim-to-Real Robustness

## Goal
Make the G1 policy survive on **uneven ground** and tolerate **physics it never saw** — the core of [[Domain_Randomization_and_Sim2Real|sim-to-real]].

## What was done (source: `g1_language_pickplace_cfg.py::LanguageConditionedG1RobustTaskCfg`)
- **Base env switched** from flat (`G1FlatEnvCfg`) to **rough terrain** (`G1RoughEnvCfg`) — procedurally generated uneven ground.
- **Domain randomization** on actuator dynamics:
  ```python
  randomize_joint_parameters.stiffness_distribution_params = (0.75, 1.25)
  randomize_joint_parameters.damping_distribution_params   = (0.75, 1.25)
  ```
  i.e. joint **stiffness** and **damping** each randomized ±25% per episode.
- Keeps the language + marker mixins. Task id `Humanoid-G1-Robust-VLA-v0`.

## Real results (source: `thesis/logs/g1_robust/train.log`)
- Trained to at least **iteration 1350/5000** (log captured ~2700 iteration entries), ~2.0s/iter.
- Late-training **mean reward ≈ 22.8**, **mean episode length ≈ 981 / 1000**.
- Checkpoint: `thesis/checkpoints/g1_robust/model_latest.pt` (3.27 MB — a real RSL-RL MLP policy).

### How to read "episode length 981/1000"
Episodes terminate early on a fall. **981/1000** means the robot **almost never falls** across the full episode, even on rough terrain with randomized joints → strong evidence the robustness training worked. This is the project's **most defensible single result**.

## Why ±25% randomization?
Narrow enough that the task stays learnable, wide enough that the policy can't overfit one exact stiffness/damping → it learns a *robust* gait. This is the standard sim-to-real recipe (randomize what you're uncertain about on real hardware: friction, mass, actuator gains).

Related: [[Domain_Randomization_and_Sim2Real]] · [[Results_Summary]] · [[PPO_for_Locomotion]]
