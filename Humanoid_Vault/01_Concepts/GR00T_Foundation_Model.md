---
tags: [concept, foundation-model, gr00t]
---

# GR00T Foundation Model

## What it is
**NVIDIA GR00T** (here `nvidia/GR00T-N1.7-3B`) is a **humanoid foundation model** — a large (~3B-param) policy pre-trained on broad humanoid manipulation/motion data, designed to be **fine-tuned** to a specific robot/task with relatively little data. It's a [[Vision_Language_Action_Models|VLA]]-style model.

## How it was used (Phase 1)
- Fine-tuned on a small `cube_to_bowl_5` demo set (5 trajectories) for 10,000 steps → `checkpoint-10000`.
- Evaluated by **action-prediction error** (MSE/MAE) vs. the demonstration — *not* a task success rate.
- Numbers: train loss 0.0855, **MSE 25.87 / MAE 3.01**. Published to HF `mitvho09/GR00T-Humanoid`. See [[Phase1_GR00T]].

## Why fine-tune a foundation model at all?
- **Data efficiency:** broad pretraining means a few demos can specialize it.
- **Portfolio signal:** shows you can run a real 3B VLA fine-tune + eval loop on cloud GPUs (L4/L40S), manage HF auth, and read action-space metrics.

## How it relates to the Isaac Lab track
GR00T = the **imitation-learning / foundation-model** path; the G1 Isaac Lab work = the **RL-in-sim** path. They're two complementary ways to get a language/vision-conditioned humanoid policy. A strong thesis would connect them (e.g. GR00T as a teacher, or sim-RL as fine-tune data).

Related: [[Vision_Language_Action_Models]] · [[Phase1_GR00T]] · [[Project_Overview]]
