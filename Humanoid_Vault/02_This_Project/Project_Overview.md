---
tags: [project, overview]
---

# Project Overview

## One sentence
A two-track humanoid study: (1) fine-tune NVIDIA's **GR00T** foundation model on G1 manipulation data, and (2) build a **language + vision conditioned** locomotion/navigation stack for the **Unitree G1** in **Isaac Lab**, trained with **PPO** and hardened with **domain randomization** for sim-to-real.

## Why it exists
A portfolio/thesis piece demonstrating end-to-end humanoid RL engineering: simulator setup, custom task design, large-scale parallel PPO, foundation-model fine-tuning, and a reproducible remote-GPU workflow.

## Where it runs
- **Compute:** [[Lightning_AI_Workflow|Lightning AI]] remote GPUs (NVIDIA **L40S** / **L4**).
- **Containerized:** Isaac Sim 5.1 + Isaac Lab in Docker (`nvcr.io/nvidia/isaac-sim:5.1.0`), reusable image pushed to GHCR.
- **Code on GitHub** (`Sushruths04/Humanoid`), large artifacts on Hugging Face, image on GHCR. Local machine is scratch only.

## The two tracks
| Track | What | Status | Key number |
|---|---|---|---|
| **GR00T** | Fine-tune `nvidia/GR00T-N1.7-3B` on cube-to-bowl data | ✅ done | Eval MSE **25.87** / MAE **3.01** |
| **Isaac Lab G1** | Locomotion → language → markers → vision → robustness | ✅ mostly | Robust reward **22.82**, ep-len **981/1000** |

See [[Architecture_Task_Hierarchy]] for how the G1 tasks build on each other, and [[Results_Summary]] for every number with its source.

## The single most important nuance
The word "**VLA**" (Vision-Language-Action) is **aspirational** here. Be precise:
- **Action** ✅ — real PPO locomotion policies, benchmarked.
- **Vision** ⚠️ — real CNN-over-camera pipeline that *runs*, but only smoke-tested.
- **Language** ❌ (as a learned model) — it's a **fixed deterministic embedding**, deliberately a placeholder for a frozen text encoder. See [[Phase2_G1_Locomotion_and_Language]].

Related: [[Phase1_GR00T]] · [[Phase3_Vision_VLA]] · [[Pipeline_and_Scripts]]
