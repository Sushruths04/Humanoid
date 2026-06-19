---
tags: [concept, vla]
---

# Vision-Language-Action (VLA) Models

## The idea
A **VLA** maps **(camera image, language instruction) → robot action**. It's the dominant 2024–2026 paradigm for generalist robots (RT-2, OpenVLA, π0, GR00T). Three ingredients:
- **V**ision — a visual encoder over camera frames.
- **L**anguage — a (usually frozen) text encoder turning instructions into embeddings.
- **A**ction — a policy/decoder producing motor actions, trained by imitation and/or RL.

## How *this* project maps onto VLA (be precise)
| Letter | Real VLA | This project |
|---|---|---|
| **Action** | learned policy | ✅ real PPO locomotion policy ([[PPO_for_Locomotion]]) |
| **Vision** | image encoder | ⚠️ Nature-CNN over a head camera — **runs, smoke-only** ([[Phase3_Vision_VLA]]) |
| **Language** | frozen text encoder | ❌ **deterministic hash embedding**, fixed per run ([[Phase2_G1_Locomotion_and_Language]]) |

So it's a **VLA-shaped scaffold**, not a trained VLA. That's a perfectly fine portfolio framing **as long as you say so**.

## Two ways VLAs get trained
1. **Imitation / behavior cloning** on teleop demos (how GR00T-style models start). See [[GR00T_Foundation_Model]].
2. **RL in sim** with language/vision in the observation (this project's path) — cheaper, but has a sim-to-real gap ([[Domain_Randomization_and_Sim2Real]]).

## To turn the scaffold into a real VLA
Randomize commands + add command-dependent reward, then a frozen text encoder, then vision via teacher→student distillation. Full ladder in [[Open_Questions_and_Next_Steps]].

Related: [[GR00T_Foundation_Model]] · [[Phase3_Vision_VLA]] · [[Phase2_G1_Locomotion_and_Language]]
