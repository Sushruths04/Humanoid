---
tags: [moc, index]
---

# 🤖 Humanoid VLA — Study Vault

> A study-oriented knowledge base for the **Humanoid VLA mini-thesis**: language- and vision-conditioned locomotion/navigation for the **Unitree G1** in **Isaac Lab**, plus **GR00T** foundation-model fine-tuning. All notes are grounded in the actual code at `/home/laptop/AUtonomous/Humanoid/`, **not** the summary markdown.

## How to use this vault
1. Read [[Project_Overview]] for the 2-minute picture.
2. Learn the building blocks in **01_Concepts** (one idea per note).
3. Study what *you* actually built in **02_This_Project** (phase by phase).
4. Drill yourself with [[Interview_QA]] and review [[Open_Questions_and_Next_Steps]].

## Map of Content

### 01 — Concepts (the theory you should know)
- [[Isaac_Lab_and_Isaac_Sim]] — the simulator + RL framework
- [[PPO_for_Locomotion]] — the RL algorithm doing the work
- [[RSL_RL_Runner]] — the trainer (manager-based env + on-policy runner)
- [[Vision_Language_Action_Models]] — what a VLA *is*, and how yours compares
- [[GR00T_Foundation_Model]] — NVIDIA's humanoid foundation model
- [[Tiled_Camera_and_Vulkan_Rendering]] — pixels in sim, and the Vulkan blocker
- [[Domain_Randomization_and_Sim2Real]] — robustness on rough terrain

### 02 — This Project (what you actually did)
- [[Project_Overview]]
- [[Architecture_Task_Hierarchy]] — the 4-level config inheritance chain
- [[Phase1_GR00T]] — fine-tune + eval (MSE 25.87)
- [[Phase2_G1_Locomotion_and_Language]] — baseline + 16-dim command embedding
- [[Phase2.5_Sim2Real_Robust]] — rough terrain + domain randomization (reward 22.82)
- [[Phase3_Vision_VLA]] — head camera + CNN policy (blocker → solved)
- [[Pipeline_and_Scripts]] — the 00→99 script pipeline
- [[Results_Summary]] — every real number, with its source file

### 03 — Study aids
- [[Interview_QA]] — defend the project in an interview
- [[Glossary]] — quick definitions
- [[Open_Questions_and_Next_Steps]] — honest gaps + what to do next

## Honesty banner (read this)
This project is **strong on infrastructure and locomotion RL**, and **honest-but-early on the "VLA" claim**:
- The "**language**" is a **deterministic 16-dim hash embedding** of 4 fixed phrases — *not* a learned text encoder (see [[Phase2_G1_Locomotion_and_Language]]).
- The "**vision**" CNN pipeline **runs and reaches PPO**, but only at smoke scale; no long vision run / saved vision checkpoint exists yet (see [[Phase3_Vision_VLA]]).
- The **real, benchmarked wins** are GR00T fine-tuning and **robust G1 locomotion on rough terrain** (see [[Results_Summary]]).

Knowing exactly where the line is between "real result" and "verified plumbing" is the most interview-valuable thing in this vault.
