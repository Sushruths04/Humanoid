---
tags: [moc, index, humanoid, physical-ai]
---

# 🤖 Physical-AI Learning Vault — START HERE

This vault documents Sushruth's 7–8 month physical-AI portfolio program: building a language-driven loco-manipulation humanoid from scratch using NVIDIA Isaac Lab, GR00T, and Cosmos.

> **Tip:** Open the entire repo root as your Obsidian vault so every `.md` file in the repo is reachable.

---

## 🗺️ What Has Been Built So Far

| Checkpoint | Task | Result | Status |
|---|---|---|---|
| P0 | [[P0 - CommandNav]] | 94.5% success | ✅ Done |
| P1.2 | [[P1.2 - LangNav]] | 98.8% per-command | ✅ Done |
| P1.3 | [[P1.3 - ObstacleNav]] | 85.9% success | ✅ Done |
| P1.4 | [[P1.4 - SeqNav]] | 80.9% full-sequence / 94.5% ordering | ✅ Done |
| P2 | [[P2 - World Model]] | Isaac rollouts trained, loss 0.76→0.011 | ✅ Done |
| T0 | [[T0 - ManipFoundation]] | BC baseline 50% task_success on libero_spatial | ✅ Done |
| T1 | [[T1 - GR00T LoRA]] | **97.0% mean task success** (10 tasks, 20 eps each) | ✅ Done |
| T2 | [[T2 - WM Manipulation]] | Dreamer-mini on 200 GR00T rollouts, loss 1.40→0.008 | ✅ Done |
| P3 | Vision nav (cameras) | Scaffold only | 🔜 Next |
| P4 | Cosmos world sim | Scaffold only | 🔜 Future |
| C5 | Loco-manipulation capstone | Not started | 🔜 Future |

---

## 📚 Learn the Concepts

Start here if you're new to any of these topics:

- [[Isaac Lab Manager-Based RL]] — how Isaac Lab structures RL environments
- [[Command-Conditioned Navigation]] — the core design pattern used in all nav tasks
- [[Velocity-Command Steering Law]] — how the robot is told to walk toward a target
- [[Reward Shaping & Progress Rewards]] — why reward design is critical
- [[Reward Engineering Deep Dive]] ⭐ — all reward formulas with math + interview Q&A
- [[Robot Kinematics & Dynamics]] ⭐ — FK, IK, Jacobian, CoM, OSC, PD control
- [[PPO with RSL-RL]] — the training algorithm
- [[Sequential Subgoal Navigation]] — multi-step goal ordering
- [[World Models (Dreamer-mini)]] — building a world model from scratch
- [[Frozen Text Encoder for Language Tasks]] — language grounding for nav

---

## 🛠️ Set Up The Environment

Read these IN ORDER if starting from a fresh GPU machine:

1. [[Lightning Studio Environment]] — repo location, persistent vs. ephemeral storage
2. [[GHCR Image & Auth]] — pulling the 17.6 GB Isaac Sim container (now public)
3. [[Isaac Sim Docker Container]] — the exact bring-up sequence
4. [[PYTHONPATH & Python Interpreters]] — the two Python interpreters; mixing them breaks everything
5. [[SSH Key Recovery]] — when `publickey denied` after machine restart

---

## 🏃 Run Training

- [[Training Recipe]] — the exact command to train any nav task
- [[Evaluation Harness]] — how results are measured + the metrics
- [[Rendering Demo Videos]] — how to render a playback video from a checkpoint
- [[Reproduce From Scratch]] — step-by-step recipe to reproduce all results from a cold start

---

## ⚠️ Failures & How They Were Solved

This is the most valuable section. Read these to avoid repeating mistakes:

- [[00 - Failure Index]] — quick-reference table of every failure
- [[SeqNav Stand-Still Local Optimum]] ⭐ — the headline debugging story; 6+ hours, root-caused a training bootstrap failure
- [[Decorative Navigation Defect]] — the original meta-lesson about fake ML
- [[Results Lost to Ephemeral Container Storage]] — exit 0 but no output file
- [[Eval Crash - Missing Buffer]] — wrong evaluator for a new task type
- [[GHCR Auth Denied]] — token expired/revoked on fresh machine
- [[container.py Forces Rebuild]] — how to bypass the helper script
- [[SSH Heredoc Apostrophe Corruption]] — why we scp files instead
- [[Video Render Never Exits]] — sim loop runs forever after saving
- [[SSH Key Drops on Restart]] — re-download procedure

---

## 📊 Reference

- [[All Parameters Cheat-Sheet]] — every hyperparameter in one table
- [[Common Failure Patterns]] — pattern → tell → fix
- [[Glossary]] — Isaac Lab, RSL-RL, RL concepts defined
- [[Interview Prep - Master Guide]] ⭐ — all interview Q&As: RL, kinematics, rewards, VLA, BC, world models
- [[Video Index]] — all portfolio videos, what each shows, how each was trained

---

## 🔗 Governing Docs (repo)

- [Master Roadmap](../docs/MASTER_ROADMAP_CONVERGED.md) — the governing program plan
- [GPU VRAM Requirements](../docs/GPU_VRAM_REQUIREMENTS.md) — rent the right card
- [Planned Scripts Runbook](../docs/PLANNED_SCRIPTS.md) — batch GPU test procedure
- [P0 Results](../docs/results/p0_baseline.md) | [P1.2](../docs/results/p1_langnav.md) | [P1.3](../docs/results/humanoid-g1-obstaclenav-v0.md) | [P1.4](../docs/results/humanoid-g1-seqnav-v0.md)

---

## ⏭️ Remaining Tasks (TODO board)

- [x] Side-by-side demo reel — CommandNav/ObstacleNav/SeqNav hstack → `programs/videos/demo_reel.mp4` (HF: `videos/demo_reel.mp4`)
- [x] P0 fall-rate follow-up — `upright_reward` (weight=0.5), retrained → **7.8% fall rate** ✅ (was 28.1%)
- [x] T0 CPU harness — `manip_metrics.py` + `evaluate_manip.py` scaffold (10 tests, all green)
- [x] T0 LIBERO install + BC baseline — `train_bc_libero.py` → **50% task_success** on libero_spatial:0 ✅
- [x] T1 complete — GR00T N1.7 eval → **97.0% mean task success** across 10 LIBERO Spatial tasks ✅
- [ ] P2 on real Isaac rollouts (currently only toy point-mass)
- [ ] P3 Vision nav (24 GB + RT cores; pixel-dependence probe)
- [ ] P3 Vision nav (24 GB + RT cores; pixel-dependence probe)
- [ ] T2/T3: WM for manip + vision manip
- [ ] P4 Cosmos Predict post-train (80 GB A100 burst)
- [ ] T4 + C5 Loco-manipulation capstone

---

*Vault last updated 2026-06-06. T1 (97.0%), T2 (WM loss 1.40→0.008), T3 training in progress. Next: P3 Vision Nav.*
