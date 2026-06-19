---
tags: [project, phase, gr00t]
---

# Phase 1 — GR00T Fine-Tuning

## Goal
Show you can fine-tune a **humanoid foundation model** ([[GR00T_Foundation_Model]]) and evaluate it on action prediction.

## What was done (verified)
- **Base model:** `nvidia/GR00T-N1.7-3B`.
- **Data:** `cube_to_bowl_5` demo dataset (5 trajectories, ~568 steps each).
- **Train:** full **10,000-step** fine-tune → `checkpoint-10000`.
- **Eval:** action-prediction error on a held-out trajectory.

## Real results (source: `thesis/results/gr00t_eval_smoke/summary.txt`)
| Metric | Value |
|---|---|
| Final train loss | 0.0855 |
| Unnormalized Action **MSE** | **25.88** |
| Unnormalized Action **MAE** | **3.01** |
| Eval exit status | 0 (clean) |

Preflight (from `gr00t_demo/summary.txt`): torch 2.7.1+cu126, CUDA on **NVIDIA L4**, GR00T import OK, model load ~13.3s, HF token present.

**Checkpoint published:** Hugging Face `mitvho09/GR00T-Humanoid`.

## How to read MSE/MAE here
These are **action-space regression errors** (predicted vs. demonstrated joint/EE actions), not a task success rate. MSE 25.9 is a *relative* baseline number from a small 5-traj eval — useful to show the eval harness works, **not** a SOTA claim. Be honest about this in interviews.

## Scripts
`01_gr00t_install.sh` → `02_gr00t_demo.sh` → `03_gr00t_gendata.sh` → `04_gr00t_finetune.sh` → `05_gr00t_eval.sh`. See [[Pipeline_and_Scripts]].

Related: [[GR00T_Foundation_Model]] · [[Vision_Language_Action_Models]] · [[Results_Summary]]
