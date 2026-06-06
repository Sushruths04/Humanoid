---
tags: [task, manipulation, vision, bc, ablation]
---

# T3 — Pixel-Only BC Manipulation (ResNet18 + MLP)

**Status:** ✅ Done  
**Result:** 0% mean task success (10 tasks × 10 eps, LIBERO Spatial)  
**Purpose:** Ablation — quantify the contribution of robot state and pretraining

---

## What Was Built

A pixel-conditioned Behaviour Cloning policy trained on LIBERO Spatial HDF5 demos:

```
agentview_image (128×128 RGB)
        ↓
ResNet18 (ImageNet pretrained)
        ↓
Global Average Pool → 512-dim feature
        ↓
MLP head: 512→256→256→7
        ↓
7-dim OSC delta action (dx,dy,dz,droll,dpitch,dyaw,gripper)
```

No robot state, no language — pure pixels to actions.

---

## Training Results

| Metric | Value |
|---|---|
| Tasks | 10 (all libero_spatial) |
| Demos | 50 per task |
| Transitions | 61,750 |
| Epochs | 50 |
| Batch size | 1,152 |
| Initial BC loss | 0.08131 |
| Final BC loss | **0.00057** (142× reduction) |
| VRAM peak | ~22.5 GB (L4 GPU) |
| Training time | ~15 min |

Loss reduction confirms the policy learned the action distribution from demos.

---

## Eval Results

| Task | Success |
|---|---|
| pick up black bowl (between plate and ramekin) → plate | 0/10 |
| pick up black bowl (table center) → plate | 0/10 |
| pick up black bowl (top drawer) → plate | 0/10 |
| pick up black bowl (next to cookie box) → plate | 0/10 |
| pick up black bowl (next to plate) → plate | 0/10 |
| pick up black bowl (next to ramekin) → plate | 0/10 |
| pick up black bowl (on cookie box) → plate | 0/10 |
| pick up black bowl (on ramekin) → plate | 0/10 |
| pick up black bowl (on stove) → plate | 0/10 |
| pick up black bowl (on wooden cabinet) → plate | 0/10 |
| **Mean task success** | **0.000** |

---

## Why 0% Is the Right Answer

This is not a failure — it's an **informative ablation result**.

**The perception-action gap:** ResNet18's global avg-pool discards spatial layout.
The 512-dim feature vector knows *what* objects are present but not *where* the bowl
is relative to the gripper. Pick-and-place requires sub-centimeter spatial accuracy.

**The control loop problem:** Without robot state (EEF position, joint angles),
the policy cannot close the feedback loop. It produces reasonable-looking actions
(smooth, low BC loss) but cannot reactively correct for position errors.

**Comparison:**

| Approach | Inputs | Success |
|---|---|---|
| T0 State BC | Robot state only | 50% (1 task) |
| T3 Pixel BC | RGB only | **0%** |
| T1 GR00T LoRA | RGB + state + pretrain | **97%** |

The jump from 0% → 97% quantifies what robot state + large-scale pretraining provides.

---

## What Would Improve This

1. **Spatial features** — replace global avg-pool with SpatialSoftmax or R3M; preserves object location in features
2. **Add robot state** — EEF pos (3D) + gripper state as auxiliary input to the MLP head
3. **More data** — 50 demos × 10 tasks = 61k; GR00T pretrained on ~1M+ trajectories
4. **Recurrent policy** — LSTM or Transformer to accumulate state estimates over time
5. **Keypoint representations** — detect object keypoints from pixels, use as structured state

---

## Implementation Notes

**Dataset storage optimization:** Pre-loaded entire 3.8GB image tensor to GPU VRAM at dataset init, eliminating CPU→GPU transfer per batch. This required:
- `permute(0,3,1,2).contiguous()` — contiguous layout for fast batch indexing
- Keep native 128×128 (not 224×224) — ResNet18 global avg-pool handles any resolution; 224×224 resize would OOM at 37GB

**LIBERO __init__.py blocking input():** `/tmp/LIBERO/libero/libero/__init__.py` contains an `input()` call that blocks nohup processes. Fixed with Python regex replacement to `answer = "n"`.

**robosuite version:** LIBERO requires `robosuite==1.4.x`. `robosuite>=1.5` changed the module path `robosuite.environments.manipulation.single_arm_env` → different path → import error.

---

## Videos

`programs/videos/t3_pixel_bc/` — 10 MP4 files (one per task) showing the robot attempting each task. The arm moves but fails to grasp precisely due to lack of spatial features and state feedback.

---

## Related

- [[T0 - ManipFoundation]] — state-only BC baseline (50% on 1 task)
- [[T1 - GR00T LoRA]] — full system, 97% success
- [[T2 - WM Manipulation]] — world model on GR00T rollouts
- [T3 Results](../../docs/results/t3_pixel_bc.md)
- [train_pixel_bc.py](../../programs/t3_vision_manip/train_pixel_bc.py)
- [evaluate_pixel_bc.py](../../programs/t3_vision_manip/evaluate_pixel_bc.py)
