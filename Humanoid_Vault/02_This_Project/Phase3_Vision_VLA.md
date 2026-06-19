---
tags: [project, phase, vision]
---

# Phase 3 — Vision-VLA (camera + CNN policy)

## Goal
Give the G1 **pixels**: a head-mounted camera feeding a CNN policy, so the robot can (eventually) act on what it sees.

## What was built (source: `g1_vla_vision_cfg.py`)
- **Sensor:** `TiledCameraCfg` on `Robot/head_link/front_camera`, RGB, default **128×128**, pinhole optics, ROS convention, offset `(0.1, 0, 0.1)`. (`TiledCamera` = the GPU-batched camera that makes thousands of parallel renders feasible — see [[Tiled_Camera_and_Vulkan_Rendering]].)
- **Obs:** `get_camera_rgb` normalizes to `[0,1]` and permutes to **CHW**; exposed as a separate `images` obs group.
- **Policy:** `RslRlCNNModelCfg` — a **Nature-CNN** encoder: channels **[32, 64, 64]**, kernels **[8, 4, 3]**, strides **[4, 2, 1]**, ReLU, then MLP heads `[256, 256]` (ELU). `share_cnn_encoders = True` → actor and critic share the visual encoder (cheaper, common in pixel PPO).
- **PPO:** clip 0.2, entropy 0.008, lr 5e-4 adaptive, γ 0.99, λ 0.95, desired_kl 0.01, 5 epochs, 4 minibatches, 24 steps/env.

## The Vulkan blocker → SOLVED
**Problem:** camera rendering needs **Vulkan**; the container errored `libGLX_nvidia.so.0: cannot open shared object file` (graphics libs not mounted by `nvidia-container-toolkit`). Compute-only phases were unaffected.

**Fix (verified by `thesis/state/30_vision_vla.done` + `STEP-30-vision-vla.md`):** run **headless with the rendering kit**, not the live GUI:
- `--experience .../apps/isaaclab.python.headless.rendering.kit`
- `--enable_cameras`
- start tiny: `VLA_CAMERA_HEIGHT=32 VLA_CAMERA_WIDTH=32`

Camera-enabled training then reached PPO and **scaled 32 → 256 → 1024 → 1536 → 2048 envs**.

## Honest status
- ✅ Pipeline **runs end-to-end with pixels** at smoke scale.
- ❌ **No long vision training run / saved vision checkpoint** exists in the repo (only locomotion `g1_robust` + GR00T checkpoints are saved).
- So this is "**vision pipeline verified**", not "**vision policy trained & benchmarked**". See [[Open_Questions_and_Next_Steps]].

## Scripts
`30_vision_vla.sh` (smoke), `31_vision_vla_cnn.sh` (128×128 CNN, 500 iters), `32_vision_vla_play.sh` (records an mp4 rollout — GUI-free, safe on Lightning).

Related: [[Tiled_Camera_and_Vulkan_Rendering]] · [[Architecture_Task_Hierarchy]] · [[Results_Summary]]
