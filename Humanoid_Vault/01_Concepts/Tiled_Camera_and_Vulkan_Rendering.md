---
tags: [concept, vision, rendering, infra]
---

# TiledCamera & Vulkan Rendering

## TiledCamera
To do **vision RL at scale** you must render thousands of camera views per step. Isaac Lab's **`TiledCamera`** packs all parallel envs' views into one big GPU "tiled" render — far faster than N separate cameras. This project mounts one on the G1 head (`Robot/head_link/front_camera`), RGB, default 128×128, pinhole optics. See `g1_vla_vision_cfg.py`.

## Observation handling
`get_camera_rgb` reads `camera.data.output["rgb"]`, **normalizes** `/255.0`, and permutes **HWC → CHW** for the CNN. It's exposed as a dedicated `images` obs group so the CNN path is separate from proprioception ([[Architecture_Task_Hierarchy]]).

## The Vulkan blocker (and the fix)
**Why it happened:** camera sensors render through **Vulkan**, which needs NVIDIA graphics libs (`libGLX_nvidia.so.0`) mounted into the container. The Lightning container had **compute** (CUDA) but not **graphics** mounted, so `nvidia-smi` worked but rendering failed.

**The fix that worked (verified):**
- Use the **headless rendering kit**: `--experience .../apps/isaaclab.python.headless.rendering.kit` (renders without a live GUI/X server).
- Pass **`--enable_cameras`**.
- Start at tiny resolution (`32×32`) to keep VRAM/perf sane, then scale.

After that, camera training reached PPO and scaled to 2048 envs. Full detail in [[Phase3_Vision_VLA]].

## Lesson (interview-worthy)
"`nvidia-smi` works" ≠ "rendering works." **Compute and graphics are separately provisioned** in containers — a classic, non-obvious GPU-infra gotcha. The cheap workaround (headless rendering kit + low-res cameras) beats fighting driver mounts.

Related: [[Phase3_Vision_VLA]] · [[Isaac_Lab_and_Isaac_Sim]]
