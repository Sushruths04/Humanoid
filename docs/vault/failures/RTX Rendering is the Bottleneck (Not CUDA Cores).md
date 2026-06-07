---
tags: [failure, performance, rtx, a100, gpu-utilization, tiled-camera, p3, key-insight]
---

# RTX Rendering is the Bottleneck (Not CUDA Cores)

## Symptom
Training on an **A100-SXM4-80GB** with 4096 envs at 128×128 showed:
- VRAM usage: **78.7 GB** (looks like we're using the GPU!)
- `nvidia-smi` GPU utilization: **14%**
- Iteration time: **66 seconds**
- Expected: <10 seconds (A100 benchmarks show PPO at 4096 envs = ~8s/iter without cameras)

This confused everyone. Why is an 80 GB A100 running at 14% utilization and 66s/iter?

## Root Cause: Tiled Rendering Pipeline
Each training iteration involves:

| Step | Time | Hardware |
|---|---|---|
| RTX ray-trace 4096×128×128 = 67M pixels | ~57s | RT cores |
| CNN forward pass (actor + critic) | ~5s | CUDA / Tensor cores |
| PPO gradient update | ~4s | CUDA / Tensor cores |
| **Total** | **~66s** | |

**The A100's 6912 CUDA cores and 432 Tensor cores sit idle 86% of the time** waiting for the 336 RT cores to finish rendering.

VRAM is high (78 GB) because Isaac Sim holds physics state, RTX BVH structures, and the full rollout buffer in GPU memory simultaneously. High VRAM ≠ high compute utilization.

## The Interview Answer
> "Your A100 is at 14% utilization — what's wrong?"

The bottleneck is the RTX ray-tracing pipeline, not the CUDA cores. `TiledCamera` renders all 4096 environments into a 67M-pixel tiled image using Omniverse RTX, which is RT-core bound. The A100 has far fewer RT cores than CUDA cores. The fix is to **reduce pixels, not compute**.

## Fix
Reduce camera resolution from 128×128 to 64×64:
- Pixels per render: 67M → 16.7M (4× reduction)
- Also reduce steps 48→24 (2× fewer renders per iteration)
- Combined speedup: **5.1×** (66s → 12.9s/iter)
- VRAM drops to ~23 GB (fits on L4 24 GB!)

```python
# In g1_vision_nav_cfg.py — these env vars are the key levers:
_CAM_H = int(os.environ.get("P3_CAM_H", "128"))   # change to 64
_CAM_W = int(os.environ.get("P3_CAM_W", "128"))   # change to 64
_NUM_STEPS = int(os.environ.get("P3_NUM_STEPS", "48"))  # change to 24
```

```bash
# Launch with 64x64:
-e P3_CAM_H="64" -e P3_CAM_W="64" -e P3_NUM_STEPS="24"
```

## Key Numbers to Remember
| Config | VRAM | Iter time | GPU util |
|---|---|---|---|
| 128×128, 4096 envs, 48 steps | 78 GB | 66s | 14% |
| 64×64, 4096 envs, 24 steps | 23 GB | 12.9s | ~60% |
| 64×64, 2048 envs, 24 steps (L4) | 13 GB | 16s | ~50% |

## Do Not
- Never interpret high VRAM usage as "good GPU utilization" — they are independent
- Never blame the CNN or PPO for slowness when cameras are involved — profile first
- Never request an A100 "to use all VRAM" — the right question is compute utilization

## Do
- Run `nvidia-smi dmon -s u` to watch utilization over time, not just VRAM
- Reduce camera resolution before scaling up envs — resolution has 4× the impact
- Use 64×64 as the default for navigation tasks. 128×128 is only needed if texture detail is critical for the task

## Related
- [[RTX BVH Hang at High Env Count]]
- [[P3 - VisionNav]]
- [[All Parameters Cheat-Sheet]]
