---
tags: [failure, isaac-lab, rtx, camera, tiled-camera, performance, p3]
---

# RTX BVH Hang at High Env Count

## Symptom
Training script with `TiledCamera` enabled launched but **hung for 30+ minutes** at scene initialization and never started printing iteration stats. No error, no crash — just silence.

## What I Tried First
Waited thinking it was JIT warmup. After 30 min still nothing. Checked `nvidia-smi` — GPU at 100% on one process, VRAM slowly climbing, no crash.

## Root Cause
Isaac Sim's RTX renderer builds a **Bounding Volume Hierarchy (BVH)** over all scene geometry before it can ray-trace. With `--num_envs 8192` and `TiledCamera`, the BVH must index 8192 copies of the G1 robot + ground — hundreds of millions of triangles. BVH construction time at this scale is **O(N log N)** in scene triangle count and took 30+ minutes.

This is a known Omniverse/Isaac Sim constraint: RTX BVH does not scale linearly with environment count for RL use cases.

## Fix
Drop to `--num_envs 4096`. At 4096 envs, BVH builds in ~2 minutes.

```bash
# Don't do this with cameras:
--num_envs 8192

# Use this:
--num_envs 4096
```

## Do Not
- Never start a camera-enabled training run at 8192+ envs without first testing at 1024
- Never assume a hang is "just JIT warmup" — if nothing prints after 5 minutes, something is wrong

## Do
- Start camera runs at 1024 envs to verify startup, then scale to 4096
- Monitor `nvidia-smi` during startup — if VRAM is climbing but no output appears, it's likely BVH construction

## Related
- [[Isaac Sim Docker Container]]
- [[RTX Rendering is the Bottleneck (Not CUDA Cores)]]
- [[P3 - VisionNav]]
