---
tags: [failure, tiled-camera, performance, isaac-lab, p3, misconception]
---

# update_period Does Not Reduce Render Time

## Symptom
Set `TiledCameraCfg(update_period=0.2)` expecting it to make the camera render at 5 Hz instead of every physics step (200 Hz), hoping to cut render time by ~40×. Iteration time was unchanged: still 66s.

## What I Expected
```python
# Expected: camera renders every 0.2s of sim time = 1 in 40 physics steps
# Expected speedup: 40× fewer RTX renders per training iteration
self.scene.head_camera = TiledCameraCfg(
    update_period=0.2,  # <-- I thought this would skip 39 out of 40 renders
    ...
)
```

## Root Cause: update_period Controls Python Reads, Not RTX Renders
`update_period` controls how often Isaac Lab **copies** the rendered image from the GPU render buffer to the Python tensor returned by `cam.data.output["rgb"]`. It does NOT tell the RTX renderer to skip frames.

Isaac Sim's RTX renderer runs every physics step regardless. The image is rendered at full rate into an internal GPU buffer; `update_period` just gates how often Python sees a new value.

Think of it like this:
- RTX renderer = a camera continuously taking photos
- `update_period` = how often you glance at the photo — not how often the camera clicks

## Fix
There is no simple way to skip RTX renders in Isaac Lab 2.3.x. The only lever is **resolution**:
- 128×128 → 64×64: renders are 4× cheaper (fewer pixels per ray-trace)

```python
TiledCameraCfg(
    update_period=0.2,  # keep this — it reduces Python-side copy overhead slightly
    height=64,          # THIS is what actually speeds things up
    width=64,
)
```

## Lesson
> `update_period` is a **Python sampling rate**, not a **GPU render frequency**.

When profiling camera-based RL, always check: is the GPU busy ray-tracing, or is it idle waiting for Python? The answer determines which knob to turn.

## Related
- [[RTX Rendering is the Bottleneck (Not CUDA Cores)]]
- [[P3 - VisionNav]]
