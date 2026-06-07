---
tags: [task, p3, vision, camera, navigation, ppo, cnn, isaac-lab, completed]
---

# P3 — Vision Nav

**Status**: ✅ Complete  
**DoD**: ≥60% success on CommandNav with pixel-only observations  
**Result**: **96.28% success** — matches state-based P0 (94.5%) with 64×64 RGB camera

---

## What It Does
Trains the G1 humanoid to follow velocity commands using only a 64×64 RGB head-mounted camera + proprioception. No privileged state information (no ground-truth velocity, no contact sensors in the observation). The CNN must learn to infer necessary state from visual input.

---

## Architecture

```
Observations:
  policy group: proprioception (29 DoF pos/vel) + velocity command (3)
  images group: 64×64×3 RGB from TiledCamera (head_link/front_camera)

CNN encoder (shared actor/critic):
  Conv2d(3→32, kernel=8, stride=4)   → 14×14×32
  Conv2d(32→64, kernel=4, stride=2)  → 6×6×64
  Conv2d(64→64, kernel=3, stride=1)  → 4×4×64
  Flatten → 1024-dim

MLP head:
  1024 + 32 (policy) → 512 → 256 → 128 → 29 actions (joint targets)
```

Key config: `my-humanoid-project/my_humanoid_project/tasks/g1_vision_nav_cfg.py`

---

## Training Configuration

| Parameter | Value | Why |
|---|---|---|
| Camera | 64×64 RGB TiledCamera | 128×128 → RTX bottleneck at 66s/iter |
| Envs | 4096 | 8192 → BVH hang; 4096 = sweet spot |
| Steps/env | 24 | 48 → too slow; 24 = 2× speedup |
| Mini-batches | 8 | Enough to fit 3.5 GB image buffer |
| Iter time | 12.9s (A100) / 16s (L4) | |
| Max iterations | 300 | |
| Save interval | 5 | Never lose more than 5 iters |

---

## Training History (5 runs)

| Run | GPU | Config | Result | Why Stopped |
|---|---|---|---|---|
| run_10iter | A100-80G | 128×128, 8192 envs | **HANG** 30+ min | RTX BVH at 8192 envs |
| run_latest | A100-80G | 128×128, 4096, 8 mb | **OOM** 77.9 GB | Image buffer too large |
| (unnamed) | A100-80G | 128×128, 4096, 64 mb | 66s/iter, ran | Slow — 5.5 hrs for 300 iters |
| run_final | A100-80G | 64×64, 4096, 24 steps | 200/300 iters, +27.74 | Machine time limit |
| **run_300_l4** | **L4-24G** | **64×64, 4096, resume** | **499 iters, +141.35** | **Completed ✓** |

---

## Reward Progression (run_300_l4)

| Iteration | Reward | Notes |
|---|---|---|
| 200 (loaded) | +27.74 | From run_final warm start |
| 215 | +10.94 | Normalizer recalibrating |
| 260 | +109.73 | 4× previous best |
| 400 | +138 | Converging |
| **499 (final)** | **+141.35** | **96.28% success** |

---

## Key Failures Encountered
1. [[RTX BVH Hang at High Env Count]] — 8192 envs hung for 30 min
2. [[RTX Rendering is the Bottleneck (Not CUDA Cores)]] — A100 at 14% utilization
3. [[OOM With Camera Rollout Buffer]] — 128×128 OOM, needed 64 mini-batches
4. [[update_period Does Not Reduce Render Time]] — misconception about render frequency
5. [[Docker Image Lost on GPU Upgrade]] — re-pull 17.6 GB on every new machine
6. [[RSL-RL Resume Resets Loop Counter]] — ran 300 new iters not 100 (actually better)
7. [[play.py Fails - Custom Task Not Registered]] — needed custom_play.py
8. [[Python File Corruption Over SSH - Use Python Write]] — heredoc mangled code
9. [[play.py Checkpoint Bare Filename Not Found]] — `--checkpoint model.pt` calls retrieve_file_path (bare path); use `--load_run` only
10. Eval OOM on L4 — training process still held VRAM after completing; kill it first

---

## Result vs P0

| Policy | Input | Success |
|---|---|---|
| P0 CommandNav (MLP) | Proprioception + velocity cmd | 94.5% |
| **P3 VisionNav (CNN+MLP)** | **64×64 RGB + proprioception + cmd** | **96.28%** |

**Pixel observations match state observations for navigation.** This is the key result — adding a camera does not degrade the policy when trained with sufficient iterations.

---

## Checkpoints (HuggingFace: mitvho09/humanoid-g1-nav)

| Run | Path | Best checkpoint | Reward |
|---|---|---|---|
| run_final | `checkpoints/p3_vision_nav/run_final/` | model_200.pt | +27.74 |
| **run_300_l4** | `checkpoints/p3_vision_nav/run_300_l4/` | **model_499.pt** | **+141.35** |

---

## Eval Command
```bash
docker exec \
  -e PYTHONPATH="/workspace:/workspace/my-humanoid-project:/workspace/isaaclab/source" \
  -e P3_CAM_H="64" -e P3_CAM_W="64" \
  isaac-lab-base /workspace/isaaclab/isaaclab.sh -p \
  /workspace/my-humanoid-project/custom_play.py \
  --task Humanoid-G1-VisionNav-v0 --headless --enable_cameras \
  --num_envs 512 \
  --load_run <run_timestamp> \
  --checkpoint model_499.pt
```

---

## Related
- [[P0 - CommandNav]] — state-based baseline this beats
- [[RTX Rendering is the Bottleneck (Not CUDA Cores)]] — the key insight
- [[Training Recipe]]
- [Full engineering log](../../results/p3_vision_nav_full_engineering_log.md)
- [Results doc](../../results/p3_vision_nav.md)
