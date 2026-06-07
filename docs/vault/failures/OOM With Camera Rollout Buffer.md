---
tags: [failure, oom, memory, ppo, mini-batches, camera, p3]
---

# OOM With Camera Rollout Buffer

## Symptom
Training with 128×128 cameras, 4096 envs, 48 steps crashed during the PPO update:
```
torch.OutOfMemoryError: CUDA out of memory.
Tried to allocate 2.82 GiB
(GPU 0; 79.20 GiB total capacity; 74.17 GiB already allocated)
```
Then again with 16 mini_batches:
```
torch.OutOfMemoryError: CUDA out of memory.
Tried to allocate 1.41 GiB
(GPU 0; 79.20 GiB total capacity; 78.38 GiB already allocated)
```

## Root Cause: Rollout Buffer Size
PPO stores the entire rollout buffer before computing gradients. With 128×128 cameras:

```
Buffer size = 4096 envs × 48 steps × 128 × 128 × 3 channels × 4 bytes (float32)
            = 4096 × 48 × 49152 bytes
            = ~14 GB for images alone
```

Plus physics state, advantages, values, log_probs → total buffer ~18 GB.

During the PPO update with `num_mini_batches=8`, each mini-batch processes 1/8 of this:
- Mini-batch images: 14/8 = 1.75 GB
- CNN forward+backward: ~2.82 GB gradient footprint
- Total: exceeds remaining headroom on 80 GB GPU

## Fix: Increase Mini-Batches

```python
# In G1VisionNavCnnRunnerCfg:
_MINI_BATCHES = int(os.environ.get("P3_MINI_BATCHES", "64"))
algorithm = RslRlPpoAlgorithmCfg(
    num_mini_batches=_MINI_BATCHES,  # 64 splits each update into 64 micro-batches
    ...
)
```

```bash
-e P3_MINI_BATCHES="64"
```

With 64 mini-batches: 14 GB / 64 = 219 MB per batch. CNN gradient ~350 MB. Total ~570 MB per step — easily fits.

## Why 64 Works (Not 8 or 16)
The memory needed for a gradient step scales with batch size:
- 8 mini_batches: 1.75 GB images + gradients → OOM at 78.4 GB
- 16 mini_batches: 875 MB images + gradients → OOM at 78.4 GB  
- 64 mini_batches: 219 MB images + gradients → stable at 78.7 GB

Isaac Sim non-PyTorch overhead (physics, RTX structures) = ~18-20 GB baseline that can't be reduced.

## Better Fix: Reduce Resolution
64×64 instead of 128×128 cuts buffer to 3.5 GB, making mini_batches=8 work fine at only 23 GB VRAM total. This also speeds up rendering 4×. Resolution reduction dominates over mini_batch tuning.

## Do Not
- Don't start at `num_mini_batches=4` with cameras — always start at 8 and increase if OOM
- Don't try to reduce `num_envs` as the fix — Isaac Sim baseline VRAM (~18 GB for physics) barely changes with half the envs, so you save very little

## Do
- Calculate buffer size before starting: `envs × steps × H × W × 3 × 4 bytes`
- Use 64×64 resolution as default — keeps buffer at 3.5 GB regardless of mini_batch count
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation

## Related
- [[RTX Rendering is the Bottleneck (Not CUDA Cores)]]
- [[PPO with RSL-RL]]
- [[All Parameters Cheat-Sheet]]
