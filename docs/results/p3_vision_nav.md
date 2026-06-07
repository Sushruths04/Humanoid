# P3 Vision Nav — Training Results

Task: Camera-conditioned CommandNav for G1 (pixel observations)  
DoD: ≥60% success on CommandNav with 64×64 RGB head-camera input  
Branch: `feat/planned-scripts`  
HuggingFace: `mitvho09/humanoid-g1-nav` → `checkpoints/p3_vision_nav/run_final/`

## Training Configuration

| Parameter | Value |
|---|---|
| Camera resolution | 64×64 RGB (TiledCamera) |
| Environments | 4096 parallel envs |
| Steps per env | 24 |
| Mini-batches | 8 |
| Max iterations | 300 (stopped at 200 — machine cutoff) |
| Save interval | 5 iterations |
| GPU | A100-SXM4-80GB |
| VRAM used | ~23 GB (RTX camera rendering + physics) |
| Iter time | ~12.9 s/iter (avg) |
| Total training time | ~43 min (200 iters) |

## Reward Progression

| Iteration | Mean Episode Reward | Notes |
|---|---|---|
| 0 | (random init) | CNN policy, no prior |
| 100 | **-6.76** | Still falling — locomotion unstable with camera obs |
| 200 | **+27.74** | Major improvement — robot navigating with pixel obs |

Reward delta iter 100→200: **+34.5 points** — policy crossed the locomotion/navigation threshold.

## Architecture

```
Observations: policy (proprioception + velocity cmd) + images (64×64 RGB)
              ↓
CNN encoder: Conv(32, 8×8, s4) → Conv(64, 4×4, s2) → Conv(64, 3×3, s1) → flatten
              ↓
MLP head: 512 → 256 → 128 → action (29 DoF)
```

- Shared CNN encoder between actor and critic (`share_cnn_encoders=True`)
- Adaptive LR schedule, desired KL=0.01
- Camera update period: 0.2 s (5 Hz) to reduce RTX rendering overhead

## Run History

| Run | Resolution | Iters | Final Reward | Status |
|---|---|---|---|---|
| run_10iter | 128×128 | 10 | N/A | A100 warmup / infra test |
| run_latest | 128×128 | ~8 | N/A | OOM at 16 mini_batches (78.4 GB) |
| run_final | 64×64 | 200 | **+27.74** | Machine cutoff at 200/300 |

## Key Finding: RTX Is the Bottleneck

On A100-SXM4-80GB with 128×128 cameras:
- RTX rendering: 4096 envs × 128×128 = 67M pixels/frame → 1.4 s/frame
- At 48 steps: 57 s rendering + 9 s CNN/PPO = **66 s/iter**
- A100 GPU compute utilization: ~14% (sitting idle 86% of the time)

With 64×64 cameras (run_final):
- RTX rendering: 4096 envs × 64×64 = 16.7M pixels/frame → 0.35 s/frame
- At 24 steps: 8.4 s rendering + 4.5 s CNN/PPO = **12.9 s/iter**
- 5.1× speedup vs 128×128

**Root cause**: TiledCamera uses Isaac Sim RTX ray-tracing with a BVH that scales with env count. Reducing resolution from 128→64 cuts render time 4× and enables completing 300 iters in ~65 min on A100.

## Analysis

The reward trajectory (−6.76 → +27.74 over 100 iters) mirrors the P0 CommandNav learning pattern: the first ~100 iters build stable locomotion, then the policy bootstraps navigation on top. At iter 200, the robot is:

1. Walking upright (upright reward component is positive)
2. Following velocity commands using pixel observations (nav_command reward positive)
3. Not falling at episode end (base_contact termination decreasing)

The training was **interrupted at 200/300 iterations** due to machine time cutoff. Extrapolating at the same reward improvement rate, iter 300 would reach ~35–45 points, comparable to P0's fully-converged reward.

**DoD assessment (≥60% success)**: Formal evaluation requires re-running the policy on CommandNav with eval metrics. The reward of +27.74 at iter 200 is strongly positive and indicates successful navigation behavior, but the exact success rate requires a GPU eval session. Estimated range: **50–70% success** based on reward correlation with P0 (94.5% success at ~30–35 reward).

## Comparison Table

| Task | Observation | Policy | Success |
|---|---|---|---|
| P0 CommandNav | Proprioception + cmd | MLP | 94.5% |
| P3 VisionNav (200 iters) | 64×64 RGB + cmd | CNN+MLP | ~50–70% (est.) |
| P3 VisionNav (300 iters, projected) | 64×64 RGB + cmd | CNN+MLP | TBD |

## Checkpoints

All 41 checkpoints (model_0 through model_200, every 5 iters) saved to:
- Persistent: `/teamspace/studios/this_studio/Humanoid/programs/checkpoints/p3_vision_nav/run_final/`
- HuggingFace: `mitvho09/humanoid-g1-nav/checkpoints/p3_vision_nav/run_final/`

Best checkpoint: `model_200.pt` (highest reward)

## Next Steps

1. **Eval run**: Load `model_200.pt`, disable training, run 100 episodes per command to get exact success rate
2. **Resume training**: 100 more iters from `model_200.pt` to reach 300 total (estimated +60 min on L40S)
3. **P4 Cosmos Predict**: Post-train video world model on navigation rollouts (A100-80G, ~6 hrs)
