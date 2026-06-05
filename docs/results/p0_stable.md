# P0 Command-Nav Stable — Results

Task: `Humanoid-G1-CommandNav-v0` (Unitree G1, Isaac Lab, RSL-RL PPO).
Stability follow-up: added `upright_reward` (weight=0.5) as a new `RewTerm` to
reduce the 28.1% fall rate from the P0 baseline.

## Training (4096 envs, 500 iters, NVIDIA T4)

| Metric | Early (iter ~5) | Final (iter 499) |
| --- | ---: | ---: |
| Mean episode length | 52 | ~940 |
| `nav_command` reward | ≈ 0 | +4.5–5.0 |
| `upright` reward | 0.024 | +0.43–0.46 |
| `base_contact` termination | 100% | ~22% (train) |
| Mean total reward | -4.7 | positive |

## Evaluation (256 episodes)

| Metric | P0 Baseline | P0 Stable | Change |
| --- | ---: | ---: | ---: |
| Commanded-target success rate | 0.945 | **0.926** | -0.019 |
| Success by command [red, blue] | [0.958, 0.934] | **[0.949, 0.906]** | — |
| **Fall rate** | **0.281** | **0.078** | **-72%** ✅ |
| Mean final distance (m) | 0.348 | 0.269 | -0.079 |
| Mean episode length | 768 | 932 | +164 |

## Definition of Done

- Commanded-target success ≥ 0.70: **MET (0.926)**.
- Fall rate < 0.10: **MET (0.078)**. ✅

## What Changed

Added `upright_reward` (weight=0.5) as a `RewTerm` to `CommandConditionedG1NavCfg`:
- Pure function: `(1 - 2*(x² + y²)).clamp(0)` from quaternion [w,x,y,z]
- Returns 1.0 when perfectly upright, 0.0 when horizontal
- TDD: 4 tests, all green before GPU run
- Weight configurable via `COMMANDNAV_UPRIGHT_WEIGHT` env var (default 0.5)

## Training Dynamics

Fall rate followed a two-phase pattern:
1. **Phase 1 (iter 0–70):** Learn upright posture first; fall rate 100% → 16%
2. **Phase 2 (iter 70–140):** Learn to navigate at cost of stability; fall rate 16% → 32%
3. **Phase 3 (iter 140–500):** Navigate + stay upright; fall rate 32% → 22% (train) → 7.8% (eval)

## Reproduce

```bash
# train
docker exec -e PYTHONPATH=/workspace:/workspace/my-humanoid-project:/workspace/isaaclab/source \
  -e COMMANDNAV_UPRIGHT_WEIGHT=0.5 \
  isaac-lab-base /workspace/isaaclab/isaaclab.sh -p /workspace/my-humanoid-project/custom_train.py \
  --task Humanoid-G1-CommandNav-v0 --headless --num_envs 4096 --max_iterations 500
# collect + evaluate
bash programs/scripts/collect_p0_stable.sh
```

Checkpoint: `mitvho09/humanoid-g1-nav` → `checkpoints/g1_commandnav_stable/model_499.pt`
