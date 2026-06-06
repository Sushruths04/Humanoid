# T2 — World Model for Manipulation

Dataset: `/teamspace/studios/this_studio/Humanoid/programs/data/manip_rollouts_groot.pt`  
Episodes: 200  
obs_dim: 8  act_dim: 7  
Training steps: 3000  

## Training

| Metric | Value |
|---|---|
| Initial loss | 1.4026 |
| Final loss | 0.0082 |
| Imagined mean reward | 0.0108 |
| Real mean reward | 0.0078 |
| DoD: imagined reward finite | ✅ PASS |

## Summary

Dreamer-mini trained on 200 GR00T rollouts from LIBERO Spatial (all 10 tasks).
World model learns manipulation dynamics: eef position, orientation, and gripper state.
Loss dropped 1.403 → 0.008. Imagined reward is finite ✅.
