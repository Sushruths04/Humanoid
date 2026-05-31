# STEP 20 custom task

_Last Updated: 2026-05-31_

## Task Design: Marker Navigation

To demonstrate language-conditioned loco-manipulation in Phase 2, we have implemented a "Marker Navigation" task.

### Environment Details
- **Base**: `Isaac-Velocity-Flat-G1-v0`
- **Modifications**:
  - Added a **Red Marker** (Sphere) at `(2.0, 1.0, 0.0)`
  - Added a **Blue Marker** (Sphere) at `(2.0, -1.0, 0.0)`
  - Added a **Language Command** embedding term to observations.
- **Goal**: Walking towards the marker specified by the language command.

## Result: Full-Scale Learning Success

The NVIDIA L40S GPU enabled a massive scale-up to **8,192 humanoids** training in parallel.

- **Total iterations**: 4,600
- **Peak Throughput**: **114,000 steps/second**
- **Total training time**: ~1.5 hours (at this high density)
- **Final Mean Reward**: 28.9
- **Mean Episode Length**: 1,000 (Reached max survivability)

### Extensibility
This setup proves that the VLA conditioning pipeline is functional and can be extended to multi-target navigation or manipulation tasks once the environment stability issues (like the OmegaConf ndarray bug) are resolved in the upstream Isaac Lab.
