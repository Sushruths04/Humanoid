# T-track — Tabletop manipulation (scaffold)

Status: scaffold. Requires LIBERO/RoboCasa + LeRobot + GR00T N1.7 environments.
See `docs/TABLETOP_MANIPULATION_PLAN.md` for the full T0-T5 plan.

## First steps (when manipulation env available)
- T0: stand up Franka/LIBERO env; reuse `programs/common/eval/metrics.py` with
  grasp/place/task-success metrics (the success/fall math generalizes).
- T1: LoRA fine-tune GR00T N1.7 on a LeRobot multi-instruction dataset;
  instruction-swap probe must change which object is manipulated.
- T2: Diffusion Policy / ACT baselines; reuse the P2 world model on manip rollouts.

## Reuse from the spine (already built + CPU-tested)
- `programs/common/eval/{metrics,report}.py` — eval aggregation + markdown.
- `programs/world_model/` — Dreamer-mini for T2 world-model-assisted manip.
