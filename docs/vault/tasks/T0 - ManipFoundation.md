---
tags: [task, t0, manipulation, libero, franka, t-track]
---

# T0 — Manipulation Foundation (T-Track)

## Summary

The tabletop manipulation track baseline. Establishes a LIBERO env + manipulation eval harness + a behaviour-cloning baseline. Runs in parallel to the P-track humanoid nav work; shares the `programs/common/` spine.

**Status: COMPLETE ✅** LIBERO installed, BC policy at 50% task_success on libero_spatial:0.

[Full result doc](../../results/t0_manip.md)

---

## Results

| Checkpoint | Metric | Value |
|---|---|---|
| CPT0.1 | LIBERO env up (headless) | ✅ |
| CPT0.2 | Eval harness (random policy) | task_success=0.0% (expected) |
| CPT0.3 | BC baseline (MLPBCPolicy) | **task_success=50.0%** ✅ |

---

## Architecture

| Component | Detail |
|---|---|
| Env | LIBERO `OffScreenRenderEnv` (robosuite + MuJoCo) |
| Task | `libero_spatial:0` — pick bowl, place on plate |
| Policy | `MLPBCPolicy` 2-layer MLP, obs=12, act=7 |
| Obs format | joint_pos(7) + eef_pos(3) + gripper_qpos(2) = 12 |
| Action | OSC_POSE delta (7-dim) |
| Training data | 50 demos × ~100 steps = 5018 transitions |
| BC loss | 0.225 → 0.044 |

---

## What Was Built

- `programs/common/eval/manip_metrics.py` — pure tensor metrics (10 TDD tests)
- `programs/t0_manip_foundation/evaluate_manip.py` — full eval harness (LIBERO wired)
- `programs/t0_manip_foundation/train_bc_libero.py` — LIBERO HDF5 → BC training
- `programs/t0_manip_foundation/bc_baseline.py` — MLPBCPolicy class

---

## Installation Notes (Lightning Studio)

LIBERO is NOT on PyPI. Install in a separate conda env:

```bash
conda create -n libero_env python=3.9 -y
conda run -n libero_env pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
conda run -n libero_env pip install numpy h5py mujoco==2.3.7 robosuite==1.4.1 bddl hydra-core==1.3.2 easydict cloudpickle robomimic gym==0.26.2

# Clone from GitHub (no PyPI package)
git clone --depth=1 https://github.com/Lifelong-Robot-Learning/LIBERO.git /tmp/LIBERO
/path/to/libero_env/bin/pip install /tmp/LIBERO/ --no-deps

# Add to sys.path via .pth file
echo "/tmp/LIBERO" > /path/to/libero_env/lib/python3.9/site-packages/libero_path.pth
```

Verify: `from libero.libero import benchmark; benchmark.get_benchmark_dict()`

Always run with `MUJOCO_GL=egl` (no display on GPU machines).

---

## Remaining Steps (T1+)

- T1: GR00T N1.7 LoRA fine-tuning on LIBERO demos for language-conditioned manipulation
- T2: World model for tabletop manipulation (reuse Dreamer-mini from P2)
- T3: Vision-based manipulation (pixel-only BC or diffusion policy)

---

## Related

- [[P0 - CommandNav]] — shares `programs/common/` eval spine
- [[P2 - World Model]] — T2 will reuse Dreamer-mini for manipulation
- [[Frozen Text Encoder for Language Tasks]] — T1 language-conditioned manip reuses it
- [docs/results/t0_manip.md](../../results/t0_manip.md)
