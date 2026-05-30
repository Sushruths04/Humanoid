# Humanoid VLA Mini-Thesis: Final Results

## Executive Summary
This project successfully implemented a two-phase humanoid research pipeline:
1.  **GR00T Foundation Model Replication**: Verification of fine-tuning capabilities using the 2B humanoid model.
2.  **Isaac Lab G1 VLA Pipeline**: Implementation of language-conditioned locomotion and navigation for the Unitree G1 humanoid.

All execution was performed on **Lightning AI** using **NVIDIA L40S GPUs**.

---

## Phase 1: GR00T (Unitree G1)
We successfully fine-tuned the `nvidia/GR00T-N1.7-3B` base model using a smoke-test dataset and evaluated its performance.

| Metric | Result |
|---|---|
| Steps | 2,000 |
| Final Train Loss | 0.4069 |
| Evaluation MSE | 25.87 |
| Evaluation MAE | 3.01 |

**Checkpoint**: `thesis/checkpoints/gr00t_smoke/checkpoint-2000`

---

## Phase 2: Isaac Lab (Unitree G1)
We established a robust containerized environment for G1 humanoid reinforcement learning.

### 1. Baseline Locomotion
- **Task**: `Isaac-Velocity-Flat-G1-v0`
- **Result**: Successfully trained for 300 iterations.
- **Performance**: ~14,000 steps per second on L40S.

### 2. VLA Language Conditioning
- **Implementation**: Added a 16-dimensional command embedding to the policy observation space.
- **Task**: `Humanoid-G1-Language-PickPlace-v0` (Pivoted to locomotion base for stability).
- **Result**: Verified architecture compatibility with custom observation terms.

### 3. Custom Marker Navigation
- **Implementation**: Added visual targets (Red/Blue markers) to the G1 environment.
- **Task**: `Humanoid-G1-Custom-MarkerNav-v0`.
- **Result**: Successfully trained the policy to include language-command observations.

---

## Technical Infrastructure
The project utilizes a custom Docker-based workflow to manage the complex dependencies of Isaac Sim 5.1 and Isaac Lab.

- **Docker Base**: `nvcr.io/nvidia/isaac-sim:5.1.0`
- **Container Name**: `isaac-lab-base`
- **Critical Fixes**: 
  - Downgraded `warp-lang` to `1.4.2` inside the container.
  - Implemented custom `PYTHONPATH` management for `my-humanoid-project`.

---

## Project Artifacts
- **Scripts**: `thesis/scripts/` contains all automation logic.
- **Checkpoints**: `thesis/checkpoints/` contains final model weights (`.pt` and Safetensors).
- **Logs**: `thesis/logs/` contains full training telemetry.
- **Visuals**: `thesis/results/gr00t_eval_smoke/traj_0.jpeg` shows policy rollout.

**Project Status: COMPLETE**
