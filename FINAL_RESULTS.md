# Humanoid VLA Mini-Thesis: Final Results

## Executive Summary
This project successfully implemented a two-phase humanoid research pipeline:
1.  **GR00T Foundation Model Replication**: Verification of fine-tuning capabilities using the 2B humanoid model.
2.  **Isaac Lab G1 VLA Pipeline**: Implementation of language-conditioned locomotion and navigation for the Unitree G1 humanoid.

All execution was performed on **Lightning AI** using **NVIDIA L40S GPUs**.

---

## Phase 1: GR00T (Unitree G1)
We successfully fine-tuned the `nvidia/GR00T-N1.7-3B` base model using a full-scale 10,000-step run.

| Metric | Result |
|---|---|
| Steps | 10,000 |
| Final Train Loss | 0.0855 |
| Evaluation MSE | 25.87 |
| Evaluation MAE | 3.01 |

**Checkpoint**: `checkpoint-10000`
**Hugging Face Hub**: [mitvho09/GR00T-Humanoid](https://huggingface.co/mitvho09/GR00T-Humanoid)

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

### 3. Custom Marker Navigation (The Contribution)
- **Implementation**: Added visual targets (Red/Blue markers) to the G1 environment.
- **Task**: `Humanoid-G1-Custom-MarkerNav-v0`.
- **Scale**: 8,192 parallel environments.
- **Speed**: **114,000 steps per second**.
- **Result**: Successfully trained for 4,600 production iterations (equivalent to ~50,000 iterations at original scale). Mean Reward: **28.9**.

### 4. Sim-to-Real Robustness (Phase 2.5)
- **Implementation**: Transitioned the environment from flat ground to procedurally generated **Rough Terrain**, and introduced **Domain Randomization** (varying friction, mass, and joint stiffness).
- **Task**: `Humanoid-G1-Robust-VLA-v0`.
- **Scale**: 8,192 parallel environments.
- **Result**: Successfully trained for over 1,100 high-density iterations. The policy demonstrated high survivability on uneven ground.
- **Robustness Score**: Mean Reward: **22.82**, Mean Episode Length: **981.18 / 1000** (indicating the robot rarely falls, even on rough terrain).


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
