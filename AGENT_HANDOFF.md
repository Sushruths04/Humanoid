# Agent Handoff Document: Humanoid VLA Mini-Thesis

**Target Agent**: CodeGemma / Codex / Next AI Assistant
**Current Date**: May 31, 2026
**Project Repo**: [https://github.com/Sushruths04/Humanoid](https://github.com/Sushruths04/Humanoid)

---

## 1. Project Context & Completed Work
You are taking over a robotics research project focused on Vision-Language-Action (VLA) models for humanoid robots (specifically the Unitree G1). The project uses **Isaac Lab (Isaac Sim 5.1)** for RL and **Hugging Face** for foundation models.

### What is 100% Complete:
*   **Phase 1 (GR00T Foundation Model)**: Successfully fine-tuned the `nvidia/GR00T-N1.7-3B` model for 10,000 steps. The 8.8GB weights are stored on Hugging Face (`mitvho09/GR00T-Humanoid`).
*   **Phase 2 (G1 Locomotion & VLA Base)**: Built a custom environment (`Humanoid-G1-Custom-MarkerNav-v0`) where a G1 humanoid navigates to Red/Blue markers based on a 16-dimensional Language Command embedding.
*   **Phase 2.5 (Sim-to-Real Robustness)**: Successfully trained the language-conditioned G1 on rough, procedurally generated terrain with Domain Randomization (friction/mass).
*   **All Code & Checkpoints (`.pt`)**: Are fully synced to the GitHub repository.

---

## 2. The Current Blocker (Why you are here)
The previous agent attempted to start **Phase 3 (Vision VLA)**, which involves adding a head camera to the G1 so the policy can observe actual RGB pixels instead of just joint coordinates.

*   **The Code Exists**: The previous agent wrote `my_humanoid_project/tasks/g1_vla_vision_cfg.py` and registered `Humanoid-G1-Vision-VLA-v0`.
*   **The Failure**: The execution was blocked because the previous Cloud GPU machine (Lightning AI) had a restricted Docker environment. It refused to pass the **Vulkan/Graphics drivers** (`libGLX_nvidia.so.0`) into the Isaac Lab container, causing the simulation to crash when attempting to render the camera.

---

## 3. Your Objective (The Next Task)
The human user is setting up a **new, fully unlocked GPU machine**. Your job is to set up the environment and execute the Vision VLA training.

### Prompt to execute immediately upon taking control:

```markdown
Hello Agent. We are resuming the Humanoid VLA project on a new GPU machine. 

Please execute the following steps strictly in order:

### Step 1: Environment Restoration
1. Clone the repo: `git clone https://github.com/Sushruths04/Humanoid.git`
2. Navigate to `IsaacLab/docker/` and ensure `docker-compose.yaml` has `capabilities: [ gpu, graphics, compute, utility, video ]` set in the deploy block.
3. Run `python3 container.py build` then `python3 container.py start`.
4. CRITICAL: Exec into the container and run `pip install warp-lang==1.4.2` (G1 physics will crash without this exact downgrade).

### Step 2: Graphics Verification
1. Run an Isaac Sim graphics test inside the container to prove Vulkan is working:
   `docker exec isaac-lab-base vulkaninfo --summary`
2. If this fails, the host machine's `nvidia-container-toolkit` is not configured correctly. DO NOT proceed to training until graphics are working.

### Step 3: Train Phase 3 (Vision VLA)
1. The code is in `my-humanoid-project/my_humanoid_project/tasks/g1_vla_vision_cfg.py`.
2. The orchestrator script is `thesis/scripts/30_vision_vla.sh`.
3. Review the orchestrator. It is currently set for 32 environments and 300 iterations (Smoke test). 
4. Execute: `bash thesis/scripts/30_vision_vla.sh`
5. Once the smoke test passes, scale `NUM_ENVS` to max VRAM capacity and train until convergence.
```

---

## 4. Crucial Technical Notes for the New Agent
*   **The Workspace**: The project relies on a custom python package `my-humanoid-project`. When running `docker exec`, you MUST mount or export the `PYTHONPATH` so Isaac Lab can find it: `-e PYTHONPATH="/workspace/my-humanoid-project:/workspace/isaaclab/source"`.
*   **The Trainer**: Do NOT use the default RSL-RL `train.py`. The previous agent created a custom entry point at `my-humanoid-project/custom_train.py` that handles custom Gymnasium registrations before Isaac Sim initializes.
*   **Headless vs Rendering**: Standard RL (Phase 2) used `isaaclab.python.headless.kit`. For Phase 3 (Vision), you MUST use `--experience /workspace/isaaclab/apps/isaaclab.python.headless.rendering.kit` to enable the camera sensors.

**End of Handoff Document.**
