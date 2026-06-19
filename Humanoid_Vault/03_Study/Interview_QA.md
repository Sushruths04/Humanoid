---
tags: [study, interview]
---

# Interview Q&A

Practice defending the project. Answers are grounded in [[Results_Summary]] — **never overclaim**.

### Q: What is this project in one line?
PPO-trained language/vision-conditioned locomotion for the Unitree G1 in Isaac Lab, plus GR00T foundation-model fine-tuning, run on Lightning AI L40S GPUs.

### Q: What's your strongest, fully-verifiable result?
A G1 humanoid trained on **rough terrain** with **±25% joint stiffness/damping domain randomization** stays upright **~98% of the episode (981/1000 steps)** at mean reward **22.8**. Backed by `logs/g1_robust/train.log` + a saved checkpoint. See [[Phase2.5_Sim2Real_Robust]].

### Q: Is the "language" part a real language model?
**No — and I'm precise about that.** It's a deterministic 16-dim SHA256 hash embedding of fixed phrases, used as a fixed-size command slot. It validates the observation interface and is designed for a frozen text encoder drop-in. The policy isn't yet conditioning on varying language. See [[Phase2_G1_Locomotion_and_Language]].

### Q: Why hash embeddings instead of CLIP?
CPU-safety and determinism during bring-up — no network dependency, reproducible, and it isolates the RL plumbing from encoder choice. The interface (`embedding_for_text`) is a clean swap point for CLIP/SentenceTransformer later.

### Q: How does vision work, and what was the blocker?
A `TiledCamera` on the head feeds a Nature-CNN encoder shared by actor & critic, then PPO. The blocker was Vulkan (`libGLX_nvidia.so.0`) in the container; fixed by running **headless with the rendering kit + `--enable_cameras`** at low resolution. It runs to PPO and scaled to 2048 envs — but isn't trained to convergence yet. See [[Phase3_Vision_VLA]].

### Q: Why PPO and not offline/model-based RL?
Massively parallel sim (8k+ envs, 100k+ steps/s) makes on-policy PPO sample-cheap and stable; RSL-RL's GPU PPO is the standard for Isaac Lab locomotion. See [[PPO_for_Locomotion]].

### Q: What does the 981/1000 episode length mean?
Episodes end early on a fall. ~981/1000 ⇒ the robot almost never falls across an episode, even on rough terrain with randomized actuators — direct evidence the robustness training generalized.

### Q: How would you make this a real VLA?
Three steps: (1) randomize commands per episode + command-dependent rewards so language is informative; (2) multi-goal + sequential grounding; (3) teacher→student distillation from privileged state to camera vision. See [[Open_Questions_and_Next_Steps]].

### Q: What's the GR00T MSE of 25.9 — is that good?
It's an action-regression error on a tiny 5-trajectory eval, demonstrating the fine-tune+eval harness works — not a SOTA or task-success claim. See [[Phase1_GR00T]].

### Q: Biggest weakness of the project?
Results aren't all reproducible from the repo — only locomotion/GR00T checkpoints are saved; nav/vision numbers are reported in prose. Fixing that (save everything, add success-rate metrics) is Step 0 of my plan.
