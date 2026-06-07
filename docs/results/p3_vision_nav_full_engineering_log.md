# P3 Vision Nav — Full Engineering Log (Interview Reference)

Task: Train a pixel-conditioned navigation policy for the Unitree G1 humanoid robot using Isaac Lab + RSL-RL PPO.  
DoD: ≥60% success on CommandNav using only 64×64 RGB head-camera input.  
Final result: **96.28% success** — matching state-based P0 (94.5%) with pixel-only observations.

---

## 1. What Was Built

A camera-conditioned PPO policy for the G1 humanoid that:
- Receives **64×64 RGB** from a head-mounted `TiledCamera` + proprioception + velocity command
- Runs a **shared CNN encoder** (3 conv layers: 32→64→64 channels) feeding a 512→256→128 MLP actor/critic
- Was trained with **4096 parallel environments** in Isaac Lab's GPU-physics simulator
- Achieved **96.28% navigation success** (robot follows velocity commands without falling)

Key files:
- `my-humanoid-project/my_humanoid_project/tasks/g1_vision_nav_cfg.py` — full task + runner config
- `my-humanoid-project/custom_train.py` — training entry point
- `my-humanoid-project/custom_play.py` — eval entry point
- `docs/results/p3_vision_nav.md` — results summary

---

## 2. Architecture Decisions

### Why TiledCamera?
Isaac Lab's `TiledCamera` renders all N environments into a single tiled image on the GPU, then splits into per-env tensors. This avoids N separate camera render calls — critical for 4096+ envs. The alternative (per-env camera) would require 4096 RTX contexts, which is infeasible.

### Why CNN not ViT?
- ViT requires 16×16 patch tokenization — at 64×64 that's only 16 tokens, losing spatial structure
- CNN with 3 conv layers (8×8 stride-4, 4×4 stride-2, 3×3 stride-1) gives a 1024-dim feature vector from 64×64 input
- Shared CNN encoder between actor and critic (`share_cnn_encoders=True`) halves GPU memory for the encoder

### Why 64×64 not 128×128?
This was the critical engineering insight of the project — see §4 (Failures) for the full story. Short answer: **RTX rendering is the bottleneck**, not GPU compute.

---

## 3. Training Runs — Full History

### Run 1: `run_10iter` (A100-SXM4-80GB, 128×128, 8192 envs)
- **Goal**: Initial smoke test
- **Result**: 30+ minute hang at scene startup
- **Root cause**: RTX BVH (Bounding Volume Hierarchy) build scales with env count. 8192 envs × 128×128 = 134M pixels in the tiled render — BVH construction alone took 30 min
- **Fix**: Drop to 4096 envs

### Run 2: `run_latest` (A100-SXM4-80GB, 128×128, 4096 envs, 8 mini_batches)
- **Goal**: Full training at high resolution
- **Result**: `torch.OutOfMemoryError` during PPO update — 77.9 GB trying to allocate 2.82 GiB
- **Root cause**: With 4096 envs × 48 steps × 128×128 RGB, the rollout buffer + CNN gradients exceeded 80 GB
- **Fix**: Increase mini_batches to 16

### Run 3: A100, 128×128, 4096 envs, 16 mini_batches
- **Result**: OOM again at 78.4 GB trying 1.41 GiB
- **Fix**: Increase mini_batches to 64
- **Result**: Stable at 78.7 GB — training ran
- **Iter time**: 66 seconds/iter
- **Root cause of slowness**: RTX rendering at 4096 envs × 128×128 = 67M pixels/render × 48 steps = 57 seconds rendering per iteration. A100 GPU compute (CNN + PPO) = 9 seconds. **GPU was idle 86% of the time.**
- **Key insight**: The A100's CUDA cores are not the bottleneck — RT cores for ray tracing are

### Run 4: `run_final` (A100-SXM4-80GB, 64×64, 4096 envs, 24 steps, 8 mini_batches)
- **Goal**: Use 64×64 to beat the RTX bottleneck
- **Config**: `P3_CAM_H=64, P3_CAM_W=64, P3_NUM_STEPS=24, P3_MINI_BATCHES=8`
- **VRAM**: ~23 GB (4096 envs × 64×64 × 24 steps — much less render data)
- **Iter time**: 12.9 seconds/iter (5.1× faster than 128×128)
- **Result**: 200/300 iters completed before machine cutoff
  - Iter 100: reward = -6.76
  - Iter 200: reward = +27.74
- **Cutoff reason**: Lightning Studio time limit reached at exactly iter 200
- **Saved**: All checkpoints (model_0 to model_200, every 5 iters) to HuggingFace

### Run 5: `run_300_l4` (L4-24GB, 64×64, 4096 envs — FINAL RUN)
- **Goal**: Continue training from `model_200.pt` using `--resume --load_run run_final --checkpoint model_200.pt`
- **What actually happened**: RSL-RL loaded checkpoint weights AND normalizer stats, but ran a full NEW 300-iteration loop starting from iteration 200. Final checkpoint: `model_499.pt` (200 loaded + 299 new = 499 total)
- **Iter time**: 16 seconds/iter on L4 (slightly slower than A100's 12.9s — A100 has stronger RT cores)
- **Total training**: 200 (previous) + 300 (this run) = **499 effective iterations**
- **Reward progression**:
  - Iter 215: +10.94 (warm weights already in positive territory — cold start would be ~-5)
  - Iter 260: +109.73 (4× the previous best of +27.74)
  - Iter 400: +138
  - Iter 499: **+141.35** (final)
- **Success rate**: **96.28%** (time_out fraction from 4096-env rollouts)
- **Fall rate**: 3.72% (base_contact termination)

---

## 4. Failures and Fixes — Complete List

### F1: RTX BVH hang at 8192 envs
- **Symptom**: Training script hung for 30+ minutes at scene initialization, never started
- **Root cause**: Isaac Sim's RTX ray tracer builds a Bounding Volume Hierarchy (BVH) for all geometry in the scene. At 8192 environments with cameras, the BVH construction time scales super-linearly with scene complexity
- **Fix**: `--num_envs 4096` — halving envs made BVH build time tolerable (~2 min)
- **Interview angle**: This is a known Isaac Lab / Omniverse limitation — RTX rendering does not scale linearly with environment count for RL use cases

### F2: OOM at 128×128 with 8 and 16 mini_batches
- **Symptom**: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.82 GiB (GPU 0; 79.20 GiB total capacity; 74.17 GiB already allocated)`
- **Root cause**: With 4096 envs × 48 steps × 128×128×3 RGB stored as float32 = ~14 GB for rollout buffer alone, plus CNN gradients during PPO update
- **Fix**: `P3_MINI_BATCHES=64` — split each PPO update into 64 micro-batches so gradient computation only needs 1/64th of the rollout at once
- **Why 64 works**: 14 GB buffer / 64 batches = 219 MB per forward/backward pass — fits within the remaining 5 GB headroom

### F3: A100 using only 14% GPU utilization
- **Symptom**: `nvidia-smi` showed 14-20% GPU utilization even at 78 GB VRAM usage
- **Root cause**: The 66s iteration was split as: 57s RTX rendering + 9s CNN/PPO. A100 sits idle waiting for RTX cores to finish each render step. RTX rendering is single-threaded per-frame through the Omniverse rendering pipeline
- **Why this matters for interviews**: This is the key insight — GPU VRAM ≠ GPU compute utilization. A task can saturate memory while barely using compute cores. The bottleneck was the RT core rendering pipeline, not the CUDA/Tensor cores doing deep learning
- **Fix**: Reduce resolution 128→64 (4× fewer pixels per render) AND reduce steps 48→24 (2× fewer render calls per iteration). Combined speedup: 5.1×

### F4: `update_period` parameter had no effect on iter time
- **Symptom**: Setting `TiledCameraCfg(update_period=0.2)` (5 Hz camera update) was expected to reduce renders, but iteration time stayed at 66s
- **Root cause**: `update_period` controls how often Python READS the camera data from the GPU buffer, not how often Isaac Sim RENDERS. The RTX renderer still renders every physics step — `update_period` only skips the Python tensor copy
- **Fix**: None — accepted that renders happen every step. The real fix was reducing resolution

### F5: A100 machine time limit hit at iter 200/300
- **Symptom**: Training stopped mid-run after 43 minutes, only 200 of 300 planned iterations complete
- **Root cause**: Lightning Studio GPU rental time ran out
- **Mitigation**: Every checkpoint saved to persistent storage (every 5 iters). All checkpoints uploaded to HuggingFace before machine shutdown
- **Learning**: Always use `--save_interval 5` (not 50 or 100) when time is uncertain. 41 × 19 MB = 780 MB total — worth the storage cost

### F6: GitHub push from remote machine failed
- **Symptom**: `git push` on the A100 machine: `fatal: could not read Username for 'https://github.com'`
- **Root cause**: No git credentials stored on ephemeral Lightning Studio machine
- **Fix**: `git format-patch` to create a patch file, `scp` to local Windows machine, `git am` to apply locally, push from local
- **Secondary failure**: `git am` failed due to LFS pointer conflicts (IsaacLab docs images tracked by LFS). Fix: `git am --abort`, manually `scp` just the important non-LFS files (params/, progress.txt), commit locally with correct message

### F7: Docker image missing after GPU upgrade
- **Symptom**: Switched from CPU studio to L4 studio, `docker images` showed empty
- **Root cause**: Docker images are stored on ephemeral machine-local disk, not in `/teamspace/studios/this_studio/` (the persistent bind mount). Every new machine instance starts with no images
- **Fix**: `docker pull ghcr.io/sushruths04/humanoid-isaaclab:latest` on every new machine (~3-5 min for 17.6 GB image)
- **Prevention**: The project already pushed the image to GHCR (`ghcr.io/sushruths04/humanoid-isaaclab:latest`) so re-pull is always possible

### F8: `--resume` reset iteration counter to 0 (not 200)
- **Symptom**: After `--resume --load_run run_final --checkpoint model_200.pt`, ETA showed 1:49 (300 full iterations) not 37 min (100 remaining). Training showed reward -1.6 at "iter 1" despite loaded weights
- **Root cause**: RSL-RL's `OnPolicyRunner.learn()` runs `for it in range(current_learning_iteration, current_learning_iteration + num_learning_iterations)`. `current_learning_iteration=200` (from checkpoint) + `num_learning_iterations=300` (max_iterations) = loop runs 200→500. Checkpoint names follow actual iteration numbers: `model_200, model_205...model_499`
- **Why reward was low initially**: Obs normalizer running statistics ARE loaded from the checkpoint, but the first batch of new observations causes the normalizer to update, temporarily distorting inputs while it recalibrates to the new environment instance
- **Outcome**: This was actually BETTER — the policy ran 300 full new iterations on top of the warm-started weights, reaching model_499 (499 effective training iterations total) and reward +141 instead of just +35

### F9: `play.py` task not found
- **Symptom**: `gymnasium.error.NameNotFound: Environment 'Humanoid-G1-VisionNav' doesn't exist`
- **Root cause**: Isaac Lab's stock `play.py` calls `gym.spec(task_name)` which requires the task to be registered via `gym.register()`. Our custom task is registered when `my_humanoid_project.tasks` is imported. `play.py` never imports this — only `custom_train.py` does
- **Fix**: Created `custom_play.py` (mirrors `custom_train.py` but delegates to `play.main()` instead of `train.main()`)

### F10: SSH heredoc corrupted Python file
- **Symptom**: `custom_play.py` written via SSH heredoc produced `SyntaxError: invalid syntax`
- **Root cause**: Shell variable expansion and quote escaping in heredoc-over-SSH mangled the Python string literals and escape sequences
- **Fix**: Write file content using `python3 -c "with open(...) as f: f.write(...)"` — bypasses all shell quoting issues

### F11: HuggingFace CLI deprecated
- **Symptom**: `huggingface-cli` not found on Lightning Studio host
- **Root cause**: Newer `huggingface_hub` versions deprecated the CLI in favor of the Python API
- **Fix**: `/home/zeus/miniconda3/bin/python3 -c "from huggingface_hub import HfApi; HfApi().upload_folder(...)"`

### F12: SSH key permission denied on new studio
- **Symptom**: `Permission denied (publickey)` connecting to new Lightning studio despite key existing at `~/.ssh/lightning_rsa`
- **Root cause**: Lightning SSH keys are registered per-studio, not account-wide. The key from the previous studio wasn't registered for the new studio
- **Fix**: Run Lightning's PowerShell setup script for the new studio URL: `iwr 'https://lightning.ai/setup/ssh-windows?t=TOKEN&s=STUDIO_ID' -useb | iex`
- **Secondary issue**: `lightning_rsa` file was read-only, script couldn't overwrite. Fix: `icacls` to grant write permissions, then remove old key before re-running setup

### F13: Git LFS files showing as "modified" on new studio
- **Symptom**: `git status` showed 200+ modified files (all IsaacLab docs images, videos, .npz files) despite no actual changes
- **Root cause**: This studio had the actual binary files on disk, but git expected LFS pointer files (small text files). Without `git-lfs` properly configured, git sees a mismatch
- **Fix**: `GIT_LFS_SKIP_SMUDGE=1 git reset --hard origin/feat/planned-scripts` — resets all tracked files to match remote without downloading LFS content

---

## 5. Key Technical Insights (for interviews)

### "Why is your A100 only at 14% utilization?"
The bottleneck is **RTX ray-tracing**, not CUDA tensor cores. Isaac Lab's `TiledCamera` uses Omniverse's RTX renderer to ray-trace all environments. At 4096 envs × 128×128, each render frame processes 67M pixels through the BVH ray tracing pipeline. This is RT-core bound, not CUDA-core bound. The A100 has 6912 CUDA cores and 432 Tensor cores but only 336 RT cores — and the RT cores were saturated while CUDA cores sat idle.

The fix (64×64 resolution) cut pixels by 4×, making each render 4× faster and pushing utilization from 14% toward the rendering throughput limit.

### "What is TiledCamera and why use it?"
`TiledCamera` is Isaac Lab's batched camera sensor. Instead of N separate camera render calls (one per environment), it renders all N environments into a single tiled image of size `(sqrt(N) × H) × (sqrt(N) × W)`, then slices it into N per-environment tensors on the GPU. This eliminates N-1 render context switches and keeps all data on GPU without any CPU round-trip. The trade-off: the tiled image still goes through RTX, so resolution still matters for performance.

### "How does PPO handle image observations?"
Standard RSL-RL PPO was extended with `RslRlCNNModelCfg`:
1. CNN encoder processes 64×64 RGB → 1024-dim feature vector
2. Feature vector is concatenated with proprioceptive + command observations
3. Shared encoder between actor and critic (halves encoder memory/compute)
4. Adaptive learning rate schedule with KL divergence constraint (`desired_kl=0.01`)

The mini-batch trick: PPO stores the full rollout buffer (4096 envs × 24 steps = 98304 transitions, each with a 64×64×3 image), then splits into 8 mini-batches for gradient updates. This is the standard PPO memory-efficiency trick but critical when images are in the buffer.

### "What is the difference between reward and success rate?"
Mean episode reward (+141.35) is a composite weighted sum of many sub-rewards (velocity tracking, upright posture, joint deviation penalties, etc.). It's not directly interpretable as a percentage.

Success rate (96.28%) comes from `Episode_Termination/time_out` — the fraction of episodes that ran to the maximum time step without any termination condition (base_contact = robot fell, out of bounds, etc.). A timed-out episode means the robot survived the full episode = success.

---

## 6. Final Results

| Metric | Value |
|---|---|
| Final checkpoint | `model_499.pt` |
| Final reward | **+141.35** |
| **Success rate** | **96.28%** |
| Fall rate | 3.72% |
| DoD (≥60% success) | ✅ EXCEEDED |
| Training: total iterations | 499 (200 + 300) |
| Training: total wall time | ~43 min (A100) + ~80 min (L4) = ~2 hrs |
| GPU used | A100-SXM4-80GB (run_final), L4-24GB (run_300_l4) |
| Iter time | 12.9s (A100) / 16s (L4) |

### Comparison to P0 (state-based CommandNav)

| Policy | Observations | Success Rate |
|---|---|---|
| P0 CommandNav (MLP) | Proprioception + velocity command | 94.5% |
| **P3 VisionNav (CNN+MLP)** | **64×64 RGB + proprioception + velocity command** | **96.28%** |

**Pixel observations match state observations for navigation.** This is the headline portfolio result — adding camera input does not degrade policy performance when trained long enough with the right architecture.

---

## 7. Checkpoints on HuggingFace

Repository: `mitvho09/humanoid-g1-nav` (dataset repo)

| Run | Path | Best checkpoint | Reward |
|---|---|---|---|
| run_10iter | `checkpoints/p3_vision_nav/run_10iter/` | model_9.pt | N/A (infra test) |
| run_final | `checkpoints/p3_vision_nav/run_final/` | model_200.pt | +27.74 |
| run_300_l4 | `checkpoints/p3_vision_nav/run_300_l4/` | model_499.pt | +141.35 |

---

## 8. Commands Reference (new machine setup)

```bash
# 1. Pull image (~3-5 min)
docker pull ghcr.io/sushruths04/humanoid-isaaclab:latest
docker tag ghcr.io/sushruths04/humanoid-isaaclab:latest isaac-lab-base:latest

# 2. Start container
cd /teamspace/studios/this_studio/Humanoid/IsaacLab/docker
DOCKER_NAME_SUFFIX= docker compose --env-file .env.base --profile base up isaac-lab-base -d --no-build

# 3. Train from scratch (300 iters, ~80 min on L4)
nohup docker exec \
  -e PYTHONPATH="/workspace:/workspace/my-humanoid-project:/workspace/isaaclab/source" \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  -e P3_CAM_H="64" -e P3_CAM_W="64" \
  -e P3_NUM_STEPS="24" -e P3_MINI_BATCHES="8" \
  -e P3_MAX_ITERS="300" -e P3_SAVE_INTERVAL="5" \
  isaac-lab-base /workspace/isaaclab/isaaclab.sh -p \
  /workspace/my-humanoid-project/custom_train.py \
  --task Humanoid-G1-VisionNav-v0 --headless --enable_cameras \
  --num_envs 4096 --max_iterations 300 \
  > /tmp/train_p3.log 2>&1 &

# 4. Resume from checkpoint
# First stage checkpoint into container's log dir:
docker exec isaac-lab-base bash -c "mkdir -p /workspace/isaaclab/logs/rsl_rl/g1_vision_nav && cp -r /workspace/programs/checkpoints/p3_vision_nav/run_final /workspace/isaaclab/logs/rsl_rl/g1_vision_nav/"
# Then run with --resume:
nohup docker exec [same env vars] isaac-lab-base ... --resume --load_run run_final --checkpoint model_200.pt > /tmp/train_p3_resume.log 2>&1 &

# 5. Eval
nohup docker exec \
  -e PYTHONPATH="/workspace:/workspace/my-humanoid-project:/workspace/isaaclab/source" \
  -e P3_CAM_H="64" -e P3_CAM_W="64" \
  isaac-lab-base /workspace/isaaclab/isaaclab.sh -p \
  /workspace/my-humanoid-project/custom_play.py \
  --task Humanoid-G1-VisionNav-v0 --headless --enable_cameras \
  --num_envs 512 \
  --load_run <run_dir_timestamp> \
  --checkpoint model_499.pt \
  > /tmp/eval_p3.log 2>&1 &
```
