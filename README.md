# Humanoid Thesis Workspace

This repository contains the Isaac Lab humanoid thesis work, including the language-conditioned and vision-conditioned G1 tasks, remote GPU launch scripts, and the machine-switch workflow for Lightning AI.

## What This Repo Does

- Runs the G1 humanoid tasks in Isaac Lab.
- Trains the Vision-VLA smoke test and the longer PPO runs.
- Supports remote execution over SSH on Lightning AI.
- Reuses a prebuilt Docker image across machines.
- Stores large artifacts outside GitHub when needed.

## Main Entry Points

- `my-humanoid-project/custom_train.py` - training entry point.
- `my-humanoid-project/my_humanoid_project/tasks/g1_vla_vision_cfg.py` - vision task config.
- `thesis/scripts/30_vision_vla.sh` - vision smoke/training launcher.
- `thesis/scripts/machine_switch.sh` - remote machine workflow helper.
- `thesis/scripts/docker_image_portability.sh` - Docker save/load/push/pull helper.
- `DOCKER_IMAGE_REUSE.md` - Docker portability notes.
- `REMOTE_WORKFLOW.md` - remote GPU workflow notes.
- `MACHINE_SWITCH_QUICK_REF.md` - short operational cheat sheet.

## Recommended Workflow

### 1. Use a remote GPU machine

Connect to Lightning AI by SSH and run the project there. The current workflow is designed for:

- no local training
- no local artifact storage
- GitHub for code
- Hugging Face for large datasets, checkpoints, or result artifacts
- Docker registry for reusable images

### 2. Start with a small vision smoke test

Before a long run, confirm the pipeline reaches PPO:

```bash
NUM_ENVS=16 MAX_ITERS=2 TRAIN_TIMEOUT_MINUTES=20 VLA_CAMERA_HEIGHT=32 VLA_CAMERA_WIDTH=32 bash thesis/scripts/30_vision_vla.sh
```

### 3. Scale up only after the smoke test passes

The vision task is expensive because it creates a camera per environment. Large env counts can stall Isaac Sim during initialization before PPO starts.

## Docker Reuse Across Machines

The reusable image is tagged on the machine as `humanoid-isaaclab:latest`.

### Push to GHCR

On the machine that already has the image:

```bash
docker login ghcr.io
cd /home/zeus/content/Humanoid
REGISTRY_IMAGE=ghcr.io/sushruths04/humanoid-isaaclab:latest bash thesis/scripts/docker_image_portability.sh push
```

### Pull on a new machine

On the next machine:

```bash
docker login ghcr.io
cd Humanoid
REGISTRY_IMAGE=ghcr.io/sushruths04/humanoid-isaaclab:latest bash thesis/scripts/docker_image_portability.sh pull
```

### Tarball fallback

If registry push is not available, you can save and load the image tarball:

- Saved tarball on the current Lightning machine: `/home/zeus/content/humanoid-isaaclab-latest.tar`
- Save:

```bash
bash thesis/scripts/docker_image_portability.sh save
```

- Load:

```bash
bash thesis/scripts/docker_image_portability.sh load
```

## Important Notes

- Do not commit GitHub, Docker, or Hugging Face tokens.
- Use GitHub for code and small text artifacts.
- Use Hugging Face for large checkpoints or result files.
- Rebuilds are expensive; prefer pulling the prebuilt Docker image first.
- The vision trainer now fails fast if camera env counts are too large unless explicitly overridden.

## Files To Read First

- [DOCKER_IMAGE_REUSE.md](./DOCKER_IMAGE_REUSE.md)
- [REMOTE_WORKFLOW.md](./REMOTE_WORKFLOW.md)
- [MACHINE_SWITCH_QUICK_REF.md](./MACHINE_SWITCH_QUICK_REF.md)

