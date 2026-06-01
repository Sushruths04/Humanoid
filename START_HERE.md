# Start Here

This is the single runbook for this project.

Use it when you:

- set up a new Lightning machine
- switch from one machine to another
- reuse the prebuilt Docker image
- run the Vision-VLA smoke test
- decide where code and large files should go

## What Goes Where

- GitHub: code, scripts, small markdown docs, runbooks.
- Hugging Face: large checkpoints, datasets, result artifacts that are too big or awkward for GitHub.
- Docker registry: the reusable Isaac Lab image.
- Local machine: only temporary runtime state.

## One-Time Setup On A New Machine

1. Open the Lightning AI terminal.
2. Clone or update the repo:

```bash
git clone https://github.com/Sushruths04/Humanoid.git
cd Humanoid
```

3. Login to GHCR if you want to reuse the Docker image:

```bash
docker login ghcr.io
```

4. Pull the reusable image:

```bash
REGISTRY_IMAGE=ghcr.io/sushruths04/humanoid-isaaclab:latest bash thesis/scripts/docker_image_portability.sh pull
```

5. Confirm the repo is current:

```bash
git pull --rebase --autostash origin main
```

## Daily Workflow

1. Sync code from GitHub.
2. Pull or reuse the Docker image.
3. Run the small Vision-VLA smoke test first.
4. Only then scale to a longer run.
5. Push code back to GitHub.
6. Push large outputs to Hugging Face or GHCR as appropriate.

## Vision-VLA Smoke Test

Always start small. This confirms the pipeline reaches PPO.

```bash
NUM_ENVS=16 MAX_ITERS=2 TRAIN_TIMEOUT_MINUTES=20 VLA_CAMERA_HEIGHT=32 VLA_CAMERA_WIDTH=32 bash thesis/scripts/30_vision_vla.sh
```

If that succeeds, increase gradually. Do not jump straight to 1024+ camera envs unless you specifically want to test scaling.

## Machine Switch Procedure

When moving to another machine:

1. Commit and push your code changes to GitHub.
2. Save checkpoints and large outputs to Hugging Face.
3. Push the Docker image to GHCR, or use the tarball fallback.
4. On the new machine, clone the repo.
5. Pull the Docker image.
6. Run the smoke test.
7. Resume the longer training job only after the smoke test passes.

## Docker Image Reuse

The reusable local image tag is `humanoid-isaaclab:latest`.

### Push To GHCR

```bash
docker login ghcr.io
cd /home/zeus/content/Humanoid
REGISTRY_IMAGE=ghcr.io/sushruths04/humanoid-isaaclab:latest bash thesis/scripts/docker_image_portability.sh push
```

### Pull From GHCR

```bash
docker login ghcr.io
cd Humanoid
REGISTRY_IMAGE=ghcr.io/sushruths04/humanoid-isaaclab:latest bash thesis/scripts/docker_image_portability.sh pull
```

### Tarball Fallback

If registry push is not available:

```bash
bash thesis/scripts/docker_image_portability.sh save
bash thesis/scripts/docker_image_portability.sh load
```

Current fallback tarball on the Lightning machine:

```bash
/home/zeus/content/humanoid-isaaclab-latest.tar
```

## Push Code

When your code changes are ready:

```bash
git add .
git commit -m "Describe the change"
git push
```

## Save Large Outputs

Use Hugging Face for large files that should survive machine changes.

Typical examples:

- checkpoints
- dataset exports
- rollout videos
- logs larger than a simple markdown summary

## Common Mistakes To Avoid

- Do not store secrets in files or in Git.
- Do not rely on the local machine for long-term storage.
- Do not start a long run before a smoke test passes.
- Do not assume a camera-based run behaves like a pure physics run.
- Do not rebuild the Docker image every time if a reusable image already exists.

## Useful Files

- [README.md](./README.md)
- [DOCKER_IMAGE_REUSE.md](./DOCKER_IMAGE_REUSE.md)
- [REMOTE_WORKFLOW.md](./REMOTE_WORKFLOW.md)
- [MACHINE_SWITCH_QUICK_REF.md](./MACHINE_SWITCH_QUICK_REF.md)
- [CPU_TO_GPU_MACHINE_SWITCH.md](./CPU_TO_GPU_MACHINE_SWITCH.md)

## Current Known Good State

- GitHub contains the current code and docs.
- Lightning has the reusable Docker image saved and pushed to GHCR.
- The Vision-VLA smoke test has already reached PPO successfully.
- The current image tag is `ghcr.io/sushruths04/humanoid-isaaclab:latest`.
