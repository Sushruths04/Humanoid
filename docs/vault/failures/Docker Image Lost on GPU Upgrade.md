---
tags: [failure, docker, lightning-studio, gpu, setup, p3]
---

# Docker Image Lost on GPU Upgrade

## Symptom
Upgraded Lightning Studio from CPU plan to L4 GPU. Ran docker compose, got:
```
Error response from daemon: No such image: isaac-lab-base:latest
```
Then checked:
```bash
docker images
# REPOSITORY   TAG   IMAGE ID   CREATED   SIZE
# (empty)
```
The 17.6 GB image that took 5 minutes to pull was completely gone.

## Root Cause
Docker images are stored on **ephemeral machine-local disk**, not in `/teamspace/studios/this_studio/` (the persistent bind mount). When Lightning switches you between machine types (CPU → L4 → A100), it provisions a **new underlying VM**. That VM starts with a blank Docker image cache.

The persistent bind mount only covers:
```
/teamspace/studios/this_studio/  →  survives machine type changes ✓
/tmp/                            →  ephemeral ✗
Docker image cache               →  ephemeral ✗
Docker named volumes             →  ephemeral ✗ (isaac-lab-logs, etc.)
```

## Fix: Re-pull on every new machine
```bash
docker pull ghcr.io/sushruths04/humanoid-isaaclab:latest
docker tag ghcr.io/sushruths04/humanoid-isaaclab:latest isaac-lab-base:latest
```
~3-5 minutes on Lightning's fast network. Always do this before starting the container.

## Also Lost: Docker Named Volumes
The RSL-RL training logs (`isaac-lab-logs` Docker volume) are also wiped. This means any checkpoints saved to `/workspace/isaaclab/logs/` inside the container are gone. 

**Always copy checkpoints to `/workspace/programs/` (the bind mount) before stopping a machine.**

## Standard New-Machine Checklist
```bash
# 1. Pull image
docker pull ghcr.io/sushruths04/humanoid-isaaclab:latest
docker tag ghcr.io/sushruths04/humanoid-isaaclab:latest isaac-lab-base:latest

# 2. Start container
cd /teamspace/studios/this_studio/Humanoid/IsaacLab/docker
touch .isaac-lab-docker-history  # required — compose bind mounts this file
DOCKER_NAME_SUFFIX= docker compose --env-file .env.base --profile base up isaac-lab-base -d --no-build

# 3. Verify
docker exec isaac-lab-base nvidia-smi
```

## Do Not
- Never assume a Docker image persists across machine type changes
- Never store important outputs only inside the container — always bind mount or `docker cp` to persistent storage

## Related
- [[Isaac Sim Docker Container]]
- [[Lightning Studio Environment]]
- [[Results Lost to Ephemeral Container Storage]]
