---
tags: [failure, docker, container, rebuild, isaac-lab]
---

# container.py Forces Rebuild

## Symptom
Running `python IsaacLab/docker/container.py start` fails with an error about missing base image `nvcr.io/nvidia/isaac-sim:5.1.0` or attempts a 30+ minute rebuild from scratch.

## Root Cause
`container.py start` always passes `--build` to docker compose, which forces a rebuild of the image from the NGC (NVIDIA GPU Cloud) base. To use that base, you need an NGC account and login. On Lightning AI (or any non-NVIDIA host), this is unavailable.

The purpose of the pre-built `ghcr.io/sushruths04/humanoid-isaaclab:latest` image is to avoid this — it already contains everything needed. But `container.py` bypasses that.

## Fix: Use docker compose directly with --no-build

```bash
cd IsaacLab/docker
touch .isaac-lab-docker-history
DOCKER_NAME_SUFFIX= docker compose --env-file .env.base --profile base up isaac-lab-base -d --no-build
```

Breaking it down:
- `touch .isaac-lab-docker-history` — compose checks for this file (references it as the history volume source)
- `DOCKER_NAME_SUFFIX=` — the env var must be set (even to empty) or compose complains about quoting
- `--no-build` — tell compose to use the already-pulled image, never build
- `-d` — detached (background)

## Lesson
When a helper script insists on doing something you don't want, bypass it and use the underlying tool directly. `docker compose` is the underlying tool; `container.py` is a wrapper. If the wrapper doesn't work for your use case, drop to raw compose.

## Related
- [[Isaac Sim Docker Container]]
- [[GHCR Image & Auth]]
