---
tags: [setup, docker, isaac-sim, container]
---

# Isaac Sim Docker Container

## The Exact Bring-Up Sequence

Copy-paste this every time you get a fresh GPU machine:

```bash
cd /teamspace/studios/this_studio/Humanoid

# Step 1: Pull and tag the image (Docker storage is ephemeral)
docker pull ghcr.io/sushruths04/humanoid-isaaclab:latest
docker tag  ghcr.io/sushruths04/humanoid-isaaclab:latest isaac-lab-base

# Step 2: Create required history file (compose needs it)
cd IsaacLab/docker
touch .isaac-lab-docker-history

# Step 3: Start with compose — NOT container.py (see [[container.py Forces Rebuild]])
DOCKER_NAME_SUFFIX= docker compose --env-file .env.base --profile base up isaac-lab-base -d --no-build
cd ../..

# Step 4: Verify
docker ps --format "{{.Names}} | {{.Status}} | {{.Image}}"
# expected: isaac-lab-base | Up X seconds | isaac-lab-base
```

> **NEVER use `container.py start`** — it forces a full rebuild needing the private NGC base image. See [[container.py Forces Rebuild]].

---

## Verify Everything Inside the Container

```bash
# GPU is visible inside container?
docker exec isaac-lab-base nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Bind mounts are present?
docker exec isaac-lab-base ls /workspace/programs/scripts/
docker exec isaac-lab-base ls /workspace/my-humanoid-project/

# Expected output: train_eval_nav.sh, batch_test_nav.sh, ...
```

---

## Bind Mounts (what's connected to the host)

These are patched into `IsaacLab/docker/docker-compose.yaml`:

| Host path | Container path | Purpose |
|---|---|---|
| `programs/` | `/workspace/programs` | Rewards, commands, sequence, eval harness, world model |
| `my-humanoid-project/` | `/workspace/my-humanoid-project` | Task configs, custom_train.py, custom_play.py |

> **`docs/` is NOT mounted.** Writing results to `docs/results/` inside the container loses them when the container is removed. See [[Results Lost to Ephemeral Container Storage]].

---

## Running Code Inside the Container

```bash
# Pattern for ANY in-container python script:
docker exec \
  -e PYTHONPATH="/workspace:/workspace/my-humanoid-project:/workspace/isaaclab/source" \
  isaac-lab-base \
  /workspace/isaaclab/isaaclab.sh -p /workspace/my-humanoid-project/custom_train.py \
  --task Humanoid-G1-CommandNav-v0 --headless --num_envs 4096 --max_iterations 500
```

For long-running jobs (train = ~20 min), always use `nohup ... &` and redirect to a log file:

```bash
nohup docker exec -e PYTHONPATH="..." isaac-lab-base \
  /workspace/isaaclab/isaaclab.sh -p ... \
  > _runlogs/myrun.log 2>&1 &
echo "PID=$!"
```

Then monitor: `tail -f _runlogs/myrun.log`

---

## Container Lifecycle Gotchas

- **First sim boot is slow** (~1–2 min) — Omniverse Kit loads, RTX pipeline warms up. Don't assume it's hung.
- **Isaac Sim processes linger during shutdown** — `pkill` on the host won't reach them; kill inside the container: `docker exec isaac-lab-base bash -lc "pkill -f custom_train.py"`.
- **Root ownership issue** — files written inside the container to bind-mounted dirs are owned by root. Fix: `docker exec isaac-lab-base chown -R $(id -u):$(id -g) /workspace/programs/results`.

---

## Image Info

- Image: `ghcr.io/sushruths04/humanoid-isaaclab:latest` (~17.6 GB)
- Base: `nvcr.io/nvidia/isaac-sim:5.1.0`
- **Public since 2026-06-05** — no token needed to pull. See [[GHCR Image & Auth]].
- Digest (as of 2026-06-05): `sha256:b6edb1a8ea7bbc3ad77d2b9ae4b1238c3f7b2ccb9da44567ba8916afe7127efa`

---

## Related

- [[Lightning Studio Environment]]
- [[GHCR Image & Auth]]
- [[PYTHONPATH & Python Interpreters]]
- [[container.py Forces Rebuild]]
- [[Results Lost to Ephemeral Container Storage]]
- [Docker Image Reuse Guide](../../DOCKER_IMAGE_REUSE.md)
