# Docker Image Reuse for Machine Switching

Use this when you change Lightning/GPU machines often and do not want to rebuild Isaac Lab every time.

## Best Path: Push to a Registry

Use a Docker registry for the Isaac Lab image. Use Hugging Face for checkpoints, logs, datasets, and result artifacts.

On the machine where the image already works:

```bash
cd /home/zeus/content/Humanoid
docker login ghcr.io
REGISTRY_IMAGE=ghcr.io/<github-user>/humanoid-isaaclab:latest bash thesis/scripts/docker_image_portability.sh push
```

On the next machine:

```bash
git clone https://github.com/Sushruths04/Humanoid.git
cd Humanoid
docker login ghcr.io
REGISTRY_IMAGE=ghcr.io/<github-user>/humanoid-isaaclab:latest bash thesis/scripts/docker_image_portability.sh pull
bash thesis/scripts/machine_switch.sh status
```

Then run a tiny training check before a long run:

```bash
NUM_ENVS=16 MAX_ITERS=2 TRAIN_TIMEOUT_MINUTES=20 VLA_CAMERA_HEIGHT=32 VLA_CAMERA_WIDTH=32 bash thesis/scripts/30_vision_vla.sh
```

## Manual Tarball Path

This is slower and requires moving a large file between machines.

Save:

```bash
bash thesis/scripts/docker_image_portability.sh save
```

Load:

```bash
bash thesis/scripts/docker_image_portability.sh load
```

## Important

- Do not commit GitHub, Hugging Face, or Docker tokens.
- Do not store large Docker images in Git.
- Start vision training with small env counts first. Camera envs scale very differently from non-vision physics envs.
- Use `NUM_ENVS<=128` for camera PPO unless you deliberately set `ALLOW_LARGE_VISION_ENVS=1`.
