# Vision-VLA CNN Runbook

Use this path when you want better visual quality than the smoke-test baseline.

## Train

```bash
cd /home/zeus/content/Humanoid
bash thesis/scripts/31_vision_vla_cnn.sh
```

Defaults in this path:
- `NUM_ENVS=128`
- `MAX_ITERS=500`
- `VLA_CAMERA_HEIGHT=128`
- `VLA_CAMERA_WIDTH=128`
- `VLA_CAMERA_UPDATE_PERIOD=0.05`

Override any of them inline if needed:

```bash
NUM_ENVS=96 MAX_ITERS=300 bash thesis/scripts/31_vision_vla_cnn.sh
```

## Play back a trained policy

This records an mp4 from the first rollout and saves it beside the checkpoint.
It is the safer inspection path on Lightning because it does not depend on a live GUI.

```bash
cd /home/zeus/content/Humanoid
bash thesis/scripts/32_vision_vla_play.sh
```

If your task registry uses a different name, set `VISION_VLA_TASK` first:

```bash
VISION_VLA_TASK=Your-Task-Name bash thesis/scripts/32_vision_vla_play.sh
```

If you want to force a specific checkpoint:

```bash
VISION_VLA_CHECKPOINT=/path/to/checkpoint.pt bash thesis/scripts/32_vision_vla_play.sh
```

## Where outputs go

- Code and scripts: GitHub
- Large checkpoints, videos, and training artifacts: Hugging Face or the Lightning machine as a temporary cache
- Docker image reuse: GHCR

## Current artifact state

While the longer `NUM_ENVS=256` run is still executing on Lightning, the current checkpoint set and rollout video are staged on the remote host at:

- `/home/zeus/content/Humanoid/thesis/artifacts/vision_vla/current_run/2026-06-01_19-19-20`
- `/home/zeus/content/Humanoid/thesis/artifacts/vision_vla/current_run/rl-video-step-0.mp4`
- `/home/zeus/content/Humanoid/thesis/artifacts/vision_vla/current_run/train.log`

These are the files to mirror into Hugging Face once a repo with write/create access is available.

## When to update the Docker image

Update the image only when you change:
- dependencies
- Isaac Lab / simulator runtime
- CUDA / Vulkan-related setup
- the Python environment used by training or playback

Do not rebuild the image for every new checkpoint.
