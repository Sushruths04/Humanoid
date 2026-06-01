# CPU To GPU Machine Switch

Use this when you land on a fresh Lightning machine and want to prepare on CPU first, then move to GPU later.

## What the CPU machine is for

Use the CPU machine to:
- clone or pull the repo
- read the runbooks
- edit code and docs
- verify shell scripts and Python syntax
- stage Git commits
- prepare Docker and Hugging Face access

Do not expect useful Vision-VLA training on CPU. The actual Isaac Lab runs need a GPU machine.

## Step 1. SSH in

Use the Lightning SSH target you were given:

```bash
ssh s_01kt2crz4evvhcs9x7gw01adtv@ssh.lightning.ai
```

## Step 2. Clone or update the repo

```bash
git clone https://github.com/Sushruths04/Humanoid.git
cd Humanoid
git pull --rebase --autostash origin main
```

## Step 3. Read the workflow files

Start with:
- `START_HERE.md`
- `README.md`
- `REMOTE_WORKFLOW.md`
- `DOCKER_IMAGE_REUSE.md`
- `VISION_VLA_CNN_RUNBOOK.md`

## Step 4. Prepare Docker reuse

If the machine already has Docker access and you want reuse instead of rebuild:

```bash
docker login ghcr.io
REGISTRY_IMAGE=ghcr.io/sushruths04/humanoid-isaaclab:latest bash thesis/scripts/docker_image_portability.sh pull
```

If pull is not available, ask for the saved tarball path or push a fresh image from the machine that already has it.

## Step 5. Prepare Hugging Face access

Use Hugging Face for large checkpoints, logs, and rollout videos.

```bash
hf auth whoami
```

If the machine has no `hf` CLI yet, install it on that machine and log in there. Keep tokens out of Git.

## Step 6. Keep code on GitHub

When you change code or docs:

```bash
git add .
git commit -m "Describe the change"
git push
```

## Step 7. Switch to GPU for training

When the GPU machine is available:

1. Clone or pull the same repo.
2. Pull the GHCR image.
3. Run the smoke test first.

```bash
NUM_ENVS=16 MAX_ITERS=2 TRAIN_TIMEOUT_MINUTES=20 VLA_CAMERA_HEIGHT=32 VLA_CAMERA_WIDTH=32 bash thesis/scripts/30_vision_vla.sh
```

4. If that passes, scale up to the longer run:

```bash
NUM_ENVS=256 ALLOW_LARGE_VISION_ENVS=1 MAX_ITERS=500 TRAIN_TIMEOUT_MINUTES=360 bash thesis/scripts/31_vision_vla_cnn.sh
```

## Step 8. Save the results

After a run:
- checkpoints stay in the run directory on the machine
- rollout videos go to the artifact folder
- upload the final folder to Hugging Face
- update the markdown runbook with the final artifact path and result summary

## Fast rule

- CPU machine: prep, edit, inspect, stage
- GPU machine: train, record rollout, export results
