# Lightning Backup Workflow

Use this when your current Lightning machine is going away, sleeping, or you want a second machine ready as fallback.

## What This Backup Is For

- keep the code in GitHub
- keep large checkpoints, logs, and rollout videos in Hugging Face
- keep the reusable Docker image in GHCR
- keep Lightning machines disposable

## Backup Rule

Do not treat one Lightning machine as permanent storage.

If the current machine stops, the next machine should be able to recover from:

- GitHub for code and docs
- Hugging Face for large artifacts
- GHCR for the prebuilt Isaac Lab image

## Recovery Steps

1. Open the new Lightning Studio.
2. Re-run the SSH setup flow if needed.
3. Clone the GitHub repo.
4. Pull the GHCR image.
5. Download the latest Hugging Face artifacts.
6. Run the Vision-VLA smoke test before any long training job.

```bash
cd /home/zeus/content
git clone https://github.com/Sushruths04/Humanoid.git
cd Humanoid

docker login ghcr.io
REGISTRY_IMAGE=ghcr.io/sushruths04/humanoid-isaaclab:latest bash thesis/scripts/docker_image_portability.sh pull

python3 thesis/scripts/hf_download.py
NUM_ENVS=16 MAX_ITERS=2 TRAIN_TIMEOUT_MINUTES=20 VLA_CAMERA_HEIGHT=32 VLA_CAMERA_WIDTH=32 bash thesis/scripts/30_vision_vla.sh
```

## What To Save Before Switching

- code and markdown: GitHub
- checkpoints and rollout videos: Hugging Face
- built image: GHCR or the tarball fallback
- any unfinished run status: a short markdown note in the repo

## What Not To Do

- do not keep secrets in repo files
- do not rely on one machine as the only copy
- do not skip the smoke test after moving machines
- do not assume the same GPU/session survives a machine switch

## Related Docs

- [Start Here](./START_HERE.md)
- [Machine switch quick reference](./MACHINE_SWITCH_QUICK_REF.md)
- [CPU to GPU machine switch](./CPU_TO_GPU_MACHINE_SWITCH.md)
- [Machine change runbook](./MACHINE_CHANGE_RUNBOOK.md)
