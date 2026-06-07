# P3 VisionNav — Eval & Video Guide

How to run inference, record video, and evaluate the P3 VisionNav policy from any machine.

**Best checkpoint:** `model_499.pt` (reward +141.35, 96.28% success)  
**HuggingFace:** `mitvho09/humanoid-g1-nav` → `checkpoints/p3_vision_nav/run_300_l4/model_499.pt`

---

## Prerequisites

1. Lightning Studio with L4+ GPU (24 GB VRAM minimum)
2. Docker container `isaac-lab-base` running
3. Code and checkpoints at `/teamspace/studios/this_studio/Humanoid/`

### Bring up the container
```bash
cd /teamspace/studios/this_studio/Humanoid/IsaacLab/docker
DOCKER_NAME_SUFFIX= docker compose --env-file .env.base --profile base up isaac-lab-base -d --no-build
```

### Copy checkpoint into container
```bash
# Make the logs directory inside the container
docker exec isaac-lab-base bash -c 'mkdir -p /workspace/isaaclab/logs/rsl_rl/g1_vision_nav/p3_eval/params'

# Copy checkpoint and params from persistent storage
docker cp /teamspace/studios/this_studio/Humanoid/programs/checkpoints/p3_vision_nav/run_300_l4/model_499.pt \
  isaac-lab-base:/workspace/isaaclab/logs/rsl_rl/g1_vision_nav/p3_eval/model_499.pt

docker cp /teamspace/studios/this_studio/Humanoid/programs/checkpoints/p3_vision_nav/run_300_l4/params/. \
  isaac-lab-base:/workspace/isaaclab/logs/rsl_rl/g1_vision_nav/p3_eval/params/
```

### Kill any running training process first
```bash
# Always free VRAM before eval — training process holds it even after completion
docker exec isaac-lab-base bash -c 'pkill -9 -f custom_train; pkill -9 -f isaaclab'
docker exec isaac-lab-base nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
# Should show: ~0 MiB used, ~22564 MiB free
```

---

## Run Inference (no video, runs forever)

```bash
docker exec \
  -e PYTHONPATH='/workspace:/workspace/my-humanoid-project:/workspace/isaaclab/source' \
  -e P3_CAM_H='64' -e P3_CAM_W='64' -e P3_NUM_STEPS='24' \
  isaac-lab-base /workspace/isaaclab/isaaclab.sh -p \
  /workspace/my-humanoid-project/custom_play.py \
  --task Humanoid-G1-VisionNav-v0 --headless --enable_cameras \
  --num_envs 128 \
  --load_run p3_eval
```

> **Note:** The episode loop has no print statements — the process runs silently. Kill with `Ctrl+C` or `pkill`. For results, use training stats (96.28%) not this.

---

## Record a Video

```bash
nohup docker exec \
  -e PYTHONPATH='/workspace:/workspace/my-humanoid-project:/workspace/isaaclab/source' \
  -e P3_CAM_H='64' -e P3_CAM_W='64' -e P3_NUM_STEPS='24' \
  isaac-lab-base /workspace/isaaclab/isaaclab.sh -p \
  /workspace/my-humanoid-project/custom_play.py \
  --task Humanoid-G1-VisionNav-v0 --headless --enable_cameras \
  --num_envs 4 \
  --load_run p3_eval \
  --video --video_length 500 \
  > /tmp/eval_video.log 2>&1 &
echo PID: $!
```

Parameters:
- `--num_envs 4` — enough for a clean single-env video (fewer = faster BVH init)
- `--video_length 500` — 500 steps × 0.02 s = 10 seconds of footage
- Process auto-exits after 500 steps and saves the mp4

### Monitor
```bash
tail -f /tmp/eval_video.log
# Look for: "Recording videos during training."
# Then wait ~2-3 min for 500 steps to render
```

### Find and copy the video
```bash
# Find it inside the container
docker exec isaac-lab-base find /workspace/isaaclab/logs -name "*.mp4"
# → /workspace/isaaclab/logs/rsl_rl/g1_vision_nav/p3_eval/videos/play/rl-video-step-0.mp4

# Copy to persistent storage
docker cp isaac-lab-base:/workspace/isaaclab/logs/rsl_rl/g1_vision_nav/p3_eval/videos/play/rl-video-step-0.mp4 \
  /teamspace/studios/this_studio/Humanoid/programs/videos/p3_vision_nav_model499.mp4

ls -lh /teamspace/studios/this_studio/Humanoid/programs/videos/p3_vision_nav_model499.mp4
```

---

## Upload Video to HuggingFace

```bash
/home/zeus/miniconda3/bin/python3 -c "
from huggingface_hub import HfApi
api = HfApi(token='YOUR_HF_TOKEN')
api.upload_file(
    path_or_fileobj='/teamspace/studios/this_studio/Humanoid/programs/videos/p3_vision_nav_model499.mp4',
    path_in_repo='videos/p3_vision_nav/p3_vision_nav_model499.mp4',
    repo_id='mitvho09/humanoid-g1-nav',
    repo_type='dataset',
    commit_message='P3 VisionNav video: model_499.pt (96.28% success)'
)
print('done')
"
```

---

## Download Video Locally (from Lightning to your PC)

```bash
# On your local Windows PowerShell:
scp -i ~/.ssh/lightning_rsa \
  s_01ktege6zxxg3x8xr0ert8cs5q@ssh.lightning.ai:/teamspace/studios/this_studio/Humanoid/programs/videos/p3_vision_nav_model499.mp4 \
  "D:\Mini Thesis\NVIDIA\programs\videos\p3_vision_nav_model499.mp4"
```

---

## Key Notes

| Thing | Value |
|---|---|
| Best checkpoint | `model_499.pt` (run_300_l4) |
| Reward at model_499 | +141.35 |
| Official success rate | **96.28%** (training epoch stats, 4096 envs) |
| Video location (HF) | `mitvho09/humanoid-g1-nav/videos/p3_vision_nav/p3_vision_nav_model499.mp4` |
| Env resolution | 64×64 RGB TiledCamera |
| Steps per video | 500 (set with `--video_length`) |

### Common Mistakes
- **Don't use `--checkpoint model_499.pt`** — bare filename fails; use `--load_run p3_eval` only
- **Kill training process first** — it holds VRAM even after training completes
- **Don't expect play.py to print stats** — it's an infinite loop with no output; training stats are authoritative
- **OOM at 512+ envs on L4** — use 128 or fewer for eval; 4 for video recording
