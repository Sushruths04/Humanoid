---
tags: [workflow, video, demo, render]
---

# Rendering Demo Videos

## Command

```bash
CKPT="/workspace/isaaclab/logs/rsl_rl/g1_flat/<run_timestamp>/model_499.pt"
PP="/workspace:/workspace/my-humanoid-project:/workspace/isaaclab/source"

nohup docker exec -e PYTHONPATH="$PP" isaac-lab-base \
  /workspace/isaaclab/isaaclab.sh -p /workspace/my-humanoid-project/custom_play.py \
  --task Humanoid-G1-ObstacleNav-v0 \
  --num_envs 16 \
  --checkpoint "$CKPT" \
  --video \
  --video_length 600 \
  --headless \
  > _runlogs/video.log 2>&1 &
echo "PID=$!"
```

The mp4 is written to:
```
/workspace/isaaclab/logs/rsl_rl/g1_flat/<run>/videos/play/rl-video-step-0.mp4
```

---

## Grab the Video and Kill the Process

The render loop runs forever after the video is saved (see [[Video Render Never Exits]]):

```bash
# Wait for the file:
while true; do
  docker exec isaac-lab-base bash -lc "ls /workspace/isaaclab/logs/rsl_rl/g1_flat/<run>/videos/play/*.mp4 2>/dev/null" && break
  sleep 15
done

# Kill the render:
docker exec isaac-lab-base bash -lc "pkill -f custom_play.py"

# Copy to host:
docker cp isaac-lab-base:/workspace/isaaclab/logs/rsl_rl/g1_flat/<run>/videos/play/rl-video-step-0.mp4 \
  docs/results/videos/obstaclenav_demo.mp4

# Upload to HF:
hf upload mitvho09/humanoid-g1-nav docs/results/videos/obstaclenav_demo.mp4 videos/obstaclenav_demo.mp4
```

---

## Parameters

| Param | Recommended | Notes |
|---|---|---|
| `--num_envs` | 16 | 16 parallel envs = 16 robots shown side-by-side |
| `--video_length` | 500–600 | steps; ~10–12 sec at 50Hz |
| `--headless` | always | no display available on Lightning |

---

## VRAM for Rendering

Camera video recording requires **RT cores** (rendering). Use L4, L40S, RTX 4090 — NOT A100 (no RT cores). At 16 envs with `--video` flag, VRAM usage is ~6–8 GB on L4.

---

## Related

- [[Isaac Sim Docker Container]]
- [[Video Render Never Exits]]
- [[Stuck Wrapper Waiting on Lingering Process]]
