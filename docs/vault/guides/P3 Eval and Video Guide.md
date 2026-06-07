---
tags: [guide, eval, video, p3, play, checkpoint]
---

# P3 Eval & Video Guide

> **Full version with all commands**: `docs/guides/p3_eval_and_video_guide.md`

---

## Best Checkpoint

`model_499.pt` — reward +141.35, **96.28% success**  
HuggingFace: `mitvho09/humanoid-g1-nav` → `checkpoints/p3_vision_nav/run_300_l4/model_499.pt`

---

## Container Setup (on new machine)

```bash
# 1. Bring up container
cd /teamspace/studios/this_studio/Humanoid/IsaacLab/docker
DOCKER_NAME_SUFFIX= docker compose --env-file .env.base --profile base up isaac-lab-base -d --no-build

# 2. Copy checkpoint into container
docker exec isaac-lab-base bash -c 'mkdir -p /workspace/isaaclab/logs/rsl_rl/g1_vision_nav/p3_eval/params'
docker cp .../run_300_l4/model_499.pt isaac-lab-base:/workspace/isaaclab/logs/rsl_rl/g1_vision_nav/p3_eval/model_499.pt
docker cp .../run_300_l4/params/. isaac-lab-base:/workspace/isaaclab/logs/rsl_rl/g1_vision_nav/p3_eval/params/

# 3. Kill any training process (holds VRAM)
docker exec isaac-lab-base bash -c 'pkill -9 -f custom_train; pkill -9 -f isaaclab'
```

---

## Record Video (auto-exits after N steps)

```bash
nohup docker exec \
  -e PYTHONPATH='/workspace:/workspace/my-humanoid-project:/workspace/isaaclab/source' \
  -e P3_CAM_H='64' -e P3_CAM_W='64' -e P3_NUM_STEPS='24' \
  isaac-lab-base /workspace/isaaclab/isaaclab.sh -p \
  /workspace/my-humanoid-project/custom_play.py \
  --task Humanoid-G1-VisionNav-v0 --headless --enable_cameras \
  --num_envs 4 --load_run p3_eval \
  --video --video_length 500 > /tmp/eval_video.log 2>&1 &

# Find video after process exits:
docker exec isaac-lab-base find /workspace/isaaclab/logs -name "*.mp4"
# → .../p3_eval/videos/play/rl-video-step-0.mp4

# Copy to persistent storage:
docker cp isaac-lab-base:.../rl-video-step-0.mp4 \
  /teamspace/.../Humanoid/programs/videos/p3_vision_nav_model499.mp4
```

**Timings on L4:** ~3 min BVH init + ~2 min recording = 5 min total

---

## Key Rules
- **Always use `--load_run p3_eval`**, never `--checkpoint model_499.pt` (bare filename fails)
- **Kill training before eval** — training process holds 20 GB VRAM after completing
- **`--num_envs 4` for video**, `--num_envs 128` max for inference on L4
- **play.py has no episode output** — success rate comes from training stats (96.28%), not play.py

---

## Related
- [[play.py Checkpoint Bare Filename Not Found]] — why --checkpoint bare fails
- [[play.py Eval Has No Episode Output]] — why there are no stats from play.py
- [[Docker Image Lost on GPU Upgrade]] — re-pull on every new machine
- [[P3 - VisionNav]]
