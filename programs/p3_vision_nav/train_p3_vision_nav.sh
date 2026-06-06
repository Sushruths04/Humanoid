#!/usr/bin/env bash
# P3 Vision Nav training on A100 80GB.
# Usage: bash programs/p3_vision_nav/train_p3_vision_nav.sh [NUM_ENVS] [MAX_ITERS]
# Run from repo root: /teamspace/studios/this_studio/Humanoid/
set -euo pipefail

NUM_ENVS="${1:-8192}"
MAX_ITERS="${2:-1500}"
CT="isaac-lab-base"
PP="/workspace:/workspace/my-humanoid-project:/workspace/isaaclab/source"
LOG="/tmp/train_p3_vision_nav.log"

mkdir -p _runlogs

echo "[P3] VisionNav training: envs=$NUM_ENVS iters=$MAX_ITERS"
echo "[P3] Log: $LOG"
echo "[P3] Monitor: tail -f $LOG"

nohup docker exec \
    -e PYTHONPATH="$PP" \
    -e P3_MAX_ITERS="$MAX_ITERS" \
    -e P3_NUM_STEPS="48" \
    -e P3_CAM_H="84" \
    -e P3_CAM_W="84" \
    "$CT" /workspace/isaaclab/isaaclab.sh -p \
    /workspace/my-humanoid-project/custom_train.py \
    --task Humanoid-G1-VisionNav-v0 \
    --headless --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERS" \
    > "$LOG" 2>&1 &

echo "[P3] PID=$!  started."
