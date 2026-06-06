#!/usr/bin/env bash
# P3 Vision Nav eval — finds the latest checkpoint and evaluates on CommandNav.
# Usage: bash programs/p3_vision_nav/eval_p3_vision_nav.sh [CHECKPOINT_PATH]
# Run from repo root: /teamspace/studios/this_studio/Humanoid/
set -euo pipefail

CT="isaac-lab-base"
PP="/workspace:/workspace/my-humanoid-project:/workspace/isaaclab/source"
EVAL_ENVS=256
OUT_HOST="programs/results/p3_vision_nav.md"
OUT_C="/workspace/programs/results/p3_vision_nav.md"
LOG="/tmp/eval_p3_vision_nav.log"

if [ -n "${1-}" ]; then
    CKPT="$1"
else
    # Auto-find latest checkpoint from training run
    CKPT=$(docker exec "$CT" bash -lc \
        "ls -t /workspace/isaaclab/logs/rsl_rl/g1_vision_nav/*/model_*.pt 2>/dev/null | head -1")
    if [ -z "$CKPT" ]; then
        echo "[P3-eval] ERROR: no checkpoint found. Pass path as arg or run training first."
        exit 1
    fi
fi

echo "[P3-eval] checkpoint: $CKPT"
echo "[P3-eval] output: $OUT_HOST"
echo "[P3-eval] log: $LOG"

docker exec "$CT" mkdir -p /workspace/programs/results

nohup docker exec \
    -e PYTHONPATH="$PP" \
    "$CT" /workspace/isaaclab/isaaclab.sh -p \
    /workspace/programs/common/eval/evaluate.py \
    --task Humanoid-G1-VisionNav-v0 \
    --headless --num-envs "$EVAL_ENVS" \
    --checkpoint "$CKPT" \
    --out "$OUT_C" \
    > "$LOG" 2>&1 &

echo "[P3-eval] PID=$!  Monitor: tail -f $LOG"
