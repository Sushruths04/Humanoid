#!/usr/bin/env bash
# Collect P0-stable-v2 checkpoint (upright_weight=2.0): copy, eval, report.
# Run from repo root on the remote after v2 training completes.
# Usage: bash programs/scripts/collect_p0_stable_v2.sh
set -euo pipefail

CT="isaac-lab-base"
PP="/workspace:/workspace/my-humanoid-project:/workspace/isaaclab/source"
HOST_CKPT_DIR="programs/checkpoints/g1_commandnav_stable_v2"
HOST_CKPT="$HOST_CKPT_DIR/model_499.pt"
RESULT_PATH="programs/results/p0_stable_v2.md"

echo "[collect-v2] Finding latest training run..."
RUN_DIR=$(docker exec "$CT" bash -lc "ls -td /workspace/isaaclab/logs/rsl_rl/g1_flat/*/ | head -1" | tr -d ' \n')
CONTAINER_CKPT="${RUN_DIR}model_499.pt"

echo "[collect-v2] Container checkpoint: $CONTAINER_CKPT"
docker exec "$CT" ls "$CONTAINER_CKPT"

echo "[collect-v2] Copying to bind-mounted path..."
mkdir -p "$HOST_CKPT_DIR"
docker exec "$CT" cp "$CONTAINER_CKPT" "/workspace/$HOST_CKPT"
echo "[collect-v2] Checkpoint at $HOST_CKPT"

echo "[eval-v2] Running evaluation (256 envs)..."
docker exec -e PYTHONPATH="$PP" "$CT" /workspace/isaaclab/isaaclab.sh -p \
    /workspace/programs/common/eval/evaluate.py \
    --task Humanoid-G1-CommandNav-v0 \
    --headless --num-envs 256 \
    --checkpoint "/workspace/$HOST_CKPT" \
    --out "$RESULT_PATH"

echo "[collect-v2] Result written to $RESULT_PATH"
cp "$RESULT_PATH" docs/results/p0_stable_v2.md
echo "[collect-v2] Copied to docs/results/p0_stable_v2.md"
echo "[collect-v2] DONE. Check fall rate in $RESULT_PATH"
