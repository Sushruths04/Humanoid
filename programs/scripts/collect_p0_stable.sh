#!/usr/bin/env bash
# Collect P0-stable checkpoint: copy from container, eval, report.
# Run from repo root on the remote after training completes.
# Usage: bash programs/scripts/collect_p0_stable.sh
set -euo pipefail

CT="isaac-lab-base"
PP="/workspace:/workspace/my-humanoid-project:/workspace/isaaclab/source"
RUN_DIR="2026-06-05_12-51-13"
CONTAINER_CKPT="/workspace/isaaclab/logs/rsl_rl/g1_flat/${RUN_DIR}/model_499.pt"
HOST_CKPT_DIR="programs/checkpoints/g1_commandnav_stable"
HOST_CKPT="$HOST_CKPT_DIR/model_499.pt"
RESULT_PATH="programs/results/p0_stable.md"

echo "[collect] Verifying checkpoint exists in container..."
docker exec "$CT" ls "$CONTAINER_CKPT"

echo "[collect] Copying checkpoint to bind-mounted path..."
mkdir -p "$HOST_CKPT_DIR"
docker exec "$CT" cp "$CONTAINER_CKPT" "/workspace/$HOST_CKPT"
echo "[collect] Checkpoint at $HOST_CKPT"

echo "[eval] Running evaluation (256 envs, Humanoid-G1-CommandNav-v0)..."
docker exec -e PYTHONPATH="$PP" "$CT" /workspace/isaaclab/isaaclab.sh -p \
    /workspace/programs/common/eval/evaluate.py \
    --task Humanoid-G1-CommandNav-v0 \
    --headless --num-envs 256 \
    --checkpoint "/workspace/$HOST_CKPT" \
    --out "$RESULT_PATH"

echo "[collect] Result written to $RESULT_PATH"
cp "$RESULT_PATH" docs/results/p0_stable.md
echo "[collect] Copied to docs/results/p0_stable.md"

echo "[collect] DONE. Check fall rate in $RESULT_PATH"
