#!/usr/bin/env bash
# Retrain P0 with stronger upright_reward (weight=2.0) if v1 fall rate > 10%.
# Run from repo root on the GPU machine after v1 eval confirms fall rate >= 10%.
# Usage: bash programs/scripts/retrain_p0_stable_v2.sh
set -euo pipefail

CT="isaac-lab-base"
PP="/workspace:/workspace/my-humanoid-project:/workspace/isaaclab/source"
RUN_DIR_V2="v2_$(date +%Y-%m-%d_%H-%M-%S)"

echo "[retrain-v2] Starting P0-stable-v2 training (COMMANDNAV_UPRIGHT_WEIGHT=2.0, 500 iters)..."
nohup docker exec \
    -e PYTHONPATH="$PP" \
    -e COMMANDNAV_UPRIGHT_WEIGHT=2.0 \
    "$CT" /workspace/isaaclab/isaaclab.sh -p \
    /workspace/my-humanoid-project/custom_train.py \
    --task Humanoid-G1-CommandNav-v0 \
    --headless --num_envs 4096 --max_iterations 500 \
    > /tmp/train_commandnav_stable_v2.log 2>&1 &

echo "[retrain-v2] PID=$!  log: /tmp/train_commandnav_stable_v2.log"
echo "[retrain-v2] Monitor: tail -f /tmp/train_commandnav_stable_v2.log"
