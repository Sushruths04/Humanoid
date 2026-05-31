#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_gpu "25_robust_training" || exit 78

ISAACLAB_DIR="$WORKSPACE_DIR/IsaacLab"
MY_PROJECT_DIR="$WORKSPACE_DIR/my-humanoid-project"
LOG_DIR="$THESIS_DIR/logs/g1_robust"
mkdir -p "$LOG_DIR" "$THESIS_DIR/checkpoints/g1_robust"

cd "$ISAACLAB_DIR"

log_step "Ensuring Isaac Lab Docker is running..."
python3 docker/container.py start

TASK="Humanoid-G1-Robust-VLA-v0"
NUM_ENVS=8192
MAX_ITERS=5000

log_step "Starting ROBUST G1 task training (Rough Terrain + DR): $TASK ($NUM_ENVS envs, $MAX_ITERS iters)"

{
  echo "## Robust task training"
  echo
  echo "- Task: \`$TASK\`"
  echo "- Envs: $NUM_ENVS"
  echo "- Max Iters: $MAX_ITERS"
  echo
  echo '```bash'
  echo "docker exec -e PYTHONPATH=/workspace/my-humanoid-project:/workspace/isaaclab/source isaac-lab-base /workspace/isaaclab/isaaclab.sh -p /workspace/my-humanoid-project/custom_train.py --task $TASK --headless --num_envs $NUM_ENVS --max_iterations $MAX_ITERS"
  echo '```'
} | md_log "25-robust-training" "STEP 25 G1 robust training"

# Run training inside container
set +e
docker exec -e PYTHONPATH="/workspace/my-humanoid-project:/workspace/isaaclab/source" \
  isaac-lab-base /workspace/isaaclab/isaaclab.sh -p /workspace/my-humanoid-project/custom_train.py \
  --task "$TASK" \
  --headless \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERS" \
  2>&1 | tee "$LOG_DIR/train.log"
status=${PIPESTATUS[0]}
set -e

if [ "$status" -eq 0 ]; then
  log_step "Robust Training successful. Copying latest checkpoint..."
  LATEST_CKPT=$(docker exec isaac-lab-base find /workspace/isaaclab/logs/rsl_rl -name 'model_*.pt' 2>/dev/null | sort -V | tail -1 || true)
  if [ -n "$LATEST_CKPT" ]; then
    docker cp "isaac-lab-base:$LATEST_CKPT" "$THESIS_DIR/checkpoints/g1_robust/model_latest.pt"
    log_step "Saved checkpoint to thesis/checkpoints/g1_robust/model_latest.pt"
  fi
  mark_done "25_robust_training"
  exit 0
else
  log_step "Robust Training failed with exit code $status"
  exit "$status"
fi
