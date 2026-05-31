#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_gpu "10_g1_baseline" || exit 78

ISAACLAB_DIR="$WORKSPACE_DIR/IsaacLab"
LOG_DIR="$THESIS_DIR/logs/g1_baseline"
mkdir -p "$LOG_DIR" "$THESIS_DIR/checkpoints/g1_baseline"

cd "$ISAACLAB_DIR"

log_step "Ensuring Isaac Lab Docker is ready..."
# Build if not present, then start
if ! docker images | grep -q "isaac-lab-base"; then
  log_step "Building Isaac Lab Docker image (this may take 10-20 min)..."
  python3 docker/container.py build
fi

log_step "Starting Isaac Lab container..."
python3 docker/container.py start

# Verify task exists and run training
TASK="${G1_TASK_ID:-Isaac-Velocity-Flat-G1-v0}"
NUM_ENVS="${G1_NUM_ENVS:-2048}"
MAX_ITERS=1000

log_step "Starting G1 baseline training: $TASK ($NUM_ENVS envs, $MAX_ITERS iters)"

{
  echo "## Baseline training"
  echo
  echo "- Task: \`$TASK\`"
  echo "- Envs: $NUM_ENVS"
  echo "- Max Iters: $MAX_ITERS"
  echo
  echo '```bash'
  echo "docker exec isaac-lab-base /workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task $TASK --headless --num_envs $NUM_ENVS --max_iterations $MAX_ITERS"
  echo '```'
} | md_log "10-g1-baseline" "STEP 10 G1 baseline training"

# Run training inside container
# Note: we mount host volumes so checkpoints/logs land in $ISAACLAB_DIR/logs/rsl_rl/
set +e
docker exec isaac-lab-base /workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task "$TASK" \
  --headless \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERS" \
  2>&1 | tee "$LOG_DIR/train.log"
status=${PIPESTATUS[0]}
set -e

if [ "$status" -eq 0 ]; then
  log_step "Training successful. Copying latest checkpoint..."
  # Find latest checkpoint in IsaacLab/logs/rsl_rl/ inside container
  LATEST_CKPT=$(docker exec isaac-lab-base find /workspace/isaaclab/logs/rsl_rl -name 'model_*.pt' 2>/dev/null | sort -V | tail -1 || true)
  if [ -n "$LATEST_CKPT" ]; then
    docker cp "isaac-lab-base:$LATEST_CKPT" "$THESIS_DIR/checkpoints/g1_baseline/model_latest.pt"
    log_step "Saved checkpoint to thesis/checkpoints/g1_baseline/model_latest.pt"
  fi
  mark_done "10_g1_baseline"
  exit 0
else
  log_step "Training failed with exit code $status"
  exit "$status"
fi

