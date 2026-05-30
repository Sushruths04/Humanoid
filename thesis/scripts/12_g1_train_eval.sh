#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_gpu "12_g1_train_eval" || exit 78

ISAACLAB_DIR="$WORKSPACE_DIR/IsaacLab"
MY_PROJECT_DIR="$WORKSPACE_DIR/my-humanoid-project"
LOG_DIR="$THESIS_DIR/logs/g1_language"
mkdir -p "$LOG_DIR" "$THESIS_DIR/checkpoints/g1_language"

cd "$ISAACLAB_DIR"

log_step "Ensuring Isaac Lab Docker is running..."
python3 docker/container.py start

TASK="${G1_LANGUAGE_TASK_ID:-Humanoid-G1-Language-PickPlace-v0}"
NUM_ENVS="${G1_NUM_ENVS:-512}"
MAX_ITERS="${G1_MAX_ITERS:-300}"

log_step "Starting language-conditioned G1 training: $TASK ($NUM_ENVS envs, $MAX_ITERS iters)"

{
  echo "## Language-conditioned training"
  echo
  echo "- Task: \`$TASK\`"
  echo "- Envs: $NUM_ENVS"
  echo "- Max Iters: $MAX_ITERS"
  echo
  echo '```bash'
  echo "docker exec -e PYTHONPATH=/workspace/my-humanoid-project:/workspace/isaaclab/source isaac-lab-base /workspace/isaaclab/isaaclab.sh -p /workspace/my-humanoid-project/custom_train.py --task $TASK --headless --num_envs $NUM_ENVS --max_iterations $MAX_ITERS"
  echo '```'
} | md_log "12-g1-train-eval" "STEP 12 G1 language training"

# Run training inside container
# We resume from the baseline checkpoint if it exists
RESUME_ARGS=""
LATEST_CKPT=$(docker exec isaac-lab-base find /workspace/isaaclab/logs/rsl_rl -name 'model_*.pt' 2>/dev/null | sort -V | tail -1 || true)
if [ -n "$LATEST_CKPT" ]; then
  log_step "Resuming from baseline checkpoint: $LATEST_CKPT"
  RESUME_ARGS="--resume --checkpoint $LATEST_CKPT"
fi

set +e
docker exec -e PYTHONPATH="/workspace/my-humanoid-project:/workspace/isaaclab/source" \
  isaac-lab-base /workspace/isaaclab/isaaclab.sh -p /workspace/my-humanoid-project/custom_train.py \
  --task "$TASK" \
  --headless \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERS" \
  $RESUME_ARGS \
  2>&1 | tee "$LOG_DIR/train.log"
status=${PIPESTATUS[0]}
set -e

if [ "$status" -eq 0 ]; then
  log_step "Training successful. Copying latest checkpoint..."
  LATEST_CKPT=$(ls -dt "$ISAACLAB_DIR/logs/rsl_rl/"*/*/*.pt 2>/dev/null | head -1 || true)
  if [ -n "$LATEST_CKPT" ]; then
    cp "$LATEST_CKPT" "$THESIS_DIR/checkpoints/g1_language/model_latest.pt"
    log_step "Saved checkpoint to thesis/checkpoints/g1_language/model_latest.pt"
  fi
  mark_done "12_g1_train_eval"
  exit 0
else
  log_step "Training failed with exit code $status"
  exit "$status"
fi

