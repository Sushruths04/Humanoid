#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_gpu "30_vision_vla" || exit 78

ISAACLAB_DIR="$WORKSPACE_DIR/IsaacLab"
MY_PROJECT_DIR="$WORKSPACE_DIR/my-humanoid-project"
LOG_DIR="$THESIS_DIR/logs/g1_vision"
mkdir -p "$LOG_DIR" "$THESIS_DIR/checkpoints/g1_vision"

cd "$ISAACLAB_DIR"

log_step "Ensuring Isaac Lab Docker is running..."
python3 docker/container.py start

# For vision, we need to ensure the container has vulkan/rendering support
# The L40S and our base image already support this in headless mode.

TASK="Humanoid-G1-Vision-VLA-v0"
NUM_ENVS=32 # Vision is HEAVY, start small
MAX_ITERS=300 # Smoke test

log_step "Starting Vision VLA G1 training: $TASK ($NUM_ENVS envs, $MAX_ITERS iters)"

{
  echo "## Vision VLA training (Smoke)"
  echo
  echo "- Task: \`$TASK\`"
  echo "- Envs: $NUM_ENVS"
  echo "- Max Iters: $MAX_ITERS"
  echo
  echo '```bash'
  echo "docker exec -e PYTHONPATH=/workspace/my-humanoid-project:/workspace/isaaclab/source isaac-lab-base /workspace/isaaclab/isaaclab.sh -p /workspace/my-humanoid-project/custom_train.py --task $TASK --headless --num_envs $NUM_ENVS --max_iterations $MAX_ITERS"
  echo '```'
} | md_log "30-vision-vla" "STEP 30 Vision VLA smoke test"

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
  log_step "Vision Smoke successful."
  mark_done "30_vision_vla"
  exit 0
else
  log_step "Vision Smoke failed with exit code $status"
  exit "$status"
fi
