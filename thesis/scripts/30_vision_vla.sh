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

TASK="${TASK:-Humanoid-G1-Vision-VLA-v0}"
NUM_ENVS="${NUM_ENVS:-64}"
MAX_ITERS="${MAX_ITERS:-300}"
TRAIN_TIMEOUT_MINUTES="${TRAIN_TIMEOUT_MINUTES:-60}"
VLA_CAMERA_HEIGHT="${VLA_CAMERA_HEIGHT:-64}"
VLA_CAMERA_WIDTH="${VLA_CAMERA_WIDTH:-64}"
VLA_CAMERA_UPDATE_PERIOD="${VLA_CAMERA_UPDATE_PERIOD:-0.2}"
VLA_ENABLE_CAMERA="${VLA_ENABLE_CAMERA:-1}"
VLA_ENABLE_CAMERA_OBS="${VLA_ENABLE_CAMERA_OBS:-1}"
MAX_VISION_ENVS_WITH_CAMERA="${MAX_VISION_ENVS_WITH_CAMERA:-128}"
PYTHONPATH_IN_CONTAINER="/workspace/my-humanoid-project:/workspace/isaaclab/source"
RENDERING_EXPERIENCE="/workspace/isaaclab/apps/isaaclab.python.headless.rendering.kit"

if [ "$VLA_ENABLE_CAMERA" != "0" ] && [ "$NUM_ENVS" -gt "$MAX_VISION_ENVS_WITH_CAMERA" ] && [ "${ALLOW_LARGE_VISION_ENVS:-0}" != "1" ]; then
  log_step "Refusing to launch $NUM_ENVS camera envs. This stalls Isaac Sim before PPO starts."
  log_step "Use NUM_ENVS<=${MAX_VISION_ENVS_WITH_CAMERA}, or set ALLOW_LARGE_VISION_ENVS=1 if you are intentionally testing camera scaling."
  exit 64
fi

log_step "Starting Vision VLA G1 training: $TASK ($NUM_ENVS envs, $MAX_ITERS iters, camera ${VLA_CAMERA_WIDTH}x${VLA_CAMERA_HEIGHT}@${VLA_CAMERA_UPDATE_PERIOD}s, timeout ${TRAIN_TIMEOUT_MINUTES}m)"

log_step "Ensuring vulkaninfo is available inside Isaac Lab container..."
docker exec isaac-lab-base bash -lc '
set -euo pipefail
if ! command -v vulkaninfo >/dev/null 2>&1; then
    apt-get update
    apt-get install -y --no-install-recommends vulkan-tools
    rm -rf /var/lib/apt/lists/*
fi
'

log_step "Checking Vulkan graphics access inside Isaac Lab container..."
docker exec isaac-lab-base /usr/bin/vulkaninfo --summary

log_step "Ensuring warp-lang is pinned to 1.4.2 inside the container..."
docker exec isaac-lab-base bash -lc '
set -euo pipefail
python_bin="/workspace/isaaclab/_isaac_sim/python.sh"
current="$("$python_bin" -m pip show warp-lang 2>/dev/null | awk "/^Version:/ {print \$2}")"
if [ "$current" != "1.4.2" ]; then
    "$python_bin" -m pip install warp-lang==1.4.2
fi
'

{
  echo "## Vision VLA training (Smoke)"
  echo
  echo "- Task: \`$TASK\`"
  echo "- Envs: $NUM_ENVS"
  echo "- Max Iters: $MAX_ITERS"
  echo "- Camera: ${VLA_CAMERA_WIDTH}x${VLA_CAMERA_HEIGHT}, update period ${VLA_CAMERA_UPDATE_PERIOD}s"
  echo "- Timeout: ${TRAIN_TIMEOUT_MINUTES} minutes"
  echo
  echo '```bash'
  echo "docker exec -e PYTHONPATH=$PYTHONPATH_IN_CONTAINER -e VLA_CAMERA_HEIGHT=$VLA_CAMERA_HEIGHT -e VLA_CAMERA_WIDTH=$VLA_CAMERA_WIDTH isaac-lab-base /workspace/isaaclab/isaaclab.sh -p /workspace/my-humanoid-project/custom_train.py --task $TASK --headless --enable_cameras --num_envs $NUM_ENVS --max_iterations $MAX_ITERS --experience $RENDERING_EXPERIENCE"
  echo '```'
} | md_log "30-vision-vla" "STEP 30 Vision VLA smoke test"

# Run training inside container.
# We force EGL and Vulkan to use NVIDIA drivers. Isaac Sim can hang while
# unloading plugins after training completes, so the wrapper treats the
# "Training time:" line as success and cleans up the process if shutdown stalls.
rm -f "$LOG_DIR/train.log"
set +e
docker exec \
  -e PYTHONPATH="$PYTHONPATH_IN_CONTAINER" \
  -e PYTHONUNBUFFERED=1 \
  -e __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
  -e VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  -e VLA_CAMERA_HEIGHT="$VLA_CAMERA_HEIGHT" \
  -e VLA_CAMERA_WIDTH="$VLA_CAMERA_WIDTH" \
  -e VLA_CAMERA_UPDATE_PERIOD="$VLA_CAMERA_UPDATE_PERIOD" \
  -e VLA_ENABLE_CAMERA="$VLA_ENABLE_CAMERA" \
  -e VLA_ENABLE_CAMERA_OBS="$VLA_ENABLE_CAMERA_OBS" \
  isaac-lab-base /workspace/isaaclab/isaaclab.sh -p /workspace/my-humanoid-project/custom_train.py \
  --task "$TASK" \
  --headless \
  --enable_cameras \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERS" \
  --experience "$RENDERING_EXPERIENCE" \
  2>&1 | tee "$LOG_DIR/train.log" &
train_pid=$!

deadline=$((SECONDS + TRAIN_TIMEOUT_MINUTES * 60))
status=124
while true; do
  if grep -q "Training time:" "$LOG_DIR/train.log" 2>/dev/null; then
    status=0
    log_step "Training completed; waiting briefly for Isaac Sim shutdown..."
    shutdown_deadline=$((SECONDS + 60))
    while kill -0 "$train_pid" 2>/dev/null && [ "$SECONDS" -lt "$shutdown_deadline" ]; do
      sleep 2
    done
    if kill -0 "$train_pid" 2>/dev/null; then
      log_step "Isaac Sim shutdown stalled after successful training; terminating custom_train.py inside the container."
      docker exec isaac-lab-base bash -lc 'pkill -9 -f custom_train.py || true'
    fi
    wait "$train_pid" >/dev/null 2>&1
    break
  fi

  if ! kill -0 "$train_pid" 2>/dev/null; then
    wait "$train_pid"
    status=$?
    break
  fi

  if [ "$SECONDS" -ge "$deadline" ]; then
    log_step "Training timed out after ${TRAIN_TIMEOUT_MINUTES} minutes; terminating custom_train.py inside the container."
    docker exec isaac-lab-base bash -lc 'pkill -9 -f custom_train.py || true'
    wait "$train_pid" >/dev/null 2>&1
    status=124
    break
  fi

  sleep 5
done
set -e

if [ "$status" -eq 0 ]; then
  log_step "Vision Smoke successful."
  mark_done "30_vision_vla"
  exit 0
elif [ "$status" -eq 124 ]; then
  log_step "Vision Smoke timed out after ${TRAIN_TIMEOUT_MINUTES} minutes"
  exit "$status"
else
  log_step "Vision Smoke failed with exit code $status"
  exit "$status"
fi
