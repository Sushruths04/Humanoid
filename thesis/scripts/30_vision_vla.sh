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
NUM_ENVS="${NUM_ENVS:-32}"
MAX_ITERS="${MAX_ITERS:-300}"
PYTHONPATH_IN_CONTAINER="/workspace/my-humanoid-project:/workspace/isaaclab/source"
RENDERING_EXPERIENCE="/workspace/isaaclab/apps/isaaclab.python.headless.rendering.kit"

log_step "Starting Vision VLA G1 training: $TASK ($NUM_ENVS envs, $MAX_ITERS iters)"

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
  echo
  echo '```bash'
  echo "docker exec -e PYTHONPATH=$PYTHONPATH_IN_CONTAINER isaac-lab-base /workspace/isaaclab/isaaclab.sh -p /workspace/my-humanoid-project/custom_train.py --task $TASK --headless --num_envs $NUM_ENVS --max_iterations $MAX_ITERS --experience $RENDERING_EXPERIENCE"
  echo '```'
} | md_log "30-vision-vla" "STEP 30 Vision VLA smoke test"

# Run training inside container
# We force EGL and Vulkan to use NVIDIA drivers
set +e
docker exec \
  -e PYTHONPATH="$PYTHONPATH_IN_CONTAINER" \
  -e __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
  -e VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  isaac-lab-base /workspace/isaaclab/isaaclab.sh -p /workspace/my-humanoid-project/custom_train.py \
  --task "$TASK" \
  --headless \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERS" \
  --experience "$RENDERING_EXPERIENCE" \
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
