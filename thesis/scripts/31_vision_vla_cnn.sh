#!/usr/bin/env bash
set -euo pipefail

# Higher-quality Vision-VLA defaults:
# - larger camera frames
# - faster camera refresh
# - more envs by default
# - longer training window
: "${NUM_ENVS:=128}"
: "${MAX_ITERS:=500}"
: "${VISION_VLA_MAX_ITERS:=${MAX_ITERS}}"
: "${VLA_CAMERA_HEIGHT:=128}"
: "${VLA_CAMERA_WIDTH:=128}"
: "${VLA_CAMERA_UPDATE_PERIOD:=0.05}"
: "${VISION_VLA_TASK:=Humanoid-G1-Vision-VLA-v0}"
TASK="${TASK:-$VISION_VLA_TASK}"

export NUM_ENVS MAX_ITERS VISION_VLA_MAX_ITERS VLA_CAMERA_HEIGHT VLA_CAMERA_WIDTH VLA_CAMERA_UPDATE_PERIOD VISION_VLA_TASK TASK

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/30_vision_vla.sh" "$@"
