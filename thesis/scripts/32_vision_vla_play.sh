#!/usr/bin/env bash
set -euo pipefail

: "${VISION_VLA_TASK:=Humanoid-G1-Vision-VLA-v0}"
: "${VISION_VLA_LOG_ROOT:=logs/rsl_rl/g1_vla_vision_cnn}"
: "${VISION_VLA_VIDEO:=1}"
: "${VISION_VLA_VIDEO_LENGTH:=400}"
TASK="${TASK:-$VISION_VLA_TASK}"

export VISION_VLA_TASK VISION_VLA_LOG_ROOT VISION_VLA_VIDEO VISION_VLA_VIDEO_LENGTH TASK

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

exec python "${ROOT_DIR}/my-humanoid-project/play_vision_vla.py" \
  --task "${VISION_VLA_TASK}" \
  --log-root "${VISION_VLA_LOG_ROOT}" \
  --video-length "${VISION_VLA_VIDEO_LENGTH}" \
  "$@"
