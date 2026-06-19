#!/usr/bin/env bash
# Record an mp4 rollout (PLAN §14 task 13). This step NEEDS the camera/graphics path, so it
# uses the headless RENDERING KIT + --enable_cameras (the fix from VISION_BLOCKER_REPORT.md).
# On Modal this may hit Vulkan limits — if so, render on Lightning AI instead.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CKPT:?set CKPT=path/to/model.pt}"
python eval.py --checkpoint "${CKPT}" --v_pull_kmh "${V_PULL_KMH:-30}" --episodes 4 \
  --out "results/video_eval.json" \
  # VERIFY: add --video flag + RecordVideo wrapper; pass the rendering.kit experience file.
echo "If rendering fails on Modal, run this on Lightning AI (RT-core GPU + graphics caps)."
