#!/usr/bin/env bash
# Batch train + evaluate ALL P0/P1 navigation tasks (run when GPU is available).
# Smoke first (override SMOKE=1 for a quick wiring check), else full runs.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TASKS=(
  "Humanoid-G1-CommandNav-v0"
  "Humanoid-G1-LangNav-v0"
  "Humanoid-G1-ObstacleNav-v0"
  "Humanoid-G1-SeqNav-v0"
)
if [ "${SMOKE:-0}" = "1" ]; then
  ENVS=16; ITERS=2; EVAL=16
else
  ENVS="${NUM_ENVS:-4096}"; ITERS="${MAX_ITERS:-500}"; EVAL="${EVAL_ENVS:-256}"
fi
for t in "${TASKS[@]}"; do
  echo "==================== $t ===================="
  bash "$HERE/train_eval_nav.sh" "$t" "$ENVS" "$ITERS" "$EVAL" || echo "FAILED: $t"
done
echo "Batch complete. Results in docs/results/."
