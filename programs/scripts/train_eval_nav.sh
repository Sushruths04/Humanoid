#!/usr/bin/env bash
# Train + evaluate a navigation task inside the Isaac Lab container.
# Usage: train_eval_nav.sh <TASK_ID> [NUM_ENVS] [MAX_ITERS] [EVAL_ENVS]
# Results are written under programs/results/ (bind-mounted -> persists on host)
# and mirrored into docs/results/ for version control.
set -euo pipefail
TASK="${1:?task id required}"
NUM_ENVS="${2:-4096}"
MAX_ITERS="${3:-500}"
EVAL_ENVS="${4:-256}"
CT="isaac-lab-base"
PP="/workspace:/workspace/my-humanoid-project:/workspace/isaaclab/source"
NAME="$(echo "$TASK" | tr "A-Z/" "a-z_")"

echo "[train] $TASK  envs=$NUM_ENVS iters=$MAX_ITERS"
docker exec -e PYTHONPATH="$PP" "$CT" /workspace/isaaclab/isaaclab.sh -p \
  /workspace/my-humanoid-project/custom_train.py \
  --task "$TASK" --headless --num_envs "$NUM_ENVS" --max_iterations "$MAX_ITERS"

RUN=$(docker exec "$CT" bash -lc "ls -td /workspace/isaaclab/logs/rsl_rl/g1_flat/*/ | head -1")
CKPT=$(docker exec "$CT" bash -lc "ls -t ${RUN}model_*.pt | head -1")
OUT_C="/workspace/programs/results/${NAME}.md"   # programs/ bind-mounted -> host programs/results/
echo "[eval] checkpoint=$CKPT -> $OUT_C"
docker exec "$CT" mkdir -p /workspace/programs/results
docker exec -e PYTHONPATH="$PP" "$CT" /workspace/isaaclab/isaaclab.sh -p \
  /workspace/programs/common/eval/evaluate.py \
  --task "$TASK" --headless --num-envs "$EVAL_ENVS" --checkpoint "$CKPT" --out "$OUT_C"

mkdir -p docs/results
cp "programs/results/${NAME}.md" "docs/results/${NAME}.md"
echo "[done] host: docs/results/${NAME}.md (checkpoint in container: ${CKPT})"
echo "${CKPT}" > "programs/results/${NAME}.ckptpath"
