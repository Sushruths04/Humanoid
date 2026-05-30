#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_gpu "05_gr00t_eval" || exit 78

CHECKPOINT_DIR="$THESIS_DIR/checkpoints/gr00t_smoke/checkpoint-2000"
DATASET_PATH="$WORKSPACE_DIR/Isaac-GR00T/demo_data/cube_to_bowl_5"
RESULT_DIR="$THESIS_DIR/results/gr00t_eval_smoke"
LOG_FILE="$THESIS_DIR/logs/05_gr00t_eval.log"
SUMMARY_FILE="$RESULT_DIR/summary.txt"
PLOT_FILE="$RESULT_DIR/traj_0.jpeg"

if [ ! -d "$CHECKPOINT_DIR" ]; then
  echo "No GR00T smoke checkpoint found under $CHECKPOINT_DIR." | md_log "05-gr00t-eval" "STEP 05 blocked"
  exit 1
fi
if [ ! -d "$DATASET_PATH" ]; then
  echo "Dataset missing: $DATASET_PATH" | md_log "05-gr00t-eval" "STEP 05 blocked"
  exit 1
fi

mkdir -p "$RESULT_DIR" "$(dirname "$LOG_FILE")"

export HF_HOME="$THESIS_DIR/cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export PYTHONUNBUFFERED=1
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

{
  echo "# STEP 05 GR00T eval smoke"
  echo
  echo "_$(date -u +%FT%TZ)_"
  echo
  echo "## Command"
  echo
  echo '```bash'
  echo "cd $WORKSPACE_DIR/Isaac-GR00T"
  echo "$PYTHON_BIN gr00t/eval/open_loop_eval.py --dataset-path $DATASET_PATH --embodiment-tag NEW_EMBODIMENT --model-path $CHECKPOINT_DIR --traj-ids 0 --steps 64 --action-horizon 16 --save-plot-path $PLOT_FILE --modality-keys single_arm gripper"
  echo '```'
} | md_log "05-gr00t-eval" "STEP 05 GR00T eval smoke"

cd "$WORKSPACE_DIR/Isaac-GR00T"

set +e
"$PYTHON_BIN" gr00t/eval/open_loop_eval.py \
  --dataset-path "$DATASET_PATH" \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path "$CHECKPOINT_DIR" \
  --traj-ids 0 \
  --steps 64 \
  --action-horizon 16 \
  --save-plot-path "$PLOT_FILE" \
  --modality-keys single_arm gripper \
  2>&1 | tee "$LOG_FILE"
status=${PIPESTATUS[0]}
set -e

{
  echo "# GR00T eval smoke summary"
  echo
  echo "Generated: $(date -u +%FT%TZ)"
  echo "Exit status: $status"
  echo "Checkpoint: $CHECKPOINT_DIR"
  echo "Dataset: $DATASET_PATH"
  echo "Plot: $PLOT_FILE"
  echo "Log: $LOG_FILE"
  echo
  echo "## Key lines"
  grep -E "Model loading time|Dataset length|Using [0-9]+ steps|MSE|MAE|Done|Traceback|error" "$LOG_FILE" || true
} > "$SUMMARY_FILE"

{
  echo
  echo "## Result"
  echo
  echo '```text'
  cat "$SUMMARY_FILE"
  echo '```'
} | md_log "05-gr00t-eval" "STEP 05 result"

if [ "$status" -ne 0 ]; then
  exit "$status"
fi

test -s "$SUMMARY_FILE"
test -s "$PLOT_FILE"
mark_done "05_gr00t_eval"
exit 0
