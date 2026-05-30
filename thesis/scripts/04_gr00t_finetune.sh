#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_gpu "04_gr00t_finetune" || exit 78

GR00T_DIR="$WORKSPACE_DIR/Isaac-GR00T"
RESULT_DIR="$THESIS_DIR/results/gr00t_finetune_smoke"
LOG_FILE="$THESIS_DIR/logs/04_gr00t_finetune.log"
SUMMARY_FILE="$RESULT_DIR/summary.txt"
OUTPUT_DIR="$THESIS_DIR/checkpoints/gr00t_smoke"
DATASET_PATH="$GR00T_DIR/demo_data/cube_to_bowl_5"
MODALITY_CONFIG_PATH="$GR00T_DIR/examples/SO100/so100_config.py"

if [ ! -d "$GR00T_DIR" ]; then
  echo "Isaac-GR00T is missing. Run STEP 01 first." | md_log "04-gr00t-finetune" "STEP 04 blocked"
  exit 1
fi

mkdir -p "$RESULT_DIR" "$OUTPUT_DIR" "$(dirname "$LOG_FILE")"

if [ ! -d "$DATASET_PATH" ]; then
  echo "Demo dataset missing: $DATASET_PATH" | md_log "04-gr00t-finetune" "STEP 04 blocked"
  exit 1
fi
if [ ! -f "$MODALITY_CONFIG_PATH" ]; then
  echo "Modality config missing: $MODALITY_CONFIG_PATH" | md_log "04-gr00t-finetune" "STEP 04 blocked"
  exit 1
fi

export HF_HOME="$THESIS_DIR/cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export PYTHONUNBUFFERED=1
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export NUM_GPUS=1
export SAVE_STEPS="${GR00T_SAVE_INTERVAL:-2}"
export MAX_STEPS="${GR00T_FT_STEPS:-2}"
export GLOBAL_BATCH_SIZE="${GR00T_GLOBAL_BATCH_SIZE:-2}"
export DATALOADER_NUM_WORKERS="${GR00T_WORKERS:-0}"
export SAVE_ONLY_MODEL=1
export USE_WANDB=0
export SHARD_SIZE="${GR00T_SHARD_SIZE:-16}"
export NUM_SHARDS_PER_EPOCH="${GR00T_NUM_SHARDS_PER_EPOCH:-5}"
export EPISODE_SAMPLING_RATE="${GR00T_EPISODE_SAMPLING_RATE:-1.0}"

{
  echo "# STEP 04 GR00T finetune smoke"
  echo
  echo "_$(date -u +%FT%TZ)_"
  echo
  echo "## GPU"
  echo
  echo '```text'
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
  echo '```'
  echo
  echo "## Command"
  echo
  echo '```bash'
  echo "cd $GR00T_DIR"
  echo "bash examples/finetune.sh --base-model-path nvidia/GR00T-N1.7-3B --dataset-path $DATASET_PATH --embodiment-tag NEW_EMBODIMENT --modality-config-path $MODALITY_CONFIG_PATH --output-dir $OUTPUT_DIR --num-gpus 1 --max-steps ${GR00T_FT_STEPS:-2} --save-steps ${GR00T_SAVE_INTERVAL:-2} --global-batch-size ${GR00T_GLOBAL_BATCH_SIZE:-2} --dataloader-num-workers ${GR00T_WORKERS:-2} --save-only-model -- --skip-weight-loading"
  echo '```'
  echo
  echo "This is a smoke finetune: tiny step count, single GPU, and checkpoint-only output."
} | md_log "04-gr00t-finetune" "STEP 04 GR00T finetune smoke"

cd "$GR00T_DIR"

set +e
bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$DATASET_PATH" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$MODALITY_CONFIG_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --save-only-model \
  -- \
  --skip-weight-loading \
  2>&1 | tee "$LOG_FILE"
status=${PIPESTATUS[0]}
set -e

{
  echo "# GR00T finetune smoke summary"
  echo
  echo "Generated: $(date -u +%FT%TZ)"
  echo "Exit status: $status"
  echo "Output dir: $OUTPUT_DIR"
  echo "Log: $LOG_FILE"
  echo
  echo "## Key lines"
  grep -E "Loading checkpoint shards|train|loss|step|saved|epoch|checkpoint|done|error|Traceback" "$LOG_FILE" || true
} > "$SUMMARY_FILE"

{
  echo
  echo "## Result"
  echo
  echo '```text'
  cat "$SUMMARY_FILE"
  echo '```'
} | md_log "04-gr00t-finetune" "STEP 04 result"

if [ "$status" -ne 0 ]; then
  exit "$status"
fi

test -s "$SUMMARY_FILE"
test -d "$OUTPUT_DIR"
mark_done "04_gr00t_finetune"
exit 0
