#!/usr/bin/env bash
# T1 — Fine-tune GR00T N1.7-3B on LIBERO spatial with LoRA
#
# Hardware: 1× GPU ≥40 GB VRAM (L40S, A100-80G recommended)
#           Single GPU: set NUM_GPUS=1, GLOBAL_BATCH_SIZE=8, grad_accum=8 (effective batch=64)
#
# Usage (single GPU):
#   bash programs/t1_groot_lora/run_finetune.sh
#
# Usage (multi-GPU, 8× L40S for benchmark replication):
#   NUM_GPUS=8 GLOBAL_BATCH_SIZE=640 MAX_STEPS=20000 bash programs/t1_groot_lora/run_finetune.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GROOT_SRC="/tmp/Isaac-GR00T"
PYTHON="/home/zeus/miniconda3/envs/groot_env/bin/python"

# ── Config ────────────────────────────────────────────────────────────────────
NUM_GPUS="${NUM_GPUS:-1}"
MAX_STEPS="${MAX_STEPS:-5000}"        # 20000 for full benchmark; 5000 for quick demo
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
SAVE_STEPS="${SAVE_STEPS:-500}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.7-3B}"
DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/programs/t1_groot_lora/datasets/libero_spatial_no_noops}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/programs/checkpoints/groot_n17/libero_spatial_ft}"

echo "=== T1 GR00T Fine-tune ==="
echo "  model:    $BASE_MODEL"
echo "  dataset:  $DATASET_DIR"
echo "  output:   $OUTPUT_DIR"
echo "  gpus:     $NUM_GPUS"
echo "  steps:    $MAX_STEPS  batch: $GLOBAL_BATCH_SIZE"

mkdir -p "$OUTPUT_DIR"

# ── Launch training ───────────────────────────────────────────────────────────
cd "$GROOT_SRC"
NUM_GPUS="$NUM_GPUS" \
MAX_STEPS="$MAX_STEPS" \
GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
SAVE_STEPS="$SAVE_STEPS" \
  "$PYTHON" -m gr00t.experiment.runner \
    finetune \
    --base-model-path "$BASE_MODEL" \
    --dataset-path "$DATASET_DIR" \
    --embodiment-tag LIBERO_PANDA \
    --output-dir "$OUTPUT_DIR" \
    --state-dropout-prob 0.2 \
    --no-wandb \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo ""
echo "=== Fine-tuning complete ==="
echo "Checkpoint: $OUTPUT_DIR/checkpoint-$MAX_STEPS"
echo ""
echo "To evaluate the fine-tuned model:"
echo "  CHECKPOINT=$OUTPUT_DIR/checkpoint-$MAX_STEPS bash programs/t1_groot_lora/run_eval.sh"
