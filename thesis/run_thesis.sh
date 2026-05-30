#!/usr/bin/env bash
set -euo pipefail

THESIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$THESIS_DIR"

source scripts/lib.sh
source config.env

MODE="all"
START_FROM=""
NO_AUTOSAVE=0

usage() {
  cat <<'EOF'
Usage: bash run_thesis.sh [--cpu-prep|--all] [--from STEP_ID] [--no-autosave]

Modes:
  --cpu-prep    Run only CPU-safe code-prep steps: 00_setup and 11_g1_language.
  --all         Run the full idempotent pipeline, stopping if GPU is required.

Recommended now, while on CPU:
  bash thesis/run_thesis.sh --cpu-prep --no-autosave

After switching Lightning to GPU:
  edit thesis/config.env -> USE_GPU=1, CPU_PREP_ONLY=0
  bash thesis/run_thesis.sh --all
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --cpu-prep) MODE="cpu-prep" ;;
    --all) MODE="all" ;;
    --from) START_FROM="${2:?missing step id after --from}"; shift ;;
    --no-autosave) NO_AUTOSAVE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

if [ "${CPU_PREP_ONLY:-0}" = "1" ] && [ "$MODE" = "all" ]; then
  MODE="cpu-prep"
fi

if [ "${ENABLE_AUTOSAVE:-0}" = "1" ] && [ "$NO_AUTOSAVE" = "0" ]; then
  mkdir -p logs
  if command -v pgrep >/dev/null 2>&1; then
    pgrep -f "scripts/autosave.sh" >/dev/null || nohup bash scripts/autosave.sh > logs/autosave.log 2>&1 &
  else
    nohup bash scripts/autosave.sh > logs/autosave.log 2>&1 &
  fi
fi

run_step() {
  local step_id="$1"
  local script_name="$2"
  local title="$3"

  if [ -n "$START_FROM" ] && [ "$START_FROM" != "$step_id" ] && [ ! -f "$STATE_DIR/.started_from_$START_FROM" ]; then
    log_step "SKIP $step_id until --from $START_FROM"
    return 0
  fi
  if [ "$START_FROM" = "$step_id" ]; then
    touch "$STATE_DIR/.started_from_$START_FROM"
  fi

  if [ "$MODE" = "cpu-prep" ]; then
    case "$step_id" in
      00_setup|11_g1_language) ;;
      *) log_step "CPU-PREP SKIP $step_id ($title)"; return 0 ;;
    esac
  fi

  if is_done "$step_id"; then
    log_step "SKIP $step_id (already done)"
    return 0
  fi

  set_state "$step_id" "in_progress"
  log_step "START $step_id $title"
  set +e
  bash "scripts/$script_name"
  rc=$?
  set -e
  if [ "$rc" -eq 0 ]; then
    mark_done "$step_id"
    set_state "$step_id" "done"
    bash scripts/checkpoint.sh "$step_id $title" || true
    log_step "DONE $step_id"
  elif [ "$rc" -eq 78 ]; then
    set_state "$step_id" "waiting"
    log_step "WAIT $step_id ($title)"
    exit 0
  else
    set_state "$step_id" "failed"
    log_step "FAILED $step_id ($title), exit $rc"
    exit "$rc"
  fi
}

run_step 00_setup "00_setup.sh" "CPU scaffold and verification"
run_step 01_gr00t_install "01_gr00t_install.sh" "Install Isaac-GR00T"
run_step 02_gr00t_demo "02_gr00t_demo.sh" "GR00T inference smoke test"
run_step 03_gr00t_gendata "03_gr00t_gendata.sh" "Prepare GR00T language dataset"
run_step 04_gr00t_finetune "04_gr00t_finetune.sh" "Fine-tune GR00T"
run_step 05_gr00t_eval "05_gr00t_eval.sh" "Evaluate fine-tuned GR00T"
run_step 10_g1_baseline "10_g1_baseline_train.sh" "Train G1 baseline"
run_step 11_g1_language "11_g1_language_cond.sh" "Author G1 language-conditioning code"
run_step 12_g1_train_eval "12_g1_train_eval.sh" "Train/evaluate language-conditioned G1"
run_step 20_custom_task "20_custom_task.sh" "Train/evaluate custom task"
run_step 99_collect_results "99_collect_results.sh" "Collect final results"

if [ "$MODE" = "cpu-prep" ]; then
  set_state "gpu_handoff" "waiting"
  log_step "CPU PREP COMPLETE - switch Lightning to GPU for --all"
  exit 0
fi

set_state "complete" "done"
log_step "ALL STEPS COMPLETE"
