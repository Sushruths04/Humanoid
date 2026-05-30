#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_gpu "05_gr00t_eval" || exit 78

if ! find "$THESIS_DIR/checkpoints/gr00t" -type f 2>/dev/null | grep -q .; then
  echo "No GR00T checkpoint found under thesis/checkpoints/gr00t." | md_log "05-gr00t-eval" "STEP 05 blocked"
  exit 1
fi

{
  echo "## Evaluation template"
  echo
  echo "| Prompt | Result | Notes |"
  echo "|---|---:|---|"
  echo "| pick up the red cube | pending | run inference with fine-tuned checkpoint |"
  echo "| pick up the blue cube | pending | held-out prompt |"
  echo "| walk to the cube | pending | behavior-change check |"
  echo "| stand still | pending | negative control |"
  echo "| pick up the object | pending | paraphrase |"
} | md_log "05-gr00t-eval" "STEP 05 GR00T eval template"
exit 1

