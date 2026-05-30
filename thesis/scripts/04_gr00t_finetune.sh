#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_gpu "04_gr00t_finetune" || exit 78

if [ ! -d "$WORKSPACE_DIR/Isaac-GR00T" ]; then
  echo "Isaac-GR00T is missing. Run STEP 01 first." | md_log "04-gr00t-finetune" "STEP 04 blocked"
  exit 1
fi

mkdir -p "$THESIS_DIR/checkpoints/gr00t"
{
  echo "## GPU run template"
  echo
  echo "Verify the current upstream fine-tune command before execution."
  echo
  echo '```bash'
  echo "cd $WORKSPACE_DIR/Isaac-GR00T"
  echo "nvidia-smi"
  echo "# Example shape, adjust to current README:"
  echo "python scripts/gr00t_finetune.py --output-dir $THESIS_DIR/checkpoints/gr00t --max-steps ${GR00T_FT_STEPS:-2000} --save-interval ${GR00T_SAVE_INTERVAL:-500}"
  echo '```'
  echo
  echo "Exit intentionally until the current README command is verified."
} | md_log "04-gr00t-finetune" "STEP 04 GR00T finetune command gate"
exit 1

