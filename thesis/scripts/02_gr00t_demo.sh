#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_gpu "02_gr00t_demo" || exit 78

if [ ! -d "$WORKSPACE_DIR/Isaac-GR00T" ]; then
  echo "Isaac-GR00T is missing. Run STEP 01 first." | md_log "02-gr00t-demo" "STEP 02 blocked"
  exit 1
fi

{
  echo "## Ready to run"
  echo
  echo "GPU detected. Next action is to verify the current Isaac-GR00T README and run its official inference demo."
  echo
  echo '```bash'
  echo "cd $WORKSPACE_DIR/Isaac-GR00T"
  echo "nvidia-smi"
  echo "# Run the current repo's documented GR00T-N1-2B inference command here."
  echo '```'
  echo
  echo "Do not mark this step done until an action tensor/output shape is captured."
} | md_log "02-gr00t-demo" "STEP 02 GR00T demo command gate"
exit 1

