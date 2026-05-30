#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"

mkdir -p \
  "$PROGRESS_DIR/step-logs" \
  "$PROGRESS_DIR/reading" \
  "$THESIS_DIR/scripts" \
  "$THESIS_DIR/state" \
  "$THESIS_DIR/checkpoints" \
  "$THESIS_DIR/data" \
  "$THESIS_DIR/logs"

touch \
  "$THESIS_DIR/checkpoints/.gitkeep" \
  "$THESIS_DIR/data/.gitkeep" \
  "$THESIS_DIR/logs/.gitkeep" \
  "$PROGRESS_DIR/reading/.gitkeep" \
  "$PROGRESS_DIR/step-logs/.gitkeep" \
  "$THESIS_DIR/state/.gitkeep"

if [ ! -f "$PROGRESS_DIR/TIMELINE.md" ]; then
  {
    echo "# Thesis Timeline"
    echo
  } > "$PROGRESS_DIR/TIMELINE.md"
fi

if [ ! -f "$PROGRESS_DIR/00-overview.md" ]; then
  {
    echo "# Thesis Execution Overview"
    echo
    echo "- Base workspace: \`$WORKSPACE_DIR\`"
    echo "- Thesis directory: \`$THESIS_DIR\`"
    echo "- CPU-prep goal: author scripts and project code before switching to GPU."
    echo "- GPU handoff: set \`USE_GPU=1\` and \`CPU_PREP_ONLY=0\` in \`config.env\`."
  } > "$PROGRESS_DIR/00-overview.md"
fi

{
  echo "## Commands"
  echo
  echo '```bash'
  echo "mkdir -p PROGRESS/{step-logs,reading} scripts state checkpoints data logs"
  echo "touch .gitkeep files for empty directories"
  echo '```'
  echo
  echo "## Verification"
  echo
  echo '```text'
  find "$THESIS_DIR" -maxdepth 2 -type d | sort
  echo '```'
  echo
  echo "- [x] Directory tree exists"
  echo "- [x] STATE.json exists"
  echo "- [x] config.env exists"
} | md_log "00-setup" "STEP 00 setup"

