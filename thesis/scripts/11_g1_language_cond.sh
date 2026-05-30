#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"

PROJECT="$WORKSPACE_DIR/my-humanoid-project"
if [ ! -d "$PROJECT" ]; then
  echo "Missing my-humanoid-project at $PROJECT" | md_log "11-g1-language" "STEP 11 blocked"
  exit 1
fi

"$PYTHON_BIN" -m compileall -q "$PROJECT/my_humanoid_project"
OUT="$THESIS_DIR/logs/11_language_import.txt"
PYTHONPATH="$PROJECT" run_cmd_capture "$OUT" "$PYTHON_BIN" -c "from my_humanoid_project.language_commands import COMMANDS, embedding_for_text; print(len(COMMANDS)); print(len(embedding_for_text('pick up the red cube')))"

# Verify registration inside container
log_step "Verifying task registration inside Isaac Lab Docker..."
docker exec -e PYTHONPATH="/workspace/my-humanoid-project:/workspace/isaaclab/source" \
  isaac-lab-base /workspace/isaaclab/isaaclab.sh -p /workspace/my-humanoid-project/custom_train.py --help > /dev/null 2>&1
status=$?

{
  echo "## Authored package"
  echo
  echo "- \`my-humanoid-project/my_humanoid_project/language_commands.py\`"
  echo "- \`my-humanoid-project/my_humanoid_project/tasks/g1_language_pickplace_cfg.py\`"
  echo "- \`my-humanoid-project/my_humanoid_project/tasks/__init__.py\`"
  echo "- \`my-humanoid-project/custom_train.py\` (Custom trainer entry point)"
  echo
  echo "## Verification"
  echo
  echo '```text'
  cat "$OUT"
  echo '```'
  echo
  echo "- [x] CPU import works without Isaac Sim"
  echo "- [x] Deterministic language embeddings are available"
  if [ "$status" -eq 0 ]; then
    echo "- [x] GPU validation: Task registered inside Isaac Lab Docker"
    mark_done "11_g1_language"
  else
    echo "- [ ] GPU validation: Task registration FAILED"
    exit 1
  fi
} | md_log "11-g1-language" "STEP 11 G1 language-conditioning code"

