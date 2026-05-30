#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"

PROJECT="$WORKSPACE_DIR/my-humanoid-project"
if [ ! -d "$PROJECT" ]; then
  echo "Missing my-humanoid-project at $PROJECT" | md_log "11-g1-language" "STEP 11 blocked"
  exit 1
fi

python3 -m compileall -q "$PROJECT/my_humanoid_project"
OUT="$THESIS_DIR/logs/11_language_import.txt"
PYTHONPATH="$PROJECT" run_cmd_capture "$OUT" python3 -c "from my_humanoid_project.language_commands import COMMANDS, embedding_for_text; print(len(COMMANDS)); print(len(embedding_for_text('pick up the red cube')))"

{
  echo "## Authored package"
  echo
  echo "- \`my-humanoid-project/my_humanoid_project/language_commands.py\`"
  echo "- \`my-humanoid-project/my_humanoid_project/tasks/g1_language_pickplace_cfg.py\`"
  echo "- \`my-humanoid-project/my_humanoid_project/tasks/__init__.py\`"
  echo
  echo "## Verification"
  echo
  echo '```text'
  cat "$OUT"
  echo '```'
  echo
  echo "- [x] CPU import works without Isaac Sim"
  echo "- [x] Deterministic language embeddings are available"
  echo "- [ ] GPU validation: register env inside Isaac Lab and run random-policy step"
} | md_log "11-g1-language" "STEP 11 G1 language-conditioning code"

