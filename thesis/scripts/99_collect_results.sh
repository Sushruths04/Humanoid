#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"

RESULTS="$PROGRESS_DIR/RESULTS.md"
{
  echo "# Results"
  echo
  echo "_Generated $(date -u +%FT%TZ)_"
  echo
  echo "| Phase | Method | Status | Checkpoint | Metrics |"
  echo "|---|---|---|---|---|"
  echo "| 1 | GR00T N1 fine-tune | pending | \`checkpoints/gr00t/\` | success %, held-out prompts |"
  echo "| 2 | G1 language-conditioned RL | pending | \`checkpoints/g1_language/\` | per-command success %, reward curve |"
  echo "| 3 | Custom distractor task | pending | \`checkpoints/custom_task/\` | red-vs-blue command success |"
  echo
  echo "## Step Logs"
  echo
  find "$PROGRESS_DIR/step-logs" -maxdepth 1 -type f -name 'STEP-*.md' -printf '- [[%f]]\n' 2>/dev/null | sort || true
} > "$RESULTS"

{
  echo "## Verification"
  echo
  echo '```text'
  wc -l "$RESULTS"
  echo '```'
} | md_log "99-collect-results" "STEP 99 collect results"

