#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"

RESULTS="$PROGRESS_DIR/RESULTS.md"
{
  echo "# Results"
  echo
  echo "_Generated $(date -u +%FT%TZ)_"
  echo
  echo "## Current State"
  echo
  echo '```json'
  cat "$THESIS_DIR/STATE.json"
  echo '```'
  echo
  echo "## Executed Checks"
  echo
  echo "| Step | Status | Evidence |"
  echo "|---|---|---|"
  echo "| 00 setup | done | state/00_setup.done |"
  echo "| 01 GR00T install | done | logs/01_gr00t_import.txt |"
  echo "| 02 GR00T GPU demo | waiting on Hugging Face auth | results/gr00t_demo/summary.txt |"
  echo "| 11 G1 language scaffold | done | logs/11_language_import.txt |"
  echo
  if [ -f "$THESIS_DIR/results/gr00t_demo/summary.txt" ]; then
    echo "## GR00T Demo Summary"
    echo
    echo '```text'
    cat "$THESIS_DIR/results/gr00t_demo/summary.txt"
    echo '```'
    echo
  fi
  echo "## Remaining Blocker"
  echo
  echo "GR00T N1.7 model loading requires Hugging Face authentication and access to gated repo nvidia/Cosmos-Reason2-2B."
  echo
  echo '```bash'
  echo "hf auth login"
  echo "bash thesis/run_thesis.sh --all --from 02_gr00t_demo --no-autosave"
  echo '```'
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
