#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_gpu "20_custom_task" || exit 78

{
  echo "## Custom task gate"
  echo
  echo "Use the language-conditioned G1 task from STEP 11 and add the red-vs-blue distractor after STEP 12 has a working policy."
  echo
  echo "- [ ] Add distractor object"
  echo "- [ ] Gate reward/success by color command"
  echo "- [ ] Train small on L4"
  echo "- [ ] Scale only if small run succeeds"
} | md_log "20-custom-task" "STEP 20 custom task gate"
exit 1

