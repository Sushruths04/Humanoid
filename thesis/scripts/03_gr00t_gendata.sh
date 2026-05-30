#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
source "$THESIS_DIR/config.env"

DATA_DIR="$THESIS_DIR/data/gr00t_language_seed"
mkdir -p "$DATA_DIR"
OUT="$DATA_DIR/instructions.jsonl"

python3 - "$OUT" "${GR00T_DATA_TRAJ:-200}" <<'PY'
import json
import sys

path = sys.argv[1]
n = int(sys.argv[2])
commands = [
    "pick up the red cube",
    "pick up the blue cube",
    "walk to the cube",
    "stand still",
]
with open(path, "w", encoding="utf-8") as f:
    for i in range(n):
        f.write(json.dumps({
            "episode_id": i,
            "language_instruction": commands[i % len(commands)],
            "source": "cpu_seed_manifest",
            "status": "needs_lerobot_observations_on_gpu_or_dataset_import",
        }) + "\n")
PY

{
  echo "## CPU seed manifest"
  echo
  echo "Created a small language-instruction manifest. This is not a full LeRobot dataset yet; it is the CPU-side seed used by the GPU/data-import step."
  echo
  echo '```text'
  wc -l "$OUT"
  head -5 "$OUT"
  echo '```'
  echo
  echo "- [x] Language labels exist"
  echo "- [ ] Convert/import observations/actions into LeRobot format"
} | md_log "03-gr00t-gendata" "STEP 03 GR00T data seed"

