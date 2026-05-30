#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export THESIS_DIR="${THESIS_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$(cd "$THESIS_DIR/.." && pwd)}"
export STATE_DIR="$THESIS_DIR/state"
export PROGRESS_DIR="$THESIS_DIR/PROGRESS"

mkdir -p "$STATE_DIR" "$PROGRESS_DIR/step-logs" "$PROGRESS_DIR/reading" "$THESIS_DIR/logs"

if [ -f "$THESIS_DIR/config.env" ]; then
  set -a
  source "$THESIS_DIR/config.env"
  set +a
fi

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x /home/zeus/miniconda3/bin/python ]; then
    export PYTHON_BIN=/home/zeus/miniconda3/bin/python
  else
    export PYTHON_BIN=python3
  fi
fi

log_step() {
  printf '[%s] %s\n' "$(date -u +%T)" "$*"
}

is_done() {
  [ -f "$STATE_DIR/$1.done" ]
}

mark_done() {
  touch "$STATE_DIR/$1.done"
}

set_state() {
  local step="${1:?step required}"
  local status="${2:?status required}"
  "$PYTHON_BIN" - "$THESIS_DIR/STATE.json" "$step" "$status" <<'PY'
import datetime
import json
import os
import sys

path, step, status = sys.argv[1:4]
now = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
if os.path.exists(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
else:
    data = {}
data.setdefault("started_at", now)
data["current_step"] = step
data["status"] = status
data["last_checkpoint"] = now
with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
PY
}

md_log() {
  local step_id="${1:?step id required}"
  local title="${2:?title required}"
  local file="$PROGRESS_DIR/step-logs/STEP-${step_id}.md"
  {
    echo "# $title"
    echo
    echo "_$(date -u +%FT%TZ)_"
    echo
    cat
    echo
  } >> "$file"
  printf -- '- %s | %s | %s\n' "$(date -u +%FT%TZ)" "$step_id" "$title" >> "$PROGRESS_DIR/TIMELINE.md"
}

require_gpu() {
  local step="${1:?step required}"
  if [ "${USE_GPU:-0}" != "1" ]; then
    {
      echo "## Deferred"
      echo
      echo "This step requires a GPU Studio. Current config has USE_GPU=${USE_GPU:-0}."
      echo
      echo "Switch Lightning to an RTX GPU, set USE_GPU=1 and CPU_PREP_ONLY=0 in thesis/config.env, then rerun:"
      echo
      echo '```bash'
      echo "bash thesis/run_thesis.sh --all --from $step"
      echo '```'
    } | md_log "$step" "GPU required"
    return 78
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is unavailable; GPU machine is not ready." | md_log "$step" "GPU missing"
    return 78
  fi
  nvidia-smi >/dev/null
}

run_cmd_capture() {
  local outfile="${1:?outfile required}"
  shift
  {
    printf '$'
    printf ' %q' "$@"
    printf '\n'
    "$@"
  } > "$outfile" 2>&1
}
