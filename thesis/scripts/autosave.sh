#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THESIS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_DIR="$(cd "$THESIS_DIR/.." && pwd)"

cd "$WORKSPACE_DIR"
while true; do
  git add thesis/PROGRESS thesis/state thesis/scripts thesis/run_thesis.sh thesis/STATE.json thesis/config.env thesis/*.md my-humanoid-project 2>/dev/null || true
  git commit -m "autosave $(date -u +%FT%TZ)" >/dev/null 2>&1 || true
  if [ "${ENABLE_GIT_PUSH:-1}" = "1" ]; then
    git push >/dev/null 2>&1 || true
  fi
  sleep 1800
done
