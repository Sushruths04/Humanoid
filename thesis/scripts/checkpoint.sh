#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THESIS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_DIR="$(cd "$THESIS_DIR/.." && pwd)"

source "$THESIS_DIR/config.env"

cd "$WORKSPACE_DIR"
git add thesis/PROGRESS thesis/state thesis/scripts thesis/run_thesis.sh thesis/STATE.json thesis/config.env thesis/*.md thesis/.gitignore my-humanoid-project 2>/dev/null || true
if ! git diff --cached --quiet; then
  git commit -m "checkpoint: ${1:-manual} $(date -u +%FT%TZ)"
fi
if [ "${ENABLE_GIT_PUSH:-1}" = "1" ]; then
  git push || true
fi

if [ -n "${HF_REPO:-}" ] && [ -d "$THESIS_DIR/checkpoints" ] && find "$THESIS_DIR/checkpoints" -type f ! -name .gitkeep | grep -q .; then
  if command -v hf >/dev/null 2>&1; then
    hf upload "$HF_REPO" "$THESIS_DIR/checkpoints" checkpoints --repo-type model || true
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli upload "$HF_REPO" "$THESIS_DIR/checkpoints" checkpoints --repo-type model || true
  fi
fi
