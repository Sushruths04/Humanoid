#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/zeus/content/Humanoid}"
HF_REPO_ID="${HF_REPO_ID:-<your-hf-namespace>/Humanoid-VLA-Artifacts}"
ARTIFACT_DIR="$PROJECT_DIR/thesis/artifacts/phase3"
export HF_REPO_ID ARTIFACT_DIR

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is required to upload artifacts."
  exit 2
fi

mkdir -p "$ARTIFACT_DIR"

echo "[sync] Collecting host logs/checkpoints"
rsync -a --delete "$PROJECT_DIR/thesis/logs/" "$ARTIFACT_DIR/thesis_logs/"
rsync -a --delete "$PROJECT_DIR/thesis/checkpoints/" "$ARTIFACT_DIR/thesis_checkpoints/"
cp -f "$PROJECT_DIR/AGENT_HANDOFF.md" "$ARTIFACT_DIR/" 2>/dev/null || true
cp -f "$PROJECT_DIR/FINAL_RESULTS.md" "$ARTIFACT_DIR/" 2>/dev/null || true

if docker ps --format '{{.Names}}' | grep -qx 'isaac-lab-base'; then
  echo "[sync] Collecting Isaac Lab container logs"
  rm -rf "$ARTIFACT_DIR/isaaclab_logs"
  docker cp isaac-lab-base:/workspace/isaaclab/logs "$ARTIFACT_DIR/isaaclab_logs" 2>/dev/null || true
fi

python3 - <<'PY'
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("huggingface_hub") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "huggingface_hub"])
PY

python3 - <<'PY'
import os
from huggingface_hub import HfApi

repo_id = os.environ["HF_REPO_ID"]
folder = os.environ["ARTIFACT_DIR"]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path=folder,
    repo_id=repo_id,
    repo_type="model",
    path_in_repo="phase3",
)
print(f"[sync] Uploaded {folder} to https://huggingface.co/{repo_id}/tree/main/phase3")
PY
