#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-/home/zeus/content/Humanoid}"
ACTION="${1:-}"

usage() {
  cat <<'EOF'
Usage:
  bash thesis/scripts/machine_switch.sh bootstrap
  bash thesis/scripts/machine_switch.sh train
  bash thesis/scripts/machine_switch.sh sync
  bash thesis/scripts/machine_switch.sh status

Environment:
  PROJECT_DIR   Remote repo path, default /home/zeus/content/Humanoid
  GITHUB_TOKEN  Needed for private GitHub clone/pull/push
  HF_TOKEN      Needed for Hugging Face upload/download
  HF_REPO_ID    Defaults to mitvho09/Humanoid-VLA-Artifacts
  NUM_ENVS      Overrides the Phase 3 env count
  MAX_ITERS     Overrides the Phase 3 iteration count
EOF
}

case "$ACTION" in
  bootstrap)
    bash "$SCRIPT_DIR/bootstrap_remote_machine.sh"
    ;;
  train)
    cd "$PROJECT_DIR"
    bash thesis/scripts/30_vision_vla.sh
    ;;
  sync)
    cd "$PROJECT_DIR"
    bash thesis/scripts/sync_phase3_artifacts.sh
    ;;
  status)
    cd "$PROJECT_DIR"
    {
      echo "Project: $PROJECT_DIR"
      echo "Date: $(date -u +%FT%TZ)"
      echo
      echo "Git:"
      git status --short
      echo
      echo "Training process:"
      ps -ef | grep -E 'custom_train.py|30_vision_vla.sh' | grep -v grep || true
      echo
      echo "GPU:"
      nvidia-smi --query-gpu=name,memory.used,utilization.gpu,power.draw --format=csv,noheader || true
      echo
      echo "Latest log tail:"
      tail -n 40 thesis/logs/g1_vision/nohup.out 2>/dev/null || true
    }
    ;;
  *)
    usage
    exit 2
    ;;
esac
