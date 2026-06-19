#!/usr/bin/env bash
# Thin run helper for the wakeboarding task. Reuses the EXISTING humanoid-isaaclab image.
#
# Image build + machine-switching (push/pull/save/load) is ALREADY handled by the repo's
# existing helper — do NOT duplicate it:
#   thesis/scripts/docker_image_portability.sh  {tag|save|load|push|pull}
#
# This script is only convenience wrappers around docker compose:
#   ./docker/run.sh shell          # interactive shell (DEBUGGING — do first GPU pass here)
#   ./docker/run.sh smoke          # run the GPU smoke test
#   ./docker/run.sh train STAGE    # STAGE = stage1 | stage2 | smoke
set -euo pipefail
COMPOSE="docker compose -f docker/docker-compose.yaml"
cd "$(dirname "$0")/.."

case "${1:-help}" in
  shell) $COMPOSE run --rm wakeboard bash ;;
  smoke) $COMPOSE run --rm wakeboard bash scripts/00_smoke.sh ;;
  train)
    STAGE="${2:?usage: run.sh train stage1|stage2|smoke}"
    $COMPOSE run --rm wakeboard python train.py --config "configs/${STAGE}.yaml" --headless ;;
  *)
    sed -n '2,14p' "$0"
    echo "For image push/pull/save/load use: thesis/scripts/docker_image_portability.sh" ;;
esac
