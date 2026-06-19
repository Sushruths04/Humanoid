#!/usr/bin/env bash
# One-stop machine-switch + run helper. Build once, push, pull anywhere.
#
#   ./docker/run.sh build         # build the image locally
#   ./docker/run.sh push          # push to GHCR (after: docker login ghcr.io)
#   ./docker/run.sh pull          # pull on a fresh Lightning machine
#   ./docker/run.sh shell         # interactive shell in the container (DEBUGGING)
#   ./docker/run.sh smoke         # run the GPU smoke test
#   ./docker/run.sh train STAGE   # STAGE = stage1 | stage2 | smoke
#   ./docker/run.sh save|load     # tarball fallback if no registry access
set -euo pipefail
IMAGE="ghcr.io/sushruths04/wakeboard-isaaclab:latest"
cd "$(dirname "$0")/.."

case "${1:-help}" in
  build) docker compose -f docker/docker-compose.yaml build ;;
  push)  docker push "$IMAGE" ;;
  pull)  docker pull "$IMAGE" ;;
  shell) docker compose -f docker/docker-compose.yaml run --rm wakeboard bash ;;
  smoke) docker compose -f docker/docker-compose.yaml run --rm wakeboard bash scripts/00_smoke.sh ;;
  train)
    STAGE="${2:?usage: run.sh train stage1|stage2|smoke}"
    docker compose -f docker/docker-compose.yaml run --rm wakeboard \
      python train.py --config "configs/${STAGE}.yaml" --headless ;;
  save)  docker save "$IMAGE" -o wakeboard-image.tar && echo "wrote wakeboard-image.tar" ;;
  load)  docker load -i wakeboard-image.tar ;;
  *) sed -n '2,12p' "$0" ;;
esac
