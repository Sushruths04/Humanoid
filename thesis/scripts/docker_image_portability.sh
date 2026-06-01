#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-isaac-lab-base}"
IMAGE_TAG="${IMAGE_TAG:-humanoid-isaaclab:latest}"
ARCHIVE_PATH="${ARCHIVE_PATH:-$HOME/humanoid-isaaclab-latest.tar}"
REGISTRY_IMAGE="${REGISTRY_IMAGE:-}"

usage() {
  cat <<'EOF'
Usage:
  thesis/scripts/docker_image_portability.sh tag
  thesis/scripts/docker_image_portability.sh save
  thesis/scripts/docker_image_portability.sh load
  thesis/scripts/docker_image_portability.sh push
  thesis/scripts/docker_image_portability.sh pull

Environment:
  IMAGE_NAME      Existing local image name. Default: isaac-lab-base
  IMAGE_TAG       Portable local tag. Default: humanoid-isaaclab:latest
  ARCHIVE_PATH    Docker tar path for save/load. Default: $HOME/humanoid-isaaclab-latest.tar
  REGISTRY_IMAGE  Remote registry tag for push/pull, for example:
                  ghcr.io/<github-user>/humanoid-isaaclab:latest

Notes:
  - Use a Docker registry for frequent machine changes.
  - Use Hugging Face for checkpoints, logs, datasets, and result artifacts.
  - Do not commit tokens. Login with `docker login ghcr.io` or your registry CLI.
EOF
}

require_registry_image() {
  if [ -z "$REGISTRY_IMAGE" ]; then
    echo "REGISTRY_IMAGE is required for this action." >&2
    exit 64
  fi
}

resolve_source_image() {
  local candidate
  for candidate in "$IMAGE_NAME" "$IMAGE_TAG"; do
    if docker image inspect "$candidate" >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done

  echo "Could not find a source image. Tried: $IMAGE_NAME, $IMAGE_TAG" >&2
  exit 1
}

case "${1:-}" in
  tag)
    source_image="$(resolve_source_image)"
    docker tag "$source_image" "$IMAGE_TAG"
    docker image ls "$IMAGE_TAG"
    ;;
  save)
    source_image="$(resolve_source_image)"
    docker tag "$source_image" "$IMAGE_TAG"
    docker save "$IMAGE_TAG" -o "$ARCHIVE_PATH"
    ls -lh "$ARCHIVE_PATH"
    ;;
  load)
    docker load -i "$ARCHIVE_PATH"
    docker image ls "$IMAGE_TAG"
    ;;
  push)
    require_registry_image
    source_image="$(resolve_source_image)"
    docker tag "$source_image" "$REGISTRY_IMAGE"
    docker push "$REGISTRY_IMAGE"
    ;;
  pull)
    require_registry_image
    docker pull "$REGISTRY_IMAGE"
    docker tag "$REGISTRY_IMAGE" "$IMAGE_NAME"
    docker image ls "$IMAGE_NAME"
    ;;
  *)
    usage
    exit 64
    ;;
esac
