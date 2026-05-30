#!/usr/bin/env bash
# Run on CPU machine AFTER: docker pull + docker build (see README-SETUP.md)
# Produces tarballs you can copy to GPU machine (S3, scp, etc.)

set -euo pipefail

WORKSPACE="${HOME}/isaaclab-workspace"
EXPORT_DIR="${WORKSPACE}/exports"
mkdir -p "${EXPORT_DIR}"

ISAAC_SIM_IMAGE="nvcr.io/nvidia/isaac-sim:5.1.0"
ISAAC_LAB_IMAGE="isaac-lab-base:latest"

echo "=== Saving Docker images to ${EXPORT_DIR} ==="
echo "This may take 30+ minutes and needs ~40GB free disk."

if docker image inspect "${ISAAC_SIM_IMAGE}" &>/dev/null; then
  echo "Saving Isaac Sim base..."
  docker save "${ISAAC_SIM_IMAGE}" -o "${EXPORT_DIR}/isaac-sim-5.1.0.tar"
fi

if docker image inspect "${ISAAC_LAB_IMAGE}" &>/dev/null; then
  echo "Saving Isaac Lab container..."
  docker save "${ISAAC_LAB_IMAGE}" -o "${EXPORT_DIR}/isaac-lab-base.tar"
else
  echo "Build Isaac Lab image first:"
  echo "  cd ~/isaaclab-workspace/IsaacLab && python docker/container.py build"
fi

echo "Done. On GPU machine:"
echo "  docker load -i isaac-sim-5.1.0.tar"
echo "  docker load -i isaac-lab-base.tar"
