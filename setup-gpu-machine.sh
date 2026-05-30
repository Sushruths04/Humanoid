#!/usr/bin/env bash
# Quick-start on a GPU machine after CPU prep work.
# Usage: bash setup-gpu-machine.sh [your-github-username]

set -euo pipefail

GITHUB_USER="${1:-YOUR-USERNAME}"
WORKSPACE="${HOME}/isaaclab-workspace"

mkdir -p "${WORKSPACE}"
cd "${WORKSPACE}"

if [[ ! -d IsaacLab ]]; then
  git clone https://github.com/isaac-sim/IsaacLab.git
fi

if [[ ! -d my-humanoid-project ]]; then
  git clone "https://github.com/${GITHUB_USER}/my-humanoid-project.git" || {
    echo "Clone my-humanoid-project manually or push it from CPU machine first."
  }
fi

cd IsaacLab

# If you transferred a saved image tarball on this machine:
# docker load -i "${WORKSPACE}/exports/isaac-lab-base.tar"

python docker/container.py start
python docker/container.py enter

echo "Inside container. Example training command:"
echo "isaaclab -p scripts/reinforcement_learning/rsl_rl/train.py \\"
echo "  --task Isaac-Velocity-Flat-H1-v0 --headless --num_envs 2048"
