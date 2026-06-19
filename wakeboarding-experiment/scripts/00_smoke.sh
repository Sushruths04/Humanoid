#!/usr/bin/env bash
# Smoke test — build env + reach PPO (PLAN §14 task 6). Run inside the Isaac Lab container.
set -euo pipefail
cd "$(dirname "$0")/.."
python train.py --config configs/smoke.yaml --headless --num_envs "${NUM_ENVS:-16}" \
  --max_iterations "${MAX_ITERS:-2}"
echo "SMOKE OK if a model_*.pt was written under runs/wakeboard_smoke/"
