#!/usr/bin/env bash
# Stage I — discovery, pull-speed curriculum 10->30 km/h (PLAN §6, §14 task 8).
set -euo pipefail
cd "$(dirname "$0")/.."
# L4-safe default (24GB). On L40S/Modal bump up: NUM_ENVS=4096 bash scripts/10_train_stage1.sh
python train.py --config configs/stage1.yaml --headless \
  --num_envs "${NUM_ENVS:-2048}" --max_iterations "${MAX_ITERS:-5000}"
