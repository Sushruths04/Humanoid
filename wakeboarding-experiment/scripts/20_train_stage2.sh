#!/usr/bin/env bash
# Stage II — deployable: slowed-traj tracking + AMP + domain randomization (PLAN §6.2).
# Requires a Stage-I checkpoint (set STAGE1_CKPT).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${STAGE1_CKPT:?set STAGE1_CKPT=path/to/ckpt_20_stage1_30.pt}"
# L4-safe default (24GB). On L40S/Modal bump up: NUM_ENVS=8192 bash scripts/20_train_stage2.sh
python train.py --config configs/stage2.yaml --headless \
  --num_envs "${NUM_ENVS:-2048}" --max_iterations "${MAX_ITERS:-10000}" \
  --resume "${STAGE1_CKPT}"
