#!/usr/bin/env bash
# Stage II — deployable: slowed-traj tracking + AMP + domain randomization (PLAN §6.2).
# Requires a Stage-I checkpoint (set STAGE1_CKPT).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${STAGE1_CKPT:?set STAGE1_CKPT=path/to/ckpt_20_stage1_30.pt}"
python train.py --config configs/stage2.yaml --headless \
  --num_envs "${NUM_ENVS:-8192}" --max_iterations "${MAX_ITERS:-10000}" \
  --resume "${STAGE1_CKPT}"
