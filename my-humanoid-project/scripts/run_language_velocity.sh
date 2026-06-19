#!/usr/bin/env bash
# Language-ON: train then prove it (behavior-separation plot). Run in the Isaac Lab container.
#   bash scripts/run_language_velocity.sh            # smoke (16 envs, 2 iters)
#   FULL=1 bash scripts/run_language_velocity.sh     # full train (4096 envs) + eval
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "${FULL:-0}" = "1" ]; then
  python scripts/train_language_velocity.py --headless --num_envs "${NUM_ENVS:-4096}" \
    --max_iterations "${MAX_ITERS:-3000}"
  python scripts/eval_language_velocity.py \
    --checkpoint runs/language_velocity/model_latest.pt \
    --out_json results/eval_language.json --out_png results/behavior_separation.png
else
  # smoke: confirm the env builds + reaches PPO
  python scripts/train_language_velocity.py --headless --num_envs "${NUM_ENVS:-16}" \
    --max_iterations "${MAX_ITERS:-2}"
  echo "SMOKE OK if runs/language_velocity/model_latest.pt was written"
fi
