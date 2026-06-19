#!/usr/bin/env bash
# Evaluate one checkpoint at one pull speed -> results/eval_<v>kmh.json (PLAN §10.2).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CKPT:?set CKPT=path/to/model.pt}"
V="${V_PULL_KMH:-30}"
python eval.py --checkpoint "${CKPT}" --v_pull_kmh "${V}" \
  --episodes "${EPISODES:-200}" --out "results/eval_${V}kmh.json"
