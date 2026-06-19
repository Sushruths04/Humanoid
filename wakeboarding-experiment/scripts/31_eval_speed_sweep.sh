#!/usr/bin/env bash
# Table A — success vs pull speed (PLAN §10.3). Sweeps 20/25/30/35 km/h.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CKPT:?set CKPT=path/to/model.pt}"
for V in 20 25 30 35; do
  python eval.py --checkpoint "${CKPT}" --v_pull_kmh "${V}" \
    --episodes "${EPISODES:-200}" --out "results/sweep_${V}kmh.json"
done
echo "Sweep done -> results/sweep_*.json (aggregate with 99_collect_results.sh)"
