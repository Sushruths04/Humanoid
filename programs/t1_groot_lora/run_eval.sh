#!/usr/bin/env bash
# T1 — Evaluate GR00T N1.7 on LIBERO spatial
#
# Uses the GR00T client-server eval protocol:
#   Server: runs the GR00T policy model (needs 16GB+ VRAM)
#   Client: runs LIBERO simulation environment
#
# Hardware: 1× GPU ≥16 GB VRAM
#
# Usage:
#   # Use pre-trained NVIDIA checkpoint (default):
#   bash programs/t1_groot_lora/run_eval.sh
#
#   # Use your fine-tuned checkpoint:
#   CHECKPOINT=programs/checkpoints/groot_n17/libero_spatial_ft/checkpoint-5000 \
#   bash programs/t1_groot_lora/run_eval.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GROOT_SRC="/tmp/Isaac-GR00T"
PYTHON="/home/zeus/miniconda3/envs/groot_env/bin/python"
LIBERO_PYTHON="$GROOT_SRC/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python"

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_CKPT="$REPO_ROOT/programs/checkpoints/groot_n17/libero_spatial/libero_spatial"
CHECKPOINT="${CHECKPOINT:-$DEFAULT_CKPT}"
N_EPISODES="${N_EPISODES:-50}"
N_ENVS="${N_ENVS:-5}"
MAX_STEPS="${MAX_STEPS:-720}"
PORT="${PORT:-5555}"
RESULTS_OUT="${RESULTS_OUT:-$REPO_ROOT/docs/results/t1_groot.md}"

# Tasks to evaluate (all 10 libero_spatial tasks)
TASKS=(
  "libero_sim/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
  "libero_sim/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate"
  "libero_sim/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"
  "libero_sim/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate"
  "libero_sim/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate"
  "libero_sim/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"
  "libero_sim/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate"
  "libero_sim/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"
  "libero_sim/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate"
  "libero_sim/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"
)

echo "=== T1 GR00T Eval ==="
echo "  checkpoint: $CHECKPOINT"
echo "  n_episodes: $N_EPISODES  n_envs: $N_ENVS"

mkdir -p "$(dirname "$RESULTS_OUT")"
TMPLOG="/tmp/groot_eval_results.jsonl"
> "$TMPLOG"

total_success=0
total_episodes=0

for TASK in "${TASKS[@]}"; do
  TASK_NAME="${TASK##*/}"
  echo ""
  echo "--- Task: $TASK_NAME ---"

  # Start GR00T policy server in background
  MUJOCO_GL=egl "$PYTHON" \
    "$GROOT_SRC/gr00t/eval/run_gr00t_server.py" \
    --model-path "$CHECKPOINT" \
    --embodiment-tag LIBERO_PANDA \
    --use-sim-policy-wrapper \
    --port "$PORT" \
    &
  SERVER_PID=$!
  echo "  server pid: $SERVER_PID"
  sleep 8  # Wait for model to load

  # Run eval client
  RESULT=$("$LIBERO_PYTHON" \
    "$GROOT_SRC/gr00t/eval/rollout_policy.py" \
    --n-episodes "$N_EPISODES" \
    --policy-client-host 127.0.0.1 \
    --policy-client-port "$PORT" \
    --max-episode-steps "$MAX_STEPS" \
    --env-name "$TASK" \
    --n-action-steps 8 \
    --n-envs "$N_ENVS" \
    2>&1 | tail -5)

  echo "  result: $RESULT"

  # Extract success from result output
  SUCCESS=$(echo "$RESULT" | grep -oP "success_rate: \K[0-9.]+|Success: \K[0-9]+" | tail -1 || echo "0")
  echo "{\"task\": \"$TASK_NAME\", \"success\": $SUCCESS}" >> "$TMPLOG"
  total_success=$(echo "$total_success + $SUCCESS * $N_EPISODES" | bc -l)
  total_episodes=$((total_episodes + N_EPISODES))

  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
done

# Write markdown report
MEAN_SR=$(echo "scale=4; $total_success / $total_episodes" | bc -l)
echo "=== Overall success: $MEAN_SR ==="

python3 - <<PYEOF
import json, math, pathlib

with open("$TMPLOG") as f:
    rows = [json.loads(l) for l in f]

lines = [
    "# T1 GR00T N1.7 Eval: LIBERO Spatial",
    "",
    f"Checkpoint: \`$CHECKPOINT\`",
    f"Episodes per task: $N_EPISODES  Tasks: {len(rows)}",
    "",
    "## Per-Task Results",
    "",
    "| Task | Success Rate |",
    "|---|---|",
]
for r in rows:
    task = r["task"].replace("_", " ")
    lines.append(f"| {task} | {r['success']:.3f} |")

total_s = sum(r['success'] for r in rows)
mean_sr = total_s / len(rows) if rows else 0.0
lines += [
    "",
    "## Summary",
    "",
    f"| Metric | Value |",
    f"|---|---|",
    f"| mean_success_rate | **{mean_sr:.3f}** |",
    f"| T0 BC baseline | 0.500 |",
    f"| GR00T improvement | +{(mean_sr - 0.500):.3f} |",
    "",
]

pathlib.Path("$RESULTS_OUT").write_text("\n".join(lines))
print(f"Report written to $RESULTS_OUT")
PYEOF
