#!/usr/bin/env bash
# T2 — World Model for Manipulation: collect rollouts → train WM → push results
#
# Run from repo root on Lightning Studio (groot_env activated):
#   MUJOCO_GL=egl bash programs/t2_manip_wm/run_t2.sh
#
# Requires: GR00T checkpoint at programs/checkpoints/groot_n17/libero_spatial/libero_spatial
#           LIBERO available (set MUJOCO_GL=egl)
#           /tmp/Isaac-GR00T cloned + gr00t installed in groot_env

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON=/home/zeus/miniconda3/envs/groot_env/bin/python
CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/programs/checkpoints/groot_n17/libero_spatial/libero_spatial}"
NUM_EPISODES="${NUM_EPISODES:-200}"
DATA_OUT="$REPO_ROOT/programs/data/manip_rollouts_groot.pt"
WM_OUT="$REPO_ROOT/programs/checkpoints/world_model/wm_manip.pt"
RESULT_DOC="$REPO_ROOT/docs/results/t2_manip_wm.md"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1

echo "=== T2 Step 1/2: Collect $NUM_EPISODES GR00T rollouts ==="
$PYTHON -u -m programs.t2_manip_wm.collect_manip_rollouts \
    --checkpoint "$CHECKPOINT" \
    --num-episodes "$NUM_EPISODES" \
    --max-steps 300 \
    --out "$DATA_OUT"

echo "=== T2 Step 2/2: Train world model ==="
$PYTHON -u -m programs.t2_manip_wm.train_wm_manip \
    --data "$DATA_OUT" \
    --steps 3000 \
    --out "$WM_OUT" \
    --result-doc "$RESULT_DOC"

echo "=== T2 complete — pushing results ==="
cd "$REPO_ROOT"
git add docs/results/t2_manip_wm.md \
        programs/t2_manip_wm/ \
        programs/checkpoints/world_model/wm_manip.pt 2>/dev/null || true
git commit -m "results: T2 world model for manipulation complete" || echo "[git] nothing new to commit"
git push origin feat/planned-scripts

echo "=== T2 done ==="
