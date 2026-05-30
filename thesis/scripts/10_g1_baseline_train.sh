#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_gpu "10_g1_baseline" || exit 78

mkdir -p "$THESIS_DIR/checkpoints/g1_baseline"
{
  echo "## Baseline command"
  echo
  echo '```bash'
  echo "cd $WORKSPACE_DIR/IsaacLab"
  echo "python scripts/environments/list_envs.py | grep -i 'G1\\|loco'"
  echo "./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task ${G1_TASK_ID:-Isaac-PickPlace-Locomanipulation-G1-Abs-v0} --headless --num_envs ${G1_NUM_ENVS:-512} --max_iterations ${G1_MAX_ITERS:-300}"
  echo '```'
  echo
  echo "Run this only after Isaac Sim/Isaac Lab GPU smoke test succeeds."
} | md_log "10-g1-baseline" "STEP 10 G1 baseline command gate"
exit 1

