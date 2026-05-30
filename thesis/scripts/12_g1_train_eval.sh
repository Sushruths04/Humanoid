#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_gpu "12_g1_train_eval" || exit 78

mkdir -p "$THESIS_DIR/checkpoints/g1_language"
{
  echo "## Language-conditioned training command"
  echo
  echo '```bash'
  echo "cd $WORKSPACE_DIR"
  echo "export PYTHONPATH=$WORKSPACE_DIR/my-humanoid-project:\$PYTHONPATH"
  echo "python IsaacLab/scripts/environments/list_envs.py | grep -i Humanoid-G1-Language"
  echo "IsaacLab/isaaclab.sh -p IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py --task ${G1_LANGUAGE_TASK_ID:-Humanoid-G1-Language-PickPlace-v0} --headless --num_envs ${G1_NUM_ENVS:-512} --max_iterations ${G1_MAX_ITERS:-300}"
  echo '```'
} | md_log "12-g1-train-eval" "STEP 12 G1 language train/eval gate"
exit 1

