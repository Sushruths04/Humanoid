#!/usr/bin/env bash
# P4 GPU execution orchestrator — runs CP4.1 → CP4.5 sequentially.
# Each checkpoint prints DONE before advancing.
# Run AFTER setup_machine.sh completes.
#
# Usage:
#   cd ~/Humanoid
#   bash programs/p4_cosmos_world_sim/run_p4.sh
#
# Skip to a specific checkpoint:
#   START_CP=4 bash programs/p4_cosmos_world_sim/run_p4.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="/home/zeus/miniconda3/envs/p4_env/bin/python"
COSMOS_CKPT="$REPO_DIR/checkpoints/cosmos_base"
P3_CKPT="$REPO_DIR/checkpoints/p3_vision_nav/run_300_l4/model_499.pt"
DATA="$REPO_DIR/datasets/g1_nav_cosmos.h5"
LORA_OUT="$REPO_DIR/checkpoints/p4_cosmos_lora"
RESULTS="$REPO_DIR/docs/results"
START_CP="${START_CP:-1}"

echo "=================================================="
echo " P4 GPU Execution"
echo " Date: $(date)"
echo " Start at CP: $START_CP"
echo "=================================================="

cd "$REPO_DIR"

# ── CP4.1 ─────────────────────────────────────────────────────────────────────
if [ "$START_CP" -le 1 ]; then
    echo ""
    echo "=== CP4.1: Stock Cosmos inference baseline ==="
    # Extract initial frame from P3 video
    $PY -c "
import imageio, numpy as np
v = imageio.get_reader('programs/videos/p3_vision_nav/p3_vision_nav_model499.mp4')
frame = v.get_data(0)
np.save('/tmp/initial_frame.npy', frame)
print('Initial frame:', frame.shape, frame.dtype)
v.close()
"
    $PY -m programs.p4_cosmos_world_sim.cp41_inference \
        --model-dir "$COSMOS_CKPT" \
        --frame /tmp/initial_frame.npy \
        --out "$RESULTS/cp41_inference.mp4" \
        --steps 16
    echo "CP4.1 DONE ✓"
fi

# ── CP4.2 ─────────────────────────────────────────────────────────────────────
if [ "$START_CP" -le 2 ]; then
    echo ""
    echo "=== CP4.2: Collect rollouts (Isaac Lab Docker) ==="
    mkdir -p "$REPO_DIR/datasets"
    docker exec \
        -e PYTHONPATH="/workspace:/workspace/my-humanoid-project:/workspace/isaaclab/source" \
        -e P3_CAM_H="64" -e P3_CAM_W="64" \
        isaac-lab-base python \
        /workspace/programs/p4_cosmos_world_sim/collect_rollouts.py \
        --checkpoint /workspace/checkpoints/p3_vision_nav/run_300_l4/model_499.pt \
        --task Humanoid-G1-VisionNav-v0 \
        --num-envs 64 \
        --num-steps 500 \
        --out /workspace/datasets/g1_nav_cosmos.h5

    $PY -m programs.p4_cosmos_world_sim.cp42_verify_data --data "$DATA"
    echo "CP4.2 DONE ✓"
fi

# ── CP4.3 smoke ───────────────────────────────────────────────────────────────
if [ "$START_CP" -le 3 ]; then
    echo ""
    echo "=== CP4.3: SMOKE TEST (2 steps) ==="
    $PY -m programs.p4_cosmos_world_sim.cp43_train \
        --data "$DATA" \
        --model-dir "$COSMOS_CKPT" \
        --lora-rank 16 \
        --smoke \
        --max-steps 2 \
        --out /tmp/cosmos_smoke/
    echo ""
    echo ">>> SMOKE GATE: Review output above."
    echo "    If PASSED (no OOM, loss printed), press Enter to start full training."
    echo "    If FAILED, check error, reduce --lora-rank, then re-run with START_CP=3."
    read -r -p "    Smoke passed? [Enter to continue / Ctrl+C to abort]: "

    echo ""
    echo "=== CP4.3: Full LoRA post-train (5000 steps) ==="
    mkdir -p "$LORA_OUT"
    nohup $PY -m programs.p4_cosmos_world_sim.cp43_train \
        --data "$DATA" \
        --model-dir "$COSMOS_CKPT" \
        --lora-rank 16 \
        --max-steps 5000 \
        --save-every 500 \
        --out "$LORA_OUT" \
        > "$REPO_DIR/p4_train.log" 2>&1 &
    TRAIN_PID=$!
    echo "Training PID: $TRAIN_PID  (log: p4_train.log)"
    echo "Waiting for training to complete ..."
    wait $TRAIN_PID
    echo "CP4.3 DONE ✓"

    # Diff-actions eval
    $PY -m programs.p4_cosmos_world_sim.cp43_train \
        --eval-only \
        --checkpoint "$LORA_OUT" \
        --out "$RESULTS/cp43_action_diff.mp4"
fi

# ── CP4.4 ─────────────────────────────────────────────────────────────────────
if [ "$START_CP" -le 4 ]; then
    echo ""
    echo "=== CP4.4: K-step rollout evaluation ==="
    $PY -m programs.p4_cosmos_world_sim.cp44_rollout \
        --checkpoint "$LORA_OUT" \
        --data "$DATA" \
        --k-steps 8 \
        --out "$RESULTS/cp44_rollout.mp4"
    echo "CP4.4 DONE ✓"
fi

# ── CP4.5 ─────────────────────────────────────────────────────────────────────
if [ "$START_CP" -le 5 ]; then
    echo ""
    echo "=== CP4.5: CEM planning demo ==="
    docker exec \
        -e PYTHONPATH="/workspace:/workspace/my-humanoid-project:/workspace/isaaclab/source" \
        isaac-lab-base python \
        /workspace/programs/p4_cosmos_world_sim/cp45_plan.py \
        --cosmos-checkpoint /workspace/checkpoints/p4_cosmos_lora/ \
        --task Humanoid-G1-VisionNav-v0 \
        --num-envs 4 \
        --plan-steps 8 \
        --cem-samples 64 \
        --out /workspace/docs/results/cp45_planning.mp4
    echo "CP4.5 DONE ✓"
fi

# ── Upload ────────────────────────────────────────────────────────────────────
echo ""
echo "=== Uploading to HuggingFace ==="
$PY -c "
from huggingface_hub import HfApi
api = HfApi(token=\"${HF_TOKEN:?HF_TOKEN env var required}\")
api.upload_folder(
    folder_path='$LORA_OUT/',
    repo_id='mitvho09/humanoid-g1-nav',
    path_in_repo='checkpoints/p4_cosmos_lora/',
    repo_type='dataset'
)
print('Checkpoint uploaded')
for fname in ['cp41_inference.mp4','cp43_action_diff.mp4','cp44_rollout.mp4','cp45_planning.mp4']:
    try:
        api.upload_file(
            path_or_fileobj=f'$RESULTS/{fname}',
            path_in_repo=f'videos/p4/{fname}',
            repo_id='mitvho09/humanoid-g1-nav',
            repo_type='dataset'
        )
        print(f'Uploaded {fname}')
    except FileNotFoundError:
        print(f'Skipped {fname} (not found)')
"

echo ""
echo "=================================================="
echo " P4 COMPLETE"
echo " Commit + push results:"
echo "   cd $REPO_DIR"
echo "   git add docs/results/"
echo "   git commit -m 'results: P4 complete'"
echo "   git push origin feat/planned-scripts"
echo "=================================================="
