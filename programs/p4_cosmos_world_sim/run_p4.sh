#!/usr/bin/env bash
# P4 GPU execution orchestrator — runs CP4.1 → CP4.5 sequentially.
# Run AFTER setup_machine.sh. Uses A100-80G.
#
# Usage:
#   export HF_TOKEN=hf_...
#   cd ~/Humanoid
#   bash programs/p4_cosmos_world_sim/run_p4.sh
#
# Skip to a specific checkpoint:
#   START_CP=3 bash programs/p4_cosmos_world_sim/run_p4.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="/home/zeus/miniconda3/envs/p4_env/bin/python"
COSMOS_ROOT="/tmp/cosmos-predict2"
DATA="$REPO_DIR/datasets/g1_nav"
LORA_OUT="$REPO_DIR/checkpoints/p4_cosmos_lora"
RESULTS="$REPO_DIR/docs/results"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN env var required — export HF_TOKEN=hf_...}"
START_CP="${START_CP:-1}"

echo "=================================================="
echo " P4 GPU Execution"
echo " Date: $(date)"
echo " Repo: $REPO_DIR"
echo " Start at CP: $START_CP"
echo "=================================================="

mkdir -p "$RESULTS"
cd "$REPO_DIR"

# ── CP4.1 ────────────────────────────────────────────────────────────────────
if [ "$START_CP" -le 1 ]; then
    echo ""
    echo "=== CP4.1: Stock Cosmos action-conditioned inference baseline ==="
    cd "$COSMOS_ROOT"
    $PY "$REPO_DIR/programs/p4_cosmos_world_sim/cp41_inference.py"         --video "$REPO_DIR/videos/p3_vision_nav/p3_vision_nav_model499.mp4"         --out "$RESULTS/cp41_inference.mp4"         --chunk-size 12
    cd "$REPO_DIR"
    echo "CP4.1 DONE ✓"
fi

# ── CP4.2 ────────────────────────────────────────────────────────────────────
if [ "$START_CP" -le 2 ]; then
    echo ""
    echo "=== CP4.2: Collect G1 nav rollouts (Isaac Lab Docker) ==="
    mkdir -p "$DATA"
    docker exec         -e PYTHONPATH="/workspace:/workspace/my-humanoid-project:/workspace/isaaclab/source"         isaac-lab-base python         /workspace/programs/p4_cosmos_world_sim/collect_rollouts.py         --checkpoint /workspace/checkpoints/p3_vision_nav/run_300_l4/model_499.pt         --task Humanoid-G1-VisionNav-v0         --num-envs 64         --num-steps 500         --out /workspace/datasets/g1_nav

    $PY "$REPO_DIR/programs/p4_cosmos_world_sim/cp42_verify_data.py"         --data "$DATA"
    echo "CP4.2 DONE ✓"
fi

# ── CP4.3 smoke gate ─────────────────────────────────────────────────────────
if [ "$START_CP" -le 3 ]; then
    echo ""
    echo "=== CP4.3: SMOKE TEST (2 training steps) ==="
    cd "$COSMOS_ROOT"
    $PY "$REPO_DIR/programs/p4_cosmos_world_sim/cp43_train.py"         --data "$DATA"         --lora-rank 16         --smoke
    cd "$REPO_DIR"

    echo ""
    echo ">>> SMOKE GATE — review output above."
    echo "    If no OOM and loss was printed, press Enter to start full training."
    echo "    Ctrl+C to abort (re-run with START_CP=3 after fixing)."
    read -r -p "    Smoke passed? [Enter to continue]: "

    echo ""
    echo "=== CP4.3: Full LoRA post-training (5000 steps) ==="
    mkdir -p "$LORA_OUT"
    cd "$COSMOS_ROOT"
    nohup $PY "$REPO_DIR/programs/p4_cosmos_world_sim/cp43_train.py"         --data "$DATA"         --out "$LORA_OUT"         --lora-rank 16         --max-steps 5000         --save-every 500         > "$REPO_DIR/p4_train.log" 2>&1 &
    TRAIN_PID=$!
    echo "Training PID: $TRAIN_PID  (log: $REPO_DIR/p4_train.log)"
    wait $TRAIN_PID
    cd "$REPO_DIR"
    echo "CP4.3 DONE ✓"
fi

# ── CP4.4 ────────────────────────────────────────────────────────────────────
if [ "$START_CP" -le 4 ]; then
    echo ""
    echo "=== CP4.4: K-step rollout evaluation ==="
    cd "$COSMOS_ROOT"
    $PY "$REPO_DIR/programs/p4_cosmos_world_sim/cp44_rollout.py"         --lora-ckpt "$LORA_OUT"         --data "$DATA"         --k 8         --out "$RESULTS/cp44_rollout.mp4"
    cd "$REPO_DIR"
    echo "CP4.4 DONE ✓"
fi

# ── CP4.5 ────────────────────────────────────────────────────────────────────
if [ "$START_CP" -le 5 ]; then
    echo ""
    echo "=== CP4.5: CEM planning demo ==="
    cd "$COSMOS_ROOT"
    $PY "$REPO_DIR/programs/p4_cosmos_world_sim/cp45_plan.py"         --cosmos-ckpt "$LORA_OUT"         --data "$DATA"         --plan-steps 8         --cem-samples 64         --out "$RESULTS/cp45_planning.mp4"
    cd "$REPO_DIR"
    echo "CP4.5 DONE ✓"
fi

# ── Upload to HuggingFace ─────────────────────────────────────────────────────
echo ""
echo "=== Uploading results to HuggingFace ==="
$PY - << 'PYEOF'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = "mitvho09/humanoid-g1-nav"

# Upload LoRA checkpoint
lora_dir = os.environ.get("LORA_OUT", "$LORA_OUT")
if os.path.isdir(lora_dir):
    api.upload_folder(
        folder_path=lora_dir,
        repo_id=repo,
        path_in_repo="checkpoints/p4_cosmos_lora/",
        repo_type="dataset"
    )
    print("LoRA checkpoint uploaded")

# Upload result videos
results_dir = os.environ.get("RESULTS", "$RESULTS")
for fname in ["cp41_inference.mp4", "cp44_rollout.mp4", "cp45_planning.mp4"]:
    fpath = os.path.join(results_dir, fname)
    if os.path.exists(fpath):
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=f"videos/p4/{fname}",
            repo_id=repo,
            repo_type="dataset"
        )
        print(f"Uploaded {fname}")
    else:
        print(f"Skipped {fname} (not found)")
PYEOF

echo ""
echo "=================================================="
echo " P4 COMPLETE"
echo " Commit results:"
echo "   cd $REPO_DIR"
echo "   git add docs/results/ checkpoints/p4_cosmos_lora/"
echo "   git commit -m 'results: P4 Cosmos LoRA complete'"
echo "   git push origin feat/planned-scripts"
echo "=================================================="
