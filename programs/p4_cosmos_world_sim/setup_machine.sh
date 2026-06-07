#!/usr/bin/env bash
# P4 Machine Setup — runs on CPU (no GPU needed).
# Installs all deps, downloads models, brings up Isaac Lab Docker.
# Run once on a fresh A100 machine before starting GPU execution.
#
# Usage:
#   bash ~/Humanoid/programs/p4_cosmos_world_sim/setup_machine.sh
#
# Expected: prints "SETUP COMPLETE" at the end. Takes ~30-40 min (download-dominated).

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Humanoid}"
CONDA="${CONDA:-/home/zeus/miniconda3/bin/conda}"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN env var required — export HF_TOKEN=hf_...}"
COSMOS_CLONE_DIR="/tmp/cosmos-predict2"
COSMOS_CKPT_DIR="$REPO_DIR/checkpoints/cosmos_base"
P3_CKPT_DIR="$REPO_DIR/checkpoints/p3_vision_nav/run_300_l4"
ISAAC_IMAGE="nvcr.io/nvidia/isaac-lab/isaac-lab:2.0.0"

echo "=================================================="
echo " P4 Machine Setup"
echo " Repo: $REPO_DIR"
echo " Date: $(date)"
echo "=================================================="

# ── Step 1: conda ────────────────────────────────────────────────────────────
echo ""
echo "[1/8] Checking conda ..."
if ! command -v conda &>/dev/null && ! "$CONDA" --version &>/dev/null; then
    echo "conda not found — installing Miniconda"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mini.sh
    bash /tmp/mini.sh -b -p /home/zeus/miniconda3
    export PATH=/home/zeus/miniconda3/bin:$PATH
fi
echo "conda: $($CONDA --version)"

# ── Step 2: p4_env ───────────────────────────────────────────────────────────
echo ""
echo "[2/8] Creating p4_env (Python 3.10) ..."
$CONDA create -n p4_env python=3.10 -y 2>&1 | tail -3
PY="$HOME/miniconda3/envs/p4_env/bin/python"
PIP="$HOME/miniconda3/envs/p4_env/bin/pip"
# Fallback to zeus path
if [ ! -f "$PY" ]; then PY="/home/zeus/miniconda3/envs/p4_env/bin/python"; fi
if [ ! -f "$PIP" ]; then PIP="/home/zeus/miniconda3/envs/p4_env/bin/pip"; fi

echo "Python: $($PY --version)"

# ── Step 3: PyTorch ──────────────────────────────────────────────────────────
echo ""
echo "[3/8] Installing PyTorch 2.7.1+cu128 ..."
$PIP install -q \
    torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

$PY -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.version.cuda)"

# ── Step 4: Cosmos Predict 2 ─────────────────────────────────────────────────
echo ""
echo "[4/8] Cloning cosmos-predict2 ..."
if [ -d "$COSMOS_CLONE_DIR" ]; then
    echo "  Already cloned at $COSMOS_CLONE_DIR"
else
    # Try primary URL first, then alternate
    git clone --depth=1 https://github.com/nvidia-cosmos/cosmos-predict2 "$COSMOS_CLONE_DIR" 2>/dev/null \
        || git clone --depth=1 https://github.com/NVIDIA-Cosmos/cosmos-predict2 "$COSMOS_CLONE_DIR"
fi
echo "  Installing cosmos-predict2 ..."
cd "$COSMOS_CLONE_DIR"
$PIP install -q -e '.' --no-build-isolation 2>&1 | tail -5
cd "$REPO_DIR"

# ── Step 5: Training deps ─────────────────────────────────────────────────────
echo ""
echo "[5/8] Installing LoRA + training deps ..."
$PIP install -q \
    peft bitsandbytes accelerate \
    huggingface_hub h5py \
    imageio imageio-ffmpeg \
    scikit-image \
    numpy==1.26.4 einops

echo "  Verifying bitsandbytes ..."
$PY -c "import bitsandbytes as bnb; print('bitsandbytes:', bnb.__version__)"

# ── Step 6: HuggingFace login + Cosmos model download ────────────────────────
echo ""
echo "[6/8] Downloading Cosmos model weights (large — ~4-10 GB) ..."
$PY -c "from huggingface_hub import HfApi; HfApi(token='$HF_TOKEN').whoami()" 2>/dev/null \
    && echo "  HF token valid" \
    || echo "  [WARN] HF token check failed — continuing anyway"

mkdir -p "$COSMOS_CKPT_DIR"

# Find the correct model ID from cosmos-predict2 README
COSMOS_MODEL_ID=$(grep -iE "nvidia/Cosmos-Predict" "$COSMOS_CLONE_DIR/README.md" \
    | grep -oE 'nvidia/Cosmos-Predict[^"` ]+' | head -1)
if [ -z "$COSMOS_MODEL_ID" ]; then
    COSMOS_MODEL_ID="nvidia/Cosmos-Predict-2-2B"
    echo "  Could not auto-detect model ID from README; defaulting to: $COSMOS_MODEL_ID"
    echo "  If download fails, check README.md and update MODEL_ID manually."
fi
echo "  Cosmos model: $COSMOS_MODEL_ID"

$PY -m huggingface_hub download \
    --token "$HF_TOKEN" \
    --local-dir "$COSMOS_CKPT_DIR" \
    --quiet \
    "$COSMOS_MODEL_ID" \
    2>&1 | tail -5

echo "  Cosmos weights: $(du -sh $COSMOS_CKPT_DIR | cut -f1)"

# ── Step 7: P3 checkpoint ────────────────────────────────────────────────────
echo ""
echo "[7/8] Downloading P3 checkpoint (model_499.pt) ..."
mkdir -p "$P3_CKPT_DIR"
$HOME/miniconda3/envs/p4_env/bin/huggingface-cli download \
    mitvho09/humanoid-g1-nav \
    --include "checkpoints/p3_vision_nav/run_300_l4/model_499.pt" \
    --token "$HF_TOKEN" \
    --local-dir "$REPO_DIR" \
    --quiet \
    2>&1 | tail -3

ls "$P3_CKPT_DIR/model_499.pt" && echo "  P3 checkpoint: OK" || echo "  [WARN] model_499.pt not found"

# ── Step 8: Isaac Lab Docker ─────────────────────────────────────────────────
echo ""
echo "[8/8] Pulling Isaac Lab Docker image (~17.6 GB) ..."
docker pull "$ISAAC_IMAGE" 2>&1 | tail -5

echo "  Starting isaac-lab-base container ..."
docker rm -f isaac-lab-base 2>/dev/null || true
docker run -d --name isaac-lab-base --gpus all \
    --network host \
    -v "$REPO_DIR":/workspace \
    -e PYTHONPATH="/workspace:/workspace/my-humanoid-project:/workspace/isaaclab/source" \
    "$ISAAC_IMAGE" sleep infinity

sleep 3
docker exec isaac-lab-base python -c "import isaaclab; print('IsaacLab OK')" \
    && echo "  Isaac Lab container: OK" \
    || echo "  [WARN] IsaacLab import check failed — verify PYTHONPATH inside container"

# ── Final verification ────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo " Verification summary"
echo "=================================================="
$PY -c "
import torch, h5py, peft, bitsandbytes, imageio, huggingface_hub
import sys; sys.path.insert(0, '$REPO_DIR')
print('torch:          ', torch.__version__)
print('CUDA available: ', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:            ', torch.cuda.get_device_name(0))
    print('VRAM (GB):      ', torch.cuda.get_device_properties(0).total_memory // 1024**3)
print('peft:           ', peft.__version__)
print('bitsandbytes:   ', bitsandbytes.__version__)
print('h5py:           ', h5py.__version__)
print('huggingface_hub:', huggingface_hub.__version__)
"

echo ""
echo "SETUP COMPLETE"
echo "Next: run GPU execution. Follow docs/superpowers/plans/2026-06-07-p4-machine-setup.md Phase C."
