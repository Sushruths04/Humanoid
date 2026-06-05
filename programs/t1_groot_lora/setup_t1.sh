#!/usr/bin/env bash
# T1 — GR00T N1.7 LoRA setup script
# Run once on a fresh machine (40GB+ VRAM recommended for fine-tuning; 16GB for eval)
#
# Usage:
#   bash programs/t1_groot_lora/setup_t1.sh [--skip-dataset] [--skip-model]
#
# After setup, see programs/t1_groot_lora/README for train/eval commands.

set -euo pipefail

SKIP_DATASET=0
SKIP_MODEL=0
for arg in "$@"; do
  [[ "$arg" == "--skip-dataset" ]] && SKIP_DATASET=1
  [[ "$arg" == "--skip-model"   ]] && SKIP_MODEL=1
done

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GROOT_SRC="/tmp/Isaac-GR00T"
CONDA_PYTHON="/home/zeus/miniconda3/envs/groot_env/bin/python"
PIP="/home/zeus/miniconda3/envs/groot_env/bin/pip"
HF_CLI="/home/zeus/miniconda3/envs/groot_env/bin/huggingface-cli"

echo "=== T1 Setup ==="
echo "repo_root: $REPO_ROOT"
echo "groot_src: $GROOT_SRC"

# ── 1. Clone Isaac-GR00T if missing ──────────────────────────────────────────
if [[ ! -d "$GROOT_SRC" ]]; then
  echo "[setup] cloning Isaac-GR00T..."
  git clone --depth=1 https://github.com/NVIDIA/Isaac-GR00T.git "$GROOT_SRC"
fi

# ── 2. Install isaac-gr00t package ───────────────────────────────────────────
if ! "$CONDA_PYTHON" -c "import gr00t" 2>/dev/null; then
  echo "[setup] installing gr00t and dependencies..."
  # Install flash-attn first (needs pre-built wheel URL for cu121/cp310)
  "$PIP" install torch==2.7.1 torchvision==0.22.1 \
      --index-url https://download.pytorch.org/whl/cu128

  # Install flash-attn from wheel (avoids long compile)
  "$PIP" install flash-attn==2.7.4.post1 \
      --no-build-isolation \
      || echo "[setup] flash-attn install failed, continuing without it (slower attention)"

  # Install the rest of gr00t deps (skip tensorrt - not needed for training/eval)
  "$PIP" install \
      albumentations==1.4.18 \
      huggingface-hub[cli] \
      "opencv-python-headless>=4.5,<4.13" \
      av==16.1.0 \
      "diffusers==0.35.1" \
      dm-tree lmdb==1.7.5 msgpack==1.1.0 msgpack-numpy==0.4.8 \
      pandas==2.2.3 \
      "peft==0.17.1" \
      termcolor==3.2.0 \
      "transformers==4.57.3" \
      tyro==0.9.17 click==8.1.8 \
      "datasets==3.6.0" \
      einops==0.8.1 gitpython==3.1.46 jsonlines==4.0.0 \
      "gymnasium==1.2.2" \
      matplotlib==3.10.1 "numpy==1.26.4" \
      "omegaconf==2.3.0" scipy==1.15.3 \
      pyzmq==27.0.1 wandb==0.23.0 \
      deepspeed==0.17.6 \
      onnx onnxscript

  # Install gr00t itself from source
  "$PIP" install -e "$GROOT_SRC" --no-deps
  echo "[setup] gr00t installed"
fi

# ── 3. Download LIBERO spatial dataset (LeRobot format) ──────────────────────
DATASET_DIR="$REPO_ROOT/programs/t1_groot_lora/datasets/libero_spatial_no_noops"
if [[ $SKIP_DATASET -eq 0 && ! -d "$DATASET_DIR" ]]; then
  echo "[setup] downloading libero_spatial LeRobot dataset..."
  mkdir -p "$DATASET_DIR"
  "$HF_CLI" download \
      --repo-type dataset \
      IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
      --local-dir "$DATASET_DIR"
  # Patch in GR00T modality config
  cp "$GROOT_SRC/examples/LIBERO/modality.json" "$DATASET_DIR/meta/"
  echo "[setup] dataset downloaded to $DATASET_DIR"
fi

# ── 4. Download GR00T-N1.7-LIBERO (libero_spatial checkpoint) ────────────────
CKPT_DIR="$REPO_ROOT/programs/checkpoints/groot_n17/libero_spatial"
if [[ $SKIP_MODEL -eq 0 && ! -d "$CKPT_DIR" ]]; then
  echo "[setup] downloading nvidia/GR00T-N1.7-LIBERO (libero_spatial)..."
  mkdir -p "$CKPT_DIR"
  "$HF_CLI" download nvidia/GR00T-N1.7-LIBERO \
      --include "libero_spatial/config.json" \
                "libero_spatial/embodiment_id.json" \
                "libero_spatial/model-*.safetensors" \
                "libero_spatial/model.safetensors.index.json" \
                "libero_spatial/processor_config.json" \
                "libero_spatial/statistics.json" \
      --local-dir "$CKPT_DIR"
  echo "[setup] checkpoint downloaded to $CKPT_DIR"
fi

# ── 5. Setup LIBERO eval environment ─────────────────────────────────────────
LIBERO_EVAL_DIR="$GROOT_SRC/gr00t/eval/sim/LIBERO"
if [[ ! -f "$LIBERO_EVAL_DIR/libero_uv/.venv/pyvenv.cfg" ]]; then
  echo "[setup] setting up LIBERO eval environment..."
  cd "$GROOT_SRC"
  bash "$LIBERO_EVAL_DIR/setup_libero.sh" || echo "[setup] LIBERO eval setup failed (may need sudo apt install)"
fi

echo ""
echo "=== T1 Setup Complete ==="
echo ""
echo "To evaluate GR00T-N1.7 on LIBERO spatial (needs 16GB+ VRAM):"
echo "  See programs/t1_groot_lora/run_eval.sh"
echo ""
echo "To fine-tune from scratch (needs 40GB+ VRAM):"
echo "  See programs/t1_groot_lora/run_finetune.sh"
