#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_gpu "02_gr00t_demo" || exit 78

GR00T_DIR="$WORKSPACE_DIR/Isaac-GR00T"
RESULT_DIR="$WORKSPACE_DIR/thesis/results/gr00t_demo"
LOG_FILE="$WORKSPACE_DIR/thesis/logs/02_gr00t_demo.log"
SUMMARY_FILE="$RESULT_DIR/summary.txt"
PLOT_FILE="$RESULT_DIR/traj_1.jpeg"
HF_HOME_DIR="$WORKSPACE_DIR/thesis/cache/huggingface"

if [ ! -d "$GR00T_DIR" ]; then
  echo "Isaac-GR00T is missing. Run STEP 01 first." | md_log "02-gr00t-demo" "STEP 02 blocked"
  exit 1
fi

mkdir -p "$RESULT_DIR" "$(dirname "$LOG_FILE")" "$HF_HOME_DIR"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export HF_HOME="$HF_HOME_DIR"
export TRANSFORMERS_CACHE="$HF_HOME_DIR/transformers"
export PYTHONUNBUFFERED=1

{
  echo "# STEP 02 GR00T GPU demo"
  echo
  echo "_$(date -u +%FT%TZ)_"
  echo
  echo "## GPU"
  echo
  echo '```text'
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
  echo '```'
  echo
  echo "## Command"
  echo
  echo '```bash'
  echo "cd $GR00T_DIR"
  echo "$PYTHON_BIN scripts/deployment/standalone_inference_script.py --model-path nvidia/GR00T-N1.7-3B --dataset-path demo_data/droid_sample --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT --traj-ids 1 --steps 16 --inference-mode pytorch --action-horizon 8 --save-plot-path $PLOT_FILE"
  echo '```'
} | md_log "02-gr00t-demo" "STEP 02 GR00T GPU demo"

cd "$GR00T_DIR"

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import torchcodec
from torchcodec.decoders import VideoDecoder
PY
then
  {
    echo "## Installing FFmpeg runtime"
    echo
    echo "torchcodec is installed but cannot load its FFmpeg runtime libraries. Installing the OS FFmpeg packages required by the demo video decoder."
  } | md_log "02-gr00t-demo" "STEP 02 install FFmpeg runtime"
  if sudo -n true >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends ffmpeg libavdevice60
  else
    echo "torchcodec cannot load and passwordless sudo is unavailable for installing ffmpeg/libavdevice60." | md_log "02-gr00t-demo" "STEP 02 blocked"
    exit 1
  fi
fi

"$PYTHON_BIN" - <<'PY' | tee "$WORKSPACE_DIR/thesis/logs/02_gr00t_preflight.log"
import torch
import gr00t
import torchcodec
from gr00t.policy.gr00t_policy import Gr00tPolicy
from torchcodec.decoders import VideoDecoder
from huggingface_hub import HfFolder
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("gr00t_import", "ok")
print("policy_import", Gr00tPolicy.__name__)
print("torchcodec_import", getattr(torchcodec, "__version__", "unknown"))
print("video_decoder", VideoDecoder.__name__)
print("hf_token_present", bool(HfFolder.get_token()))
PY

set +e
"$PYTHON_BIN" scripts/deployment/standalone_inference_script.py \
  --model-path nvidia/GR00T-N1.7-3B \
  --dataset-path demo_data/droid_sample \
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
  --traj-ids 1 \
  --steps 16 \
  --inference-mode pytorch \
  --action-horizon 8 \
  --save-plot-path "$PLOT_FILE" \
  2>&1 | tee "$LOG_FILE"
status=${PIPESTATUS[0]}
set -e

blocked=0
if grep -qE "Cannot access gated repo|GatedRepoError|401 Client Error|Please log in" "$LOG_FILE"; then
  blocked=1
fi

{
  echo "# GR00T demo summary"
  echo
  echo "Generated: $(date -u +%FT%TZ)"
  echo "Exit status: $status"
  echo "Blocked by gated Hugging Face access: $blocked"
  echo "Log: $LOG_FILE"
  echo "Plot: $PLOT_FILE"
  echo
  echo "## Preflight"
  cat "$WORKSPACE_DIR/thesis/logs/02_gr00t_preflight.log"
  echo
  echo "## Key lines"
  grep -E "Model loading time|Dataset length|Using [0-9]+ steps|pred_action_joints|Average MSE|Average MAE|Avg inference time|Done|Cannot access gated repo|Please log in|hf_token_present" "$LOG_FILE" "$WORKSPACE_DIR/thesis/logs/02_gr00t_preflight.log" || true
} > "$SUMMARY_FILE"

{
  echo
  echo "## Result"
  echo
  echo '```text'
  cat "$SUMMARY_FILE"
  echo '```'
} | md_log "02-gr00t-demo" "STEP 02 result"

if [ "$blocked" -eq 1 ]; then
  {
    echo "## Waiting for Hugging Face authentication"
    echo
    echo "The GPU, Torch CUDA, GR00T import, and policy import passed. Model loading is blocked because the checkpoint depends on gated repo nvidia/Cosmos-Reason2-2B."
    echo
    echo '```bash'
    echo "hf auth login"
    echo "# then rerun: bash thesis/run_thesis.sh --all --from 02_gr00t_demo --no-autosave"
    echo '```'
  } | md_log "02-gr00t-demo" "STEP 02 waiting on HF auth"
  exit 78
fi

if [ "$status" -ne 0 ]; then
  exit "$status"
fi

test -s "$PLOT_FILE"
test -s "$SUMMARY_FILE"
exit 0
