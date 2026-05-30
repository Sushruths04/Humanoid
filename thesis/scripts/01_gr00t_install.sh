#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"

if [ "${RUN_CPU_INSTALLS:-0}" != "1" ]; then
  {
    echo "## Deferred to explicit CPU install"
    echo
    echo "This step is CPU-compatible but downloads and installs a large upstream repo."
    echo "To run it, edit \`thesis/config.env\`:"
    echo
    echo '```bash'
    echo "RUN_CPU_INSTALLS=1"
    echo '```'
    echo
    echo "Then execute:"
    echo
    echo '```bash'
    echo "bash thesis/run_thesis.sh --all --from 01_gr00t_install"
    echo '```'
  } | md_log "01-gr00t-install" "STEP 01 GR00T install deferred"
  exit 78
fi

cd "$WORKSPACE_DIR"
if [ ! -d Isaac-GR00T ]; then
  git clone https://github.com/NVIDIA/Isaac-GR00T.git
fi

cd Isaac-GR00T
"$PYTHON_BIN" -m pip install -U pip
"$PYTHON_BIN" -m pip install torch==2.7.1 torchvision==0.22.1 psutil ninja packaging wheel setuptools
mkdir -p /home/zeus/content/pip-tmp /home/zeus/content/pip-cache
TMPDIR=/home/zeus/content/pip-tmp PIP_CACHE_DIR=/home/zeus/content/pip-cache CUDA_HOME=/usr/local/cuda \
  "$PYTHON_BIN" -m pip install flash-attn==2.7.4.post1 --no-build-isolation
TMPDIR=/home/zeus/content/pip-tmp PIP_CACHE_DIR=/home/zeus/content/pip-cache CUDA_HOME=/usr/local/cuda \
  "$PYTHON_BIN" -m pip install -e .

OUT="$THESIS_DIR/logs/01_gr00t_import.txt"
run_cmd_capture "$OUT" "$PYTHON_BIN" -c "import gr00t; print('ok')"

{
  echo "## Commands"
  echo
  echo '```bash'
  echo "git clone https://github.com/NVIDIA/Isaac-GR00T.git"
  echo ""$PYTHON_BIN" -m pip install -e ."
  echo "python3 -c \"import gr00t; print('ok')\""
  echo '```'
  echo
  echo "## Verification"
  echo
  echo '```text'
  cat "$OUT"
  echo '```'
} | md_log "01-gr00t-install" "STEP 01 GR00T install"

