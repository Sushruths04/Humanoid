#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Sushruths04/Humanoid.git}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/home/zeus/content}"
PROJECT_DIR="${PROJECT_DIR:-$WORKSPACE_DIR/Humanoid}"

clone_or_update_repo() {
  mkdir -p "$WORKSPACE_DIR"
  local git_pull_cmd=(git -C "$PROJECT_DIR" pull --ff-only)
  if [ -d "$PROJECT_DIR/.git" ]; then
    echo "[bootstrap] Updating $PROJECT_DIR"
    if [ -n "${GITHUB_TOKEN:-}" ]; then
      askpass="$(mktemp)"
      chmod 700 "$askpass"
      cat > "$askpass" <<'EOF'
#!/usr/bin/env bash
case "$1" in
  *Username*) printf '%s\n' "x-access-token" ;;
  *Password*) printf '%s\n' "$GITHUB_TOKEN" ;;
  *) printf '\n' ;;
esac
EOF
      GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0 "${git_pull_cmd[@]}"
      rm -f "$askpass"
    else
      "${git_pull_cmd[@]}"
    fi
    return
  fi

  if [ -d "$PROJECT_DIR" ] && [ "$(find "$PROJECT_DIR" -mindepth 1 -maxdepth 1 | head -1)" ]; then
    echo "[bootstrap] $PROJECT_DIR exists but is not a git checkout."
    echo "[bootstrap] Move it aside or set PROJECT_DIR to a clean path."
    exit 2
  fi

  echo "[bootstrap] Cloning $REPO_URL into $PROJECT_DIR"
  if [ -n "${GITHUB_TOKEN:-}" ]; then
    askpass="$(mktemp)"
    chmod 700 "$askpass"
    cat > "$askpass" <<'EOF'
#!/usr/bin/env bash
case "$1" in
  *Username*) printf '%s\n' "x-access-token" ;;
  *Password*) printf '%s\n' "$GITHUB_TOKEN" ;;
  *) printf '\n' ;;
esac
EOF
    GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0 git clone "$REPO_URL" "$PROJECT_DIR"
    rm -f "$askpass"
  else
    git clone "$REPO_URL" "$PROJECT_DIR"
  fi
}

require_gpu() {
  echo "[bootstrap] Checking NVIDIA GPU"
  nvidia-smi
  docker info >/dev/null
  if ! docker info 2>/dev/null | grep -q 'Runtimes:.*nvidia'; then
    echo "[bootstrap] Docker is missing the nvidia runtime. Install/configure nvidia-container-toolkit first."
    exit 3
  fi
}

prepare_container() {
  cd "$PROJECT_DIR/IsaacLab/docker"
  echo "[bootstrap] Building and starting Isaac Lab container"
  python3 container.py build
  python3 container.py start

  echo "[bootstrap] Installing container runtime checks/fixes"
  docker exec isaac-lab-base bash -lc '
set -euo pipefail
if ! command -v vulkaninfo >/dev/null 2>&1; then
  apt-get update
  apt-get install -y --no-install-recommends vulkan-tools
  rm -rf /var/lib/apt/lists/*
fi
python_bin="/workspace/isaaclab/_isaac_sim/python.sh"
current="$("$python_bin" -m pip show warp-lang 2>/dev/null | awk "/^Version:/ {print \$2}")"
if [ "$current" != "1.4.2" ]; then
  "$python_bin" -m pip install warp-lang==1.4.2
fi
'

  echo "[bootstrap] Verifying Vulkan"
  docker exec isaac-lab-base vulkaninfo --summary
}

configure_hugging_face() {
  if [ -n "${HF_TOKEN:-}" ]; then
    echo "[bootstrap] HF_TOKEN is present; Hugging Face upload/download scripts can use it."
  else
    echo "[bootstrap] HF_TOKEN is not set; skipping Hugging Face auth setup."
  fi
}

clone_or_update_repo
require_gpu
prepare_container
configure_hugging_face

echo "[bootstrap] Ready. To run Phase 3:"
echo "cd $PROJECT_DIR && bash thesis/scripts/30_vision_vla.sh"
