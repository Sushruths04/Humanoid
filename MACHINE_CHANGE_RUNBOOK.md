# Machine Change Runbook

Use this when the current GPU machine may stop soon, or when moving from one Lightning/GPU machine to another.

## Rule

Do not depend on local files or a running GPU machine for handoff.

- Push code/docs/configs to GitHub.
- Upload large artifacts to Hugging Face.
- Keep secrets out of Git.
- Treat every GPU machine as disposable.

## 30 Minutes Before Shutdown

Run these from the remote project directory:

```bash
cd /home/zeus/content/Humanoid
```

Save large run artifacts to Hugging Face:

```bash
export HF_TOKEN="..."
export HF_REPO_ID="mitvho09/Humanoid-VLA-Artifacts"
bash thesis/scripts/machine_switch.sh sync
```

Commit and push code/docs/config changes to GitHub:

```bash
export GITHUB_TOKEN="..."

git status
git add <changed-code-doc-config-files>
git commit -m "Save remote progress"

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

GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0 git push origin main
rm -f "$askpass"
```

Write a short handoff note if the run did not finish:

```bash
cat > CURRENT_REMOTE_STATUS.md <<'EOF'
# Current Remote Status

- Date:
- Machine:
- Last command:
- Last successful step:
- Current blocker:
- Artifacts uploaded to:
- Next command to run:
EOF

git add CURRENT_REMOTE_STATUS.md
git commit -m "Add current remote status"
GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0 git push origin main
```

## If Training Is Still Running

Check whether it is making progress:

```bash
nvidia-smi
tail -80 thesis/logs/g1_vision/train.log
```

If it is healthy and you have enough time, let it continue.

If the machine must stop soon, terminate training cleanly if possible:

```bash
docker exec isaac-lab-base bash -lc "pkill -TERM -f custom_train.py || true"
sleep 10
```

If it does not stop:

```bash
docker exec isaac-lab-base bash -lc "pkill -KILL -f custom_train.py || true"
```

Then upload artifacts with `machine_switch.sh sync`.

## Starting A New Machine

Install/connect SSH from the Lightning UI, then:

```bash
ssh <user>@ssh.lightning.ai
```

Set tokens in the remote shell or Lightning secrets:

```bash
export GITHUB_TOKEN="..."
export HF_TOKEN="..."
```

Clone the private GitHub repo without putting the token in the URL:

```bash
cd /home/zeus/content

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

GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0 \
git clone https://github.com/Sushruths04/Humanoid.git

rm -f "$askpass"
cd Humanoid
```

Bootstrap Docker/Isaac Lab:

```bash
bash thesis/scripts/machine_switch.sh bootstrap
```

If prior artifacts are needed, download them from Hugging Face:

```bash
python3 thesis/scripts/hf_download.py
```

## Resume Phase 3

For the current Vision VLA smoke run:

```bash
cd /home/zeus/content/Humanoid
bash thesis/scripts/machine_switch.sh train
```

If the 32-env smoke test succeeds, increase `NUM_ENVS` in `thesis/scripts/30_vision_vla.sh` based on VRAM:

- Start with `64`.
- If stable, try `128`.
- Stop increasing when GPU memory is close to full or training becomes unstable.

After scaling, increase `MAX_ITERS` from `300` to a production value such as `2000` or more.

For the current L40S run, the working production setting is:

```bash
export NUM_ENVS=2048
export MAX_ITERS=5000
bash thesis/scripts/machine_switch.sh train
```

## End Of Session Checklist

- `git status` checked.
- Code/docs/configs pushed to GitHub.
- Large checkpoints/logs uploaded to Hugging Face.
- `CURRENT_REMOTE_STATUS.md` updated if there is any unresolved issue.
- No token committed to Git.
