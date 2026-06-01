# Remote Machine Workflow

Use this when moving between Lightning/GPU machines. The goal is to avoid local tarball transfers.

For sudden GPU-machine changes, shutdowns, or handoff under time pressure, use `MACHINE_CHANGE_RUNBOOK.md`.

For a short command reference, use `MACHINE_SWITCH_QUICK_REF.md`.

## Source Of Truth

Do not treat the local Windows workspace as the project source of truth.

- **GitHub** stores code, scripts, docs, configs, lightweight metadata, and reproducibility notes.
- **Hugging Face** stores large artifacts: checkpoints, model shards, rollout media, logs, datasets, and training result bundles.
- **Remote GPU machines** are execution environments. They should clone/pull from GitHub, download needed artifacts from Hugging Face, run training, then upload results back to Hugging Face.
- **Local files** are only a temporary cache or emergency backup. Do not rely on local tarballs for normal handoff.
- **Secrets** stay outside Git. Use `SECRETS.env` locally or Lightning secrets remotely.

## Tokens

Set tokens only in the remote shell or Lightning secrets. Do not commit them.

```bash
export GITHUB_TOKEN="..."
export HF_TOKEN="..."
```

On this workspace, keep local persistent values in `SECRETS.env`:

```bash
source SECRETS.env
```

Use a GitHub fine-grained token with read access to `Sushruths04/Humanoid`. Add write access only when you want the machine to push code/results back.

## Bootstrap A Fresh Machine

For a private repo, clone with a temporary askpass helper so the token is not placed in the remote URL:

```bash
export GITHUB_TOKEN="..."
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
GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0 git clone https://github.com/Sushruths04/Humanoid.git
rm -f "$askpass"
cd Humanoid
bash thesis/scripts/machine_switch.sh bootstrap
```

This clones/updates the repo, builds and starts Isaac Lab Docker, installs `vulkan-tools` when needed, pins `warp-lang==1.4.2`, and verifies Vulkan.

If the repo is public temporarily, this shortcut also works:

```bash
cd /home/zeus/content
curl -fsSL https://raw.githubusercontent.com/Sushruths04/Humanoid/main/thesis/scripts/bootstrap_remote_machine.sh | bash
```

For a single entry point on a prepared machine:

```bash
bash thesis/scripts/machine_switch.sh status
bash thesis/scripts/machine_switch.sh train
bash thesis/scripts/machine_switch.sh sync
```

## Run Phase 3

```bash
cd /home/zeus/content/Humanoid
bash thesis/scripts/machine_switch.sh train
```

## Upload Results Back To Hugging Face

```bash
export HF_TOKEN="..."
export HF_REPO_ID="mitvho09/Humanoid-VLA-Artifacts"
bash thesis/scripts/machine_switch.sh sync
```

This collects `thesis/logs`, `thesis/checkpoints`, and Isaac Lab container logs, then uploads them under `phase3/` in the Hugging Face repo.

After any successful remote run, upload artifacts before stopping/deleting the machine:

```bash
source SECRETS.env  # or use Lightning secrets
bash thesis/scripts/machine_switch.sh sync
```

Then commit only code/doc/config changes to GitHub:

```bash
git status
git add <code-or-doc-files>
git commit -m "Describe the change"
git push origin main
```
