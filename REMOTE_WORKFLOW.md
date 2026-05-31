# Remote Machine Workflow

Use this when moving between Lightning/GPU machines. The goal is to avoid local tarball transfers.

## Tokens

Set tokens only in the remote shell or Lightning secrets. Do not commit them.

```bash
export GITHUB_TOKEN="..."
export HF_TOKEN="..."
```

Use a GitHub fine-grained token with read access to `Sushruths04/Humanoid`. Add write access only when you want the machine to push code/results back.

## Bootstrap A Fresh Machine

```bash
cd /home/zeus/content
curl -fsSL https://raw.githubusercontent.com/Sushruths04/Humanoid/main/thesis/scripts/bootstrap_remote_machine.sh | bash
```

For a private repo, download the script after cloning with your token, or run:

```bash
export GITHUB_TOKEN="..."
export REPO_URL="https://github.com/Sushruths04/Humanoid.git"
bash thesis/scripts/bootstrap_remote_machine.sh
```

This clones/updates the repo, builds and starts Isaac Lab Docker, installs `vulkan-tools` when needed, pins `warp-lang==1.4.2`, and verifies Vulkan.

## Run Phase 3

```bash
cd /home/zeus/content/Humanoid
bash thesis/scripts/30_vision_vla.sh
```

## Upload Results Back To Hugging Face

```bash
export HF_TOKEN="..."
export HF_REPO_ID="mitvho09/Humanoid-VLA-Artifacts"
bash thesis/scripts/sync_phase3_artifacts.sh
```

This collects `thesis/logs`, `thesis/checkpoints`, and Isaac Lab container logs, then uploads them under `phase3/` in the Hugging Face repo.
