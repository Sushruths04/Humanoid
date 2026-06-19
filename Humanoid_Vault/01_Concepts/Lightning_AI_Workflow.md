---
tags: [concept, infra, workflow]
---

# Lightning AI Workflow

## The model
This project follows a strict **"no local training"** workflow, because Isaac Sim needs a big GPU and the work moves between rented machines:

- **GitHub** → code, scripts, small docs.
- **Hugging Face** → large checkpoints, datasets, result artifacts.
- **Docker registry (GHCR)** → the reusable `humanoid-isaaclab` image.
- **Lightning AI machine** → disposable compute (L4 / L40S GPUs).
- **Local machine** → scratch only.

## Why this exists
Lightning machines are ephemeral — you lose them on shutdown. So the workflow is built around **fast machine-switching**: clone repo, `docker pull` the prebuilt image, run a **smoke test first**, then scale. The repo has a small army of helper scripts for this (`machine_switch.sh`, `bootstrap_remote_machine.sh`, `docker_image_portability.sh`, `autosave.sh`, `checkpoint.sh`) and runbooks (`MACHINE_CHANGE_RUNBOOK.md`, `EMERGENCY-TRANSFER.md`, `RECOVERY_GUIDE.md`).

## The golden rule
**Always run the smoke test before a long run** (e.g. `NUM_ENVS=16 MAX_ITERS=2 ... 30_vision_vla.sh`). It confirms the pipeline reaches PPO before you spend GPU-hours.

## Contrast with the rest of AUtonomous
This Lightning workflow is **unique to the Humanoid project**. The sibling `rl-advanced` / `rl-portfolio` work uses Colab/Kaggle free tiers instead — don't confuse the two.

Related: [[Pipeline_and_Scripts]] · [[Isaac_Lab_and_Isaac_Sim]]
