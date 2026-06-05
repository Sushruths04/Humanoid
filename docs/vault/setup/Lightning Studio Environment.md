---
tags: [setup, lightning-ai, environment, storage]
---

# Lightning Studio Environment

## The Mental Model: Persistent vs. Ephemeral

This is the single most important concept for working on Lightning AI. **Get this wrong and you lose work.**

```
PERSISTENT (survives machine restarts / GPU swaps):
  /teamspace/studios/this_studio/...   ← your code, git, conda, HF token cache, SSH keys
  ~/.ssh/                              ← SSH keys (survive restart but NOT machine sleep/wake)
  ~/.cache/huggingface/                ← HF token

EPHEMERAL (wiped on every machine restart / GPU swap):
  Docker images                        ← you must re-pull every time
  Docker containers                    ← you must re-create every time
  /tmp, /var, and other OS-level dirs
```

**Consequence:** every time you get a fresh GPU machine, your code is there but the Isaac Sim container is gone. The first thing you do is always re-pull the image and restart the container.

---

## Key Paths

| What | Path |
|---|---|
| Repo root | `/teamspace/studios/this_studio/Humanoid` |
| Conda python | `/home/zeus/miniconda3/envs/cloudspace/bin/python` |
| Isaac Sim logs (in container) | `/workspace/isaaclab/logs/rsl_rl/g1_flat/<timestamp>/` |
| Programs (bind-mounted) | host `programs/` → container `/workspace/programs` |
| Task configs (bind-mounted) | host `my-humanoid-project/` → container `/workspace/my-humanoid-project` |
| GPU session log dir | `_runlogs/` (gitignored, host-owned) |

---

## SSH Access

```bash
ssh s_01kt558jf0ra2chne251dtsg8k@ssh.lightning.ai
```

If you get `Permission denied (publickey)`: see [[SSH Key Recovery]].

---

## Quick Health Check (run on fresh machine)

```bash
# 1. GPU alive?
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 2. Repo on right branch?
cd /teamspace/studios/this_studio/Humanoid
git checkout feat/planned-scripts && git pull

# 3. Container alive?
docker ps --format "{{.Names}} {{.Status}}" | grep isaac
# if empty -> see [[Isaac Sim Docker Container]]

# 4. CPU tests pass?
cd programs
/home/zeus/miniconda3/envs/cloudspace/bin/python -m pytest -q
# should say "34 passed"
```

---

## GitHub Push

```bash
# Uses ssh:// remote + ~/.ssh/github_humanoid key
# (gitconfig insteadOf rewrites git@ -> https, so use ssh:// form explicitly)
git push origin feat/planned-scripts
```

## Hugging Face Upload

```bash
# Login (token stored at ~/.cache/huggingface/token — persists)
hf auth login --token <your_token>
# Upload a file
hf upload mitvho09/humanoid-g1-nav <local_path> <repo_path>
```

HF model repo: `huggingface.co/mitvho09/humanoid-g1-nav` — checkpoints, results, demo videos all live here.

---

## Related

- [[Isaac Sim Docker Container]]
- [[GHCR Image & Auth]]
- [[SSH Key Recovery]]
- [[PYTHONPATH & Python Interpreters]]
- [Machine Change Runbook](../../MACHINE_CHANGE_RUNBOOK.md)
- [CPU to GPU Switch Guide](../../CPU_TO_GPU_MACHINE_SWITCH.md)
