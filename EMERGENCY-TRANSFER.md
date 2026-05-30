# Emergency transfer — machine closing in 30 minutes

Use this when you **cannot come back** to this machine. Goal: save **your work** (code, checkpoints, logs, configs) — **not** the 33 GB Docker images (rebuild those on the new machine).

---

## 30-minute action plan (do in this order)

| Minutes | Action | Why |
|---------|--------|-----|
| 0–2 | Run `bash emergency-pack.sh` | Finds and packs all your data |
| 2–5 | Push code to GitHub | Small, fast, permanent |
| 5–25 | Upload `exports/emergency-backup-*.tar.gz` | Checkpoints + logs (the big stuff) |
| 25–30 | Verify upload finished; note paths on new machine | Don’t close until upload completes |

**Skip on a deadline:** Docker images (`isaac-sim`, `isaac-lab-base`). Re-pull or re-build on the new GPU machine (see `SETUP-STATUS.md`).

---

## Where your simulation data actually lives

Depending on how you ran training, files are in **different places**. Check all of these:

### A) Ran training on the host (pip / conda, no container)

| What | Typical path |
|------|----------------|
| RSL-RL logs + checkpoints | `~/isaaclab-workspace/IsaacLab/logs/rsl_rl/<experiment>/<date_time>/` |
| SKRL logs | `~/isaaclab-workspace/IsaacLab/logs/skrl/...` |
| Your project | `~/isaaclab-workspace/my-humanoid-project/` |
| Shared checkpoints folder | `~/isaaclab-workspace/checkpoints/` |

Inside each run folder you usually want:
- `model_*.pt` — **weights (critical)**
- `params/env.yaml`, `params/agent.yaml` — **config to reproduce run**
- `events.out.tfevents.*` — TensorBoard (optional, large)

### B) Ran training inside Docker (`container.py enter`)

Data may be in **Docker named volumes**, not in your home folder:

| Volume name | Contents |
|-------------|----------|
| `isaac-lab-logs` | `IsaacLab/logs/` (training runs) |
| `isaac-lab-data` | `data_storage/` |
| `isaac-lab-docs` | docs build cache (skip) |

**You must export volumes before shutdown** — see `emergency-pack.sh`.

### C) Other tools

| Tool | Where to look |
|------|----------------|
| Weights & Biases | [wandb.ai](https://wandb.ai) — syncs online if you used `--wandb` |
| Hugging Face | If you pushed a model during training |
| Custom path | Whatever you passed as `--checkpoint` or `log_dir` |

---

## One command: pack everything

```bash
cd ~/isaaclab-workspace
bash emergency-pack.sh
```

This creates:

```
~/isaaclab-workspace/exports/
  emergency-backup-YYYYMMDD-HHMMSS.tar.gz   ← upload THIS
  MANIFEST.txt                               ← list of what was included
```

Then upload `emergency-backup-*.tar.gz` using one of the methods below.

---

## How to upload big files (pick one)

### 1) Hugging Face Hub (good for checkpoints, free tier)

```bash
pip install -q huggingface_hub
huggingface-cli login   # paste token from huggingface.co/settings/tokens

export BACKUP=~/isaaclab-workspace/exports/emergency-backup-*.tar.gz
huggingface-cli upload YOUR-USERNAME/isaaclab-emergency-backup "$(ls -t $BACKUP | head -1)" --repo-type model
```

On the new machine:
```bash
huggingface-cli download YOUR-USERNAME/isaaclab-emergency-backup --local-dir ./restored
```

### 2) AWS S3 (fast if you have credentials)

```bash
aws s3 cp ~/isaaclab-workspace/exports/emergency-backup-*.tar.gz \
  s3://YOUR-BUCKET/isaaclab-backup/ --only-show-errors
```

Download on new machine:
```bash
aws s3 cp s3://YOUR-BUCKET/isaaclab-backup/emergency-backup-....tar.gz ./
```

### 3) `scp` to another server you control

```bash
scp ~/isaaclab-workspace/exports/emergency-backup-*.tar.gz \
  user@OTHER-MACHINE:/home/user/backups/
```

### 4) Google Drive / Dropbox

If `rclone` is configured:
```bash
rclone copy ~/isaaclab-workspace/exports/ gdrive:isaaclab-backup -P
```

### 5) GitHub — **code only, not big checkpoints**

GitHub rejects files **> 100 MB**. Use it for code + configs, not full training archives.

```bash
cd ~/isaaclab-workspace/my-humanoid-project
git add -A && git commit -m "Emergency save before shutdown" && git push
```

For a single small checkpoint (<100 MB), Git LFS:
```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes && git add checkpoints/ && git commit && git push
```

### 6) Split huge archive (if upload keeps failing)

```bash
cd ~/isaaclab-workspace/exports
ARCHIVE=$(ls -t emergency-backup-*.tar.gz | head -1)
split -b 2000M "$ARCHIVE" "${ARCHIVE}.part-"
# Upload each .part-* file separately, then on new machine:
# cat emergency-backup-....tar.gz.part-* > emergency-backup-....tar.gz
```

---

## On the new machine — restore

```bash
mkdir -p ~/isaaclab-workspace && cd ~/isaaclab-workspace

# 1. Download your backup (from S3 / HF / scp / etc.)
# 2. Extract
tar -xzf emergency-backup-YYYYMMDD-HHMMSS.tar.gz -C .

# 3. Clone Isaac Lab again (or use included copy if you packed it)
git clone https://github.com/isaac-sim/IsaacLab.git   # if not in backup

# 4. Restore logs into Isaac Lab tree
#    (emergency-pack puts logs under restored/logs/rsl_rl/ — copy into IsaacLab/logs/)

# 5. Rebuild Docker OR pull images (not in backup)
cd IsaacLab && python docker/container.py build   # or load .tar from SETUP-STATUS.md

# 6. Resume training from checkpoint
isaaclab -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task YOUR_TASK --checkpoint /path/to/model_XXXX.pt
```

---

## What to save vs what to skip

| Save (your work) | Skip (rebuild on new machine) |
|------------------|-------------------------------|
| `*.pt`, `*.pth`, `*.ckpt` | Docker images (~33 GB) |
| `logs/rsl_rl/.../params/*.yaml` | `docker builder` cache |
| `my-humanoid-project/` code | Isaac Sim install (re-pull) |
| Custom envs / scripts | `__pycache__/`, `.git` in clones (optional) |
| `wandb/` offline runs (if any) | Tutorial repos (re-clone) |
| Recorded videos `.mp4` (if any) | |

---

## If you used Weights & Biases

If training synced to W&B, checkpoints and metrics may **already be in the cloud**. On the new machine:

```bash
wandb login
# Download artifact from the run page on wandb.ai
```

You might only need to push **custom code** to GitHub.

---

## Prevent this next time

1. **Log to a cloud-backed path from day one** — W&B, S3, or HF Hub every N iterations.
2. **Mount checkpoints to host folder** outside Docker:
   ```bash
   # When starting container, ensure logs bind to host (Isaac Lab compose already uses volumes — export them regularly)
   ```
3. **Push code to GitHub after every session.**
4. **Periodic checkpoint upload:**
   ```bash
   huggingface-cli upload user/repo logs/rsl_rl/.../model_500.pt
   ```
5. **Do not store the only copy inside a Docker volume** without backing up to object storage.

---

## Quick find: “where are my big files?”

```bash
find ~/isaaclab-workspace -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' \) -exec du -h {} \; 2>/dev/null | sort -hr | head -20
find ~/isaaclab-workspace -path '*/logs/rsl_rl/*' -type d 2>/dev/null
docker volume ls
du -sh ~/isaaclab-workspace/exports/* 2>/dev/null
```
