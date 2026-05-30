# Isaac Lab Setup — Status, Options & Disk Management

**Last updated:** May 28, 2026  
**Workspace path:** `~/isaaclab-workspace/`  
(on this studio: `/teamspace/studios/this_studio/isaaclab-workspace/`)

This document records everything that was set up on your **CPU machine**, what is **still left for you**, your **three options** for moving to a GPU machine, and how to **stop Docker/disk usage from growing forever**.

---

## Quick summary

| Category | Status |
|----------|--------|
| System (Docker, Git, Python) | Done — was already on Lightning Studio |
| Repos cloned | Done |
| Isaac Lab Python (CPU, no GPU Sim) | Done — v0.54.3 |
| Docker images built | Done — ~33 GB total |
| Your project on GitHub | **You** need to push |
| SSH keys for GitHub | **You** need to set up (optional) |
| Export Docker `.tar` for transfer | **Optional** — not run yet |
| GPU machine training | **Later** — on GPU host |

---

## What has been done (complete checklist)

### Step 1 — System prerequisites
| Item | Status | Notes |
|------|--------|-------|
| Git | Done | v2.42.0 |
| Docker | Done | v28.0.1, `hello-world` works |
| Python 3.10+ | Done | Python 3.12.11 (conda `cloudspace`) |
| `apt upgrade` / extra tools | Skipped | Managed cloud studio; not required |

### Step 2 — Docker
| Item | Status | Notes |
|------|--------|-------|
| Docker installed | Done | Already present |
| User in `docker` group | Done | No `sudo` needed for docker |
| NGC login | Not verified | Pull worked without login on this machine; **may be required on a fresh GPU host** |

### Step 3 — Workspace layout
| Item | Status |
|------|--------|
| `~/isaaclab-workspace/` | Done |
| `checkpoints/` | Done (empty) |
| `logs/` | Done (empty) |

```
~/isaaclab-workspace/
├── IsaacLab/                    ← official repo
├── IsaacLab-Tutorial/            ← beginner tutorial
├── humanoid-gym/                 ← humanoid RL reference
├── awesome-humanoid-learning/    ← papers + repo list
├── my-humanoid-project/          ← YOUR code (git initialized)
├── checkpoints/                  ← model weights (host, not in git)
├── logs/                         ← training logs (host, not in git)
├── README-SETUP.md               ← technical setup notes
├── SETUP-STATUS.md               ← this file
├── setup-gpu-machine.sh          ← GPU quick-start script
└── export-docker-for-transfer.sh ← save images to .tar files
```

### Step 4 — Clone Isaac Lab
| Item | Status |
|------|--------|
| `git clone` Isaac Lab | Done |
| Shallow clone (`--depth 1`) | Done — saves space |

### Step 5 — Isaac Lab Python install (CPU mode)
| Item | Status | Notes |
|------|--------|-------|
| `pip install -e source/isaaclab` | Done |
| `pip install -e source/isaaclab_assets` | Done |
| `pip install -e source/isaaclab_tasks` | Done |
| `import isaaclab` | Done | v0.54.3 |
| Separate `venv` | Skipped | Lightning Studio allows only one conda env; packages installed in default env |

### Step 6 — Tutorial repos
| Repo | Status |
|------|--------|
| IsaacLab-Tutorial | Done |
| humanoid-gym | Done |
| awesome-humanoid-learning | Done |

### Step 7 — GitHub / SSH
| Item | Status |
|------|--------|
| Global `git config` (name/email) | **Not done** — use your real name/email |
| SSH key → GitHub | **Not done** — optional but recommended |

### Step 8 — Your project repo
| Item | Status |
|------|--------|
| `my-humanoid-project/` folders (`envs`, `scripts`, `configs`, …) | Done |
| `.gitignore` (checkpoints, logs, `.pth`, venv) | Done |
| Initial git commit | Done |
| Push to GitHub | **Not done** — needs your repo URL |

### Step 9 — Docker images
| Image | Status | Size (approx.) |
|-------|--------|----------------|
| `nvcr.io/nvidia/isaac-sim:5.1.0` | Pulled | 15.1 GB |
| `isaac-lab-base:latest` | Built | 17.6 GB |
| `nvcr.io/nvidia/isaac-lab:2.3.0` (from old guide) | **Not used** | Current Isaac Lab builds on Sim 5.1.0 |

### Step 10 — Verification
| Check | Result |
|-------|--------|
| Git | OK |
| Docker | OK |
| Python | 3.12.11 |
| `import isaaclab` | 0.54.3 |
| Workspace folders | OK |
| Docker build non-interactive config | Done — `docker/.container.cfg` with `X11_FORWARDING_ENABLED=0` |

### Helper scripts created
| Script | Purpose |
|--------|---------|
| `setup-gpu-machine.sh` | Clone repos + start container on GPU machine |
| `export-docker-for-transfer.sh` | Save Docker images to `.tar` for offline transfer |

---

## What is left for you to do

### Required (before GPU training)
1. **Create a GitHub repo** for `my-humanoid-project` and push:
   ```bash
   cd ~/isaaclab-workspace/my-humanoid-project
   git config --global user.name "Your Name"
   git config --global user.email "your@email.com"
   git remote add origin https://github.com/YOUR-USERNAME/my-humanoid-project.git
   git branch -M main
   git push -u origin main
   ```

2. **On GPU machine:** clone Isaac Lab + your project, start Docker, train (see options below).

### Recommended
3. **SSH key for GitHub** (no password when switching machines):
   ```bash
   ssh-keygen -t ed25519 -C "your@email.com"
   cat ~/.ssh/id_ed25519.pub
   # Add to GitHub → Settings → SSH and GPG keys
   ```

4. **NGC login on GPU machine** (if `docker pull` fails):
   ```bash
   docker login nvcr.io
   # Username: $oauthtoken
   # Password: <NGC API key from ngc.nvidia.com>
   ```

### Optional
5. **Export Docker images** to `.tar` if GPU machine has slow internet:
   ```bash
   bash ~/isaaclab-workspace/export-docker-for-transfer.sh
   ```
6. **Re-clone Isaac Lab with full history** if you need older tags (`git fetch --unshallow`).

---

## Three options for setting up the GPU machine

Pick based on **internet speed** and **disk space** on the GPU host.

### Option A — Git only (fastest setup, slowest download)

**Best when:** GPU machine has fast internet (~30+ min download once).

```bash
mkdir -p ~/isaaclab-workspace && cd ~/isaaclab-workspace
git clone https://github.com/isaac-sim/IsaacLab.git
git clone https://github.com/YOUR-USERNAME/my-humanoid-project.git

cd IsaacLab
# If fresh machine, may need: docker login nvcr.io
python docker/container.py start    # pulls/builds if images missing
python docker/container.py enter

isaaclab -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-H1-v0 --headless --num_envs 2048
```

**Pros:** Simple, no huge file copy.  
**Cons:** Re-downloads ~15–20 GB Isaac Sim + build time on GPU machine.

---

### Option B — Export/import Docker `.tar` files (best for slow internet)

**Best when:** You already built images on CPU (like now) and want **zero** re-download on GPU.

**On this CPU machine:**
```bash
bash ~/isaaclab-workspace/export-docker-for-transfer.sh
# Creates ~/isaaclab-workspace/exports/
#   isaac-sim-5.1.0.tar   (~15 GB)
#   isaac-lab-base.tar    (~18 GB)
```

Copy `exports/` to GPU machine (USB, S3, `scp`, etc.).

**On GPU machine:**
```bash
docker load -i isaac-sim-5.1.0.tar
docker load -i isaac-lab-base.tar
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab && python docker/container.py start && python docker/container.py enter
```

**Pros:** No NGC download on GPU; works offline.  
**Cons:** ~32 GB to copy; needs free disk on both machines.

---

### Option C — Helper script on GPU

**Best when:** You want a one-command start after GitHub push.

```bash
bash ~/isaaclab-workspace/setup-gpu-machine.sh YOUR-GITHUB-USERNAME
```

Then enter the container and run training (same as Option A).

**Pros:** Less typing; documents the flow.  
**Cons:** Still needs images on GPU (pull, load, or build).

---

## Comparison table

| | Option A (Git) | Option B (tar export) | Option C (script) |
|--|----------------|----------------------|-------------------|
| Setup time on GPU | ~30–60 min | ~10 min (after copy) | ~30–60 min |
| Internet on GPU | Heavy | Light / none | Heavy |
| Disk to transfer | Small (git only) | ~32 GB | Small |
| Works offline on GPU | No | Yes | No |

**Practical recommendation:** Use **Option B** if you already built images here and GPU internet is expensive or slow. Use **Option A** if GPU has fast unlimited bandwidth.

---

## Docker & disk: why things grow and how to fix it

Yes — if you are not careful, **Docker and your project can eat disk repeatedly**. Here is what grows and what to do.

### What makes Docker “bigger and bigger”

| Cause | What grows | Typical size |
|-------|------------|--------------|
| **Multiple image tags** | Old `isaac-lab-base` after each `build` | +17 GB per unused tag |
| **Dangling images** | `<none>` layers after rebuild | GBs |
| **Build cache** | `docker build` intermediate layers | ~700 MB–several GB |
| **Stopped containers** | Writable layer on container filesystem | Varies |
| **Data inside container** | Logs, checkpoints written *inside* container | Can be huge |
| **Exported `.tar` files** | `docker save` duplicates image on disk | ~32 GB |
| **Training outputs on host** | `checkpoints/`, `logs/`, TensorBoard | GBs–TBs over time |

Important: **Training checkpoints and logs should NOT live inside the Docker image.** They should live on the **host** in mounted folders or in `my-humanoid-project/checkpoints/` (gitignored).

### What Isaac Lab already does (good)

Isaac Lab Docker compose mounts host volumes for:
- Logs
- Data
- Docs cache

Your `.gitignore` already excludes:
- `checkpoints/`
- `logs/`
- `*.pth`, `*.pt`

Keep using those paths on the **host**, not only inside the container.

### Rules to keep disk under control

1. **Never commit weights to git** — only code + configs.
2. **Store checkpoints on host** — e.g. `~/isaaclab-workspace/checkpoints/` or `my-humanoid-project/checkpoints/`.
3. **Prune after rebuilding** an image:
   ```bash
   docker image prune -f          # remove dangling images
   docker builder prune -f        # clear build cache
   ```
4. **Remove old tags** when you know you only need `latest`:
   ```bash
   docker images                  # list all
   docker rmi <old-image-id>      # delete specific old image
   ```
5. **Do not run `export-docker-for-transfer.sh` repeatedly** unless you need a new copy — each export duplicates ~32 GB on disk.
6. **Delete `.tar` after `docker load` on GPU machine** if space is tight:
   ```bash
   docker load -i isaac-lab-base.tar && rm isaac-lab-base.tar
   ```
7. **Rotate old checkpoints** — keep last N best models, delete the rest:
   ```bash
   # Example: keep only 5 newest .pt files
   ls -t checkpoints/*.pt | tail -n +6 | xargs rm -f
   ```

### Safe cleanup commands (read before running)

```bash
# See what Docker is using
docker system df

# Remove unused images, stopped containers, networks (SAFE-ish)
docker system prune -f

# Also remove unused images not tied to a container (more aggressive)
docker image prune -a -f

# Clear build cache only
docker builder prune -f

# Nuclear option — removes ALL unused images/containers/volumes (careful!)
docker system prune -a --volumes -f
```

**Do not** run `docker system prune -a` if you still need `isaac-lab-base` or `isaac-sim` and have no way to re-pull them.

### Where your project files should grow (allowed)

| Location | Grow? | In git? |
|----------|-------|---------|
| `my-humanoid-project/envs/`, `scripts/`, `configs/` | Small | Yes |
| `checkpoints/`, `logs/` | Large | **No** (.gitignore) |
| Docker images | Fixed ~33 GB until rebuild | N/A |
| `exports/*.tar` | Huge, temporary | N/A — delete after load |

### If disk is full on this machine right now

Current rough usage:
- Docker images: ~17.6 GB reported (isaac-sim + isaac-lab-base ≈ 33 GB logical)
- Workspace (git repos): ~143 MB
- Build cache: ~713 MB reclaimable

To free space without losing your built image:
```bash
docker builder prune -f
docker image prune -f
# Do NOT delete isaac-lab-base or isaac-sim if you still need them
```

---

## Original guide vs what we actually did

| Your guide said | What we did |
|-----------------|-------------|
| `isaac-lab:2.3.0` image | `isaac-sim:5.1.0` + `docker/container.py build` → `isaac-lab-base:latest` |
| `python3 -m venv isaaclab-env` | Installed into studio conda env (venv blocked) |
| Steps 1–2 apt/docker install | Skipped — already installed |
| Step 9 pull only | Pull **and** build completed |

---

## GPU machine checklist (when you switch)

```
⬜ Clone my-humanoid-project from GitHub
⬜ Clone Isaac Lab (or copy workspace)
⬜ Have Docker images (pull, load .tar, or build)
⬜ docker login nvcr.io (if pull fails)
⬜ python docker/container.py start
⬜ python docker/container.py enter
⬜ Run training command
⬜ Save checkpoints to host checkpoints/ (not inside image)
```

---

## Useful commands reference

```bash
# Verify everything
echo "=== Git ===" && git --version
echo "=== Docker ===" && docker --version
echo "=== Python ===" && python3 --version
echo "=== Isaac Lab ===" && python -c "import isaaclab; print(isaaclab.__version__)"
ls ~/isaaclab-workspace/
docker images | grep -E 'isaac|nvcr'
docker system df

# Start training (inside container)
isaaclab -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-H1-v0 --headless --num_envs 2048
```

---

## Questions?

- **More detail on Docker/build:** see `README-SETUP.md`
- **Export images:** run `export-docker-for-transfer.sh`
- **GPU quick start:** run `setup-gpu-machine.sh YOUR-USERNAME`
