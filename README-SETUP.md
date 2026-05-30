# Isaac Lab Workspace — CPU Prep → GPU Transfer

## What’s already done on this machine

| Step | Status |
|------|--------|
| Docker installed & working | ✅ |
| `~/isaaclab-workspace/` layout | ✅ |
| Isaac Lab cloned | ✅ |
| Isaac Lab Python packages (CPU, no Sim) | ✅ v0.54.3 |
| Tutorial repos cloned | ✅ |
| `my-humanoid-project/` skeleton + git init | ✅ |
| Isaac Sim Docker image pull | ✅ `nvcr.io/nvidia/isaac-sim:5.1.0` (15.1 GB) |
| Isaac Lab Docker **build** | ✅ `isaac-lab-base:latest` (17.6 GB) |

**Note:** Lightning Studio blocks extra `python3 -m venv`; packages are in the default conda env.

---

## Finish Docker on this CPU machine

### 1. NGC login (one time)

1. Sign up: https://ngc.nvidia.com  
2. Profile → **Setup** → Generate API Key  
3. Login:

```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: <your NGC API key>
```

### 2. Pull Isaac Sim base (~15–20 GB)

```bash
docker pull nvcr.io/nvidia/isaac-sim:5.1.0
```

Your guide mentioned `nvcr.io/nvidia/isaac-lab:2.3.0`. Current Isaac Lab `main` uses **Isaac Sim 5.1.0** and builds its own image via `docker/container.py`.

### 3. Build Isaac Lab image (CPU is fine — no GPU needed to build)

```bash
cd ~/isaaclab-workspace/IsaacLab
python docker/container.py build
```

### 4. Optional — export for offline GPU machine

```bash
bash ~/isaaclab-workspace/export-docker-for-transfer.sh
```

Copy `~/isaaclab-workspace/exports/*.tar` to the GPU host, then:

```bash
docker load -i isaac-sim-5.1.0.tar
docker load -i isaac-lab-base.tar
```

---

## Push your project to GitHub

```bash
cd ~/isaaclab-workspace/my-humanoid-project
git remote add origin https://github.com/YOUR-USERNAME/my-humanoid-project.git
git branch -M main
git push -u origin main
```

SSH (recommended for machine switching):

```bash
ssh-keygen -t ed25519 -C "your@email.com"
cat ~/.ssh/id_ed25519.pub   # add to GitHub → Settings → SSH Keys
```

---

## GPU machine — fast path (~5 min)

```bash
bash ~/isaaclab-workspace/setup-gpu-machine.sh YOUR-USERNAME
```

Or manually:

```bash
cd ~/isaaclab-workspace/IsaacLab
python docker/container.py start
python docker/container.py enter
isaaclab -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-H1-v0 --headless --num_envs 2048
```

---

## Verify

```bash
echo "=== Git ===" && git --version
echo "=== Docker ===" && docker --version
echo "=== Python ===" && python3 --version
echo "=== Isaac Lab ===" && python -c "import isaaclab; print(isaaclab.__version__)"
echo "=== Workspace ===" && ls ~/isaaclab-workspace/
docker images | grep -E 'isaac|nvcr'
```
