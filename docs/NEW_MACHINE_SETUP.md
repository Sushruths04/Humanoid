# New Machine Setup — Complete Runbook

Use this every time you spin up a fresh Lightning Studio machine. Covers pulling everything from GitHub + HuggingFace, setting up all environments, and knowing which machine to rent for each remaining task.

---

## Step 0 — Which Machine to Rent

| Task | GPU | VRAM | Why | Est. cost |
|---|---|---|---|---|
| T1 eval / T2 WM | **L4** | 24 GB | GR00T inference needs 16GB; L4 is cheapest that fits | ~$0.50/hr × 1 hr |
| P3 Vision Nav | **L40S** | 48 GB | Isaac Lab + cameras + Vulkan rendering needs 24GB+ | ~$1.50/hr × 3 hrs |
| T3 Vision Manip | **L4** | 24 GB | Pixel LIBERO policy, no Isaac | ~$0.50/hr × 2 hrs |
| T1 LoRA fine-tune | **L40S** | 48 GB | 3B param LoRA needs 40GB+ | ~$1.50/hr × 4 hrs |
| P4 Cosmos Predict | **A100-80G** | 80 GB | Cosmos video model post-train | ~$3/hr × 6 hrs |
| C5 Capstone | **L40S** or A100 | 48-80 GB | Isaac Lab + GR00T combined | ~$1.50/hr × 8 hrs |

**Next task is P3 — rent an L40S.**

---

## Step 1 — SSH Key Setup

Every new Lightning Studio wipes SSH keys. After machine starts:

1. Open Studio in browser → **Settings → SSH Keys** → add your public key
2. Connect: `ssh -i ~/.ssh/lightning_new s_01kt558jf0ra2chne251dtsg8k@ssh.lightning.ai`
3. Test: `nvidia-smi` → should show GPU name + VRAM

---

## Step 2 — Pull Repo from GitHub

```bash
cd /teamspace/studios/this_studio

# Clone if first time on this machine
git clone https://github.com/Sushruths04/Humanoid.git Humanoid
cd Humanoid
git checkout feat/planned-scripts

# OR if repo already exists, just pull latest
cd /teamspace/studios/this_studio/Humanoid
git pull origin feat/planned-scripts
```

---

## Step 3 — Pull Checkpoints from HuggingFace

```bash
cd /teamspace/studios/this_studio/Humanoid

# GR00T N1.7 LIBERO checkpoint (needed for T1/T2)
huggingface-cli download nvidia/GR00T-N1.7-LIBERO \
    --local-dir programs/checkpoints/groot_n17/libero_spatial \
    --include "libero_spatial/**"

# P0/P1 nav checkpoints (on our HF)
huggingface-cli download mitvho09/humanoid-g1-nav \
    --local-dir programs/checkpoints \
    --include "checkpoints/**" \
    --repo-type dataset

# T0 BC checkpoint
# (included in above download under checkpoints/t0_bc/)
```

---

## Step 4 — Set Up groot_env (for T1, T2)

```bash
# Create env (skip if already exists on this machine)
/home/zeus/miniconda3/bin/conda create -n groot_env python=3.10 -y

# Install PyTorch
/home/zeus/miniconda3/envs/groot_env/bin/pip install \
    torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128

# Clone Isaac-GR00T (always to /tmp — ephemeral is fine)
git clone --depth=1 https://github.com/NVIDIA/Isaac-GR00T.git /tmp/Isaac-GR00T

# Install gr00t
cd /tmp/Isaac-GR00T
/home/zeus/miniconda3/envs/groot_env/bin/pip install -e '.' --no-build-isolation

# flash-attn — MUST use prebuilt wheel (pip install flash-attn fails)
/home/zeus/miniconda3/envs/groot_env/bin/pip install \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Fix numpy conflict
/home/zeus/miniconda3/envs/groot_env/bin/pip install 'numpy==1.26.4'

# Clone LIBERO + fix namespace pth
git clone --depth=1 https://github.com/Lifelong-Robot-Learning/LIBERO.git /tmp/LIBERO
echo '/tmp/LIBERO' > /home/zeus/miniconda3/envs/groot_env/lib/python3.10/site-packages/libero_path.pth

# Verify
/home/zeus/miniconda3/envs/groot_env/bin/python -c 'import gr00t; import libero; print("groot_env OK")'
```

---

## Step 5 — Set Up libero_env (for T0 BC training only)

```bash
# Create env (Python 3.9 — LIBERO is incompatible with 3.10+)
/home/zeus/miniconda3/bin/conda create -n libero_env python=3.9 -y

/home/zeus/miniconda3/envs/libero_env/bin/pip install \
    numpy h5py scipy matplotlib \
    torch torchvision --index-url https://download.pytorch.org/whl/cu118 \
    mujoco==2.3.7 robosuite==1.4.1 \
    bddl hydra-core==1.3.2 omegaconf \
    easydict cloudpickle robomimic \
    gym==0.26.2 huggingface_hub

# LIBERO (use /tmp — ephemeral fine)
git clone --depth=1 https://github.com/Lifelong-Robot-Learning/LIBERO.git /tmp/LIBERO
/home/zeus/miniconda3/envs/libero_env/bin/pip install /tmp/LIBERO/ --no-deps
echo '/tmp/LIBERO' > /home/zeus/miniconda3/envs/libero_env/lib/python3.9/site-packages/libero_path.pth

# Download LIBERO demos (300MB, needed for T0 training only)
/home/zeus/miniconda3/envs/libero_env/bin/python \
    /tmp/LIBERO/benchmark_scripts/download_libero_datasets.py \
    --download-dir /teamspace/studios/this_studio/libero_datasets \
    --datasets libero_spatial --use-huggingface

# Verify
/home/zeus/miniconda3/envs/libero_env/bin/python -c 'import libero; print("libero_env OK")'
```

---

## Step 6 — After Every Machine Restart (envs exist, /tmp wiped)

```bash
# Re-clone /tmp dependencies in parallel
git clone --depth=1 https://github.com/NVIDIA/Isaac-GR00T.git /tmp/Isaac-GR00T -q &
git clone --depth=1 https://github.com/Lifelong-Robot-Learning/LIBERO.git /tmp/LIBERO -q &
wait

# Reinstall gr00t
cd /tmp/Isaac-GR00T
/home/zeus/miniconda3/envs/groot_env/bin/pip install -e '.' --no-build-isolation -q

# Restore pth files
echo '/tmp/LIBERO' > /home/zeus/miniconda3/envs/groot_env/lib/python3.10/site-packages/libero_path.pth
echo '/tmp/LIBERO' > /home/zeus/miniconda3/envs/libero_env/lib/python3.9/site-packages/libero_path.pth

# Verify both
/home/zeus/miniconda3/envs/groot_env/bin/python -c 'import gr00t; import libero; print("groot_env OK")'
/home/zeus/miniconda3/envs/libero_env/bin/python -c 'import libero; print("libero_env OK")'
```

---

## What to Run Next — P3 (Vision Nav)

**Machine needed: L40S 48GB**

P3 adds RGB camera observations to the Isaac Lab G1 navigation tasks. The policy must learn to navigate using pixels instead of (or in addition to) proprioceptive state.

```bash
cd /teamspace/studios/this_studio/Humanoid
git pull origin feat/planned-scripts

# P3 scripts are scaffolded — Claude will write them when machine is available
# Expected location: programs/p3_vision_nav/
```

DoD: pixel-conditioned nav policy trains to ≥60% success on CommandNav.

---

## What to Run Next — T3 (Vision Manipulation)

**Machine needed: L4 16GB**

T3 switches LIBERO to pixel-only observations (no proprioception). Uses the `groot_env` environment.

DoD: image-conditioned policy achieves measurable success (>20%) on libero_spatial.

---

## What to Run Next — P4 (Cosmos Predict)

**Machine needed: A100-80G (burst, expensive)**

Fine-tune Cosmos video world model on robot rollout videos.

```bash
# Cosmos requires 80GB — don't attempt on smaller GPUs
# Will use: programs/p4_cosmos/ (to be written)
```

---

## Repo Structure Quick Reference

```
Humanoid/
├── programs/
│   ├── common/              ← shared eval harness, rewards, video recorder
│   ├── world_model/         ← Dreamer-mini RSSM (P2, reused by T2)
│   ├── t0_manip_foundation/ ← BC baseline (T0)
│   ├── t1_groot_lora/       ← GR00T eval harness (T1)
│   ├── t2_manip_wm/         ← WM for manipulation (T2)
│   ├── p3_vision_nav/       ← vision nav scaffold (P3 — to be completed)
│   ├── checkpoints/         ← all model checkpoints (persistent)
│   ├── videos/              ← all demo + eval videos
│   └── data/                ← rollout datasets
├── docs/
│   ├── results/             ← all result markdown files
│   ├── vault/               ← Obsidian notes (tasks, reference, sessions)
│   ├── GR00T_ROBOT_GUIDE.md ← standalone GR00T setup guide
│   └── NEW_MACHINE_SETUP.md ← this file
└── README.md
```

---

## HuggingFace Assets

Repo: `mitvho09/humanoid-g1-nav` (dataset repo)

| Path on HF | Contents |
|---|---|
| `checkpoints/g1_commandnav/` | P0 CommandNav checkpoint |
| `checkpoints/g1_commandnav_stable/` | P0-stable (low fall rate) |
| `checkpoints/g1_obstaclenav/` | P1.3 ObstacleNav checkpoint |
| `checkpoints/g1_seqnav/` | P1.4 SeqNav checkpoint |
| `checkpoints/t0_bc/` | T0 BC baseline checkpoint |
| `checkpoints/world_model/` | P2 + T2 world model checkpoints |
| `videos/t1_groot/` | 10 GR00T success videos (all LIBERO Spatial tasks) |
| `videos/` | Nav demo reel (commandnav, obstaclenav, seqnav) |

```bash
# Download everything from HF at once
huggingface-cli download mitvho09/humanoid-g1-nav \
    --local-dir /teamspace/studios/this_studio/Humanoid \
    --repo-type dataset
```

---

## SSH Key Reminder

Lightning Studio drops SSH keys every restart. Fix:
1. Browser → Studio → Settings → SSH Keys → add public key
2. Or run from local terminal: `cat ~/.ssh/lightning_new.pub` and paste it

---

*Last updated: 2026-06-06. T1 + T2 complete. Next: P3 on L40S.*
