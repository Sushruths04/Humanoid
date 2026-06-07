# P4 Cosmos Predict — Machine Setup & Full Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Go from zero (fresh A100-80G machine, SSH key provided) to completed P4 — action-conditioned Cosmos Predict 2.5 LoRA world model trained on G1 nav rollouts, K-step rollout validation, and CEM planning demo. Results uploaded to HuggingFace.

**Architecture:** Two phases. Phase B is CPU-only (install, download, no GPU charged). Phase C is GPU execution: five checkpoints (CP4.1 → CP4.5) with a hard smoke gate at CP4.3 before spending GPU budget.

**Tech Stack:** conda Python 3.10, PyTorch 2.7.1+cu128, cosmos-predict2 (NVIDIA GHCR/GitHub), bitsandbytes (8-bit Adam), HuggingFace PEFT (LoRA), HuggingFace Hub, Isaac Lab Docker (for data collection), RSL-RL (P3 nav policy).

---

## File Structure (Phase A — already done in this session)

| File | Status | Purpose |
|---|---|---|
| `programs/p4_cosmos_world_sim/__init__.py` | create | package init |
| `programs/p4_cosmos_world_sim/collect_rollouts.py` | create | run P3 nav in docker → save (frame_t, action_t, frame_t+1) as HDF5 |
| `programs/p4_cosmos_world_sim/cp41_inference.py` | create | CP4.1: stock Cosmos inference → mp4 |
| `programs/p4_cosmos_world_sim/cp42_verify_data.py` | create | CP4.2: print dataset shapes + sample frame |
| `programs/p4_cosmos_world_sim/cp43_train.py` | create | CP4.3: LoRA post-train (--smoke / --max-steps) |
| `programs/p4_cosmos_world_sim/cp44_rollout.py` | create | CP4.4: K-step action-conditioned rollout |
| `programs/p4_cosmos_world_sim/cp45_plan.py` | create | CP4.5: CEM planner on imagined rollouts |
| `programs/p4_cosmos_world_sim/setup_machine.sh` | create | Phase B one-shot machine bootstrap |
| `programs/p4_cosmos_world_sim/run_p4.sh` | create | Phase C orchestrator (CP4.1 → CP4.5) |

---

## PHASE A — CPU Prep (done in this session, push to branch)

> These tasks were executed before the machine was available. All code is on branch `feat/planned-scripts`. Skip to Phase B when you have the SSH key.

**All scripts are in `programs/p4_cosmos_world_sim/`. See the actual files — they are the implementation.**

---

## PHASE B — Machine Setup (CPU phase, no GPU cost)

> Execute these steps immediately after SSH-ing into the A100 machine. GPU is NOT needed yet — CPU only. This takes ~20-40 min (dominated by model downloads).

### Task B1: SSH in and pull the repo

**Files:**
- `programs/p4_cosmos_world_sim/setup_machine.sh`

- [ ] **Step B1.1: SSH into the machine**

```bash
ssh -i <key_file> <user>@<host>
# or if Lightning Studio: open the terminal tab
```

- [ ] **Step B1.2: Check conda available**

```bash
which conda || /home/zeus/miniconda3/bin/conda --version
# Lightning Studio: /home/zeus/miniconda3/bin/conda
# Generic Ubuntu: may need to install miniconda
```

If conda is missing, install it:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mini.sh
bash /tmp/mini.sh -b -p /home/zeus/miniconda3
echo 'export PATH=/home/zeus/miniconda3/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
```

- [ ] **Step B1.3: Clone or pull the repo**

```bash
# If Lightning Studio (repo already at ~/Humanoid):
cd ~/Humanoid && git pull origin feat/planned-scripts

# If fresh machine:
git clone https://github.com/<your-repo> ~/Humanoid
cd ~/Humanoid && git checkout feat/planned-scripts
```

Expected: branch is `feat/planned-scripts`, `programs/p4_cosmos_world_sim/` exists.

---

### Task B2: Create the p4_env conda environment

- [ ] **Step B2.1: Create env**

```bash
/home/zeus/miniconda3/bin/conda create -n p4_env python=3.10 -y
```

- [ ] **Step B2.2: Install PyTorch (cu128)**

```bash
/home/zeus/miniconda3/envs/p4_env/bin/pip install \
    torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128
```

- [ ] **Step B2.3: Verify GPU visible**

```bash
/home/zeus/miniconda3/envs/p4_env/bin/python -c \
    "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory // 1024**3, 'GB')"
```

Expected output: `NVIDIA A100 80GB PCIe 80 GB` (or HBM variant).

---

### Task B3: Clone Cosmos Predict 2 and install deps

- [ ] **Step B3.1: Clone cosmos-predict2**

```bash
git clone --depth=1 https://github.com/nvidia-cosmos/cosmos-predict2 /tmp/cosmos-predict2
```

If that URL fails, try:
```bash
git clone --depth=1 https://github.com/NVIDIA-Cosmos/cosmos-predict2 /tmp/cosmos-predict2
# or check: https://github.com/nvidia-cosmos
```

- [ ] **Step B3.2: Install cosmos-predict2**

```bash
cd /tmp/cosmos-predict2
/home/zeus/miniconda3/envs/p4_env/bin/pip install -e '.' --no-build-isolation
```

- [ ] **Step B3.3: Install LoRA + 8-bit optimizer deps**

```bash
/home/zeus/miniconda3/envs/p4_env/bin/pip install \
    peft bitsandbytes accelerate \
    huggingface_hub h5py imageio imageio-ffmpeg \
    numpy==1.26.4 einops
```

- [ ] **Step B3.4: Verify cosmos import**

```bash
/home/zeus/miniconda3/envs/p4_env/bin/python -c \
    "import cosmos_predict2; print('cosmos OK')"
```

If import name differs (check repo README), adjust import name accordingly.

---

### Task B4: Download Cosmos Predict 2.5 model weights

> This is a ~4-10GB download. Do it on CPU/before GPU starts. Weights persist in HF cache.

- [ ] **Step B4.1: Log in to HuggingFace**

```bash
/home/zeus/miniconda3/envs/p4_env/bin/huggingface-cli login \
    --token $HF_TOKEN
```

- [ ] **Step B4.2: Find the exact model ID**

```bash
# Check the cosmos-predict2 repo README for the exact HF model ID:
cat /tmp/cosmos-predict2/README.md | grep -i "huggingface\|hf.co\|model_id" | head -20
```

The model is expected to be one of:
- `nvidia/Cosmos-Predict-2-2B`
- `nvidia/Cosmos-Predict2-2B`
- `nvidia/Cosmos-Predict-2.5-2B`

- [ ] **Step B4.3: Download the model**

```bash
# Replace MODEL_ID with the ID found above
MODEL_ID="nvidia/Cosmos-Predict-2-2B"  # verify from README
/home/zeus/miniconda3/envs/p4_env/bin/huggingface-cli download $MODEL_ID \
    --local-dir ~/Humanoid/checkpoints/cosmos_base/ --quiet
```

Expected: ~4-10 GB download. Prints local path when done.

- [ ] **Step B4.4: Verify weights downloaded**

```bash
ls ~/Humanoid/checkpoints/cosmos_base/
# Expect: model files (*.safetensors or *.pt), config.json, etc.
```

---

### Task B5: Download P3 nav policy checkpoint (model_499.pt)

- [ ] **Step B5.1: Download from HuggingFace**

```bash
/home/zeus/miniconda3/envs/p4_env/bin/huggingface-cli download \
    mitvho09/humanoid-g1-nav \
    --include "checkpoints/p3_vision_nav/run_300_l4/model_499.pt" \
    --local-dir ~/Humanoid/ --quiet
```

- [ ] **Step B5.2: Verify**

```bash
ls ~/Humanoid/checkpoints/p3_vision_nav/run_300_l4/model_499.pt
# Expect: file exists, ~few MB (RSL-RL checkpoint)
```

---

### Task B6: Bring up Isaac Lab Docker (for data collection)

> Isaac Lab docker is needed for CP4.2 (data collection). A100 has no RT cores so camera rendering may be slow, but correctness is unaffected. Use small env count (64).

- [ ] **Step B6.1: Pull Isaac Lab Docker image**

```bash
docker pull nvcr.io/nvidia/isaac-lab/isaac-lab:2.0.0
# This is ~17.6 GB. Starts streaming immediately.
# Alternative: check PLANNED_SCRIPTS.md for exact image tag used in P3.
```

- [ ] **Step B6.2: Start the container**

```bash
cd ~/Humanoid
docker run -d --name isaac-lab-base --gpus all \
    --network host \
    -v ~/Humanoid:/workspace \
    -e PYTHONPATH="/workspace:/workspace/my-humanoid-project:/workspace/isaaclab/source" \
    nvcr.io/nvidia/isaac-lab/isaac-lab:2.0.0 sleep infinity
```

- [ ] **Step B6.3: Verify container is up**

```bash
docker exec isaac-lab-base python -c "import isaaclab; print('IsaacLab OK')"
```

---

### Task B7: Verify full environment

- [ ] **Step B7.1: Run the p4 import check**

```bash
cd ~/Humanoid
/home/zeus/miniconda3/envs/p4_env/bin/python -c "
import torch, h5py, imageio, peft, bitsandbytes
import sys; sys.path.insert(0, '.')
from programs.p4_cosmos_world_sim import cp41_inference, cp42_verify_data
print('All imports OK')
print('CUDA:', torch.cuda.get_device_name(0))
"
```

Expected: `All imports OK` + GPU name.

---

## PHASE C — GPU Execution (all billed time)

> Start here only after Phase B is fully complete. Each checkpoint has a verification step — do not skip.

### Task C1: CP4.1 — Stock Cosmos inference baseline

**DoD:** Stock (unmodified) Cosmos Predict 2.5 generates a plausible mp4 from the initial nav frame.

- [ ] **Step C1.1: Get an initial frame from P3 video**

```bash
cd ~/Humanoid
/home/zeus/miniconda3/envs/p4_env/bin/python -c "
import imageio, numpy as np
# Extract frame 0 from the P3 eval video
v = imageio.get_reader('programs/videos/p3_vision_nav/p3_vision_nav_model499.mp4')
frame = v.get_data(0)  # H x W x 3 uint8
np.save('/tmp/initial_frame.npy', frame)
print('Frame shape:', frame.shape)
v.close()
"
```

Expected: `Frame shape: (H, W, 3)` — some resolution from P3 eval video.

- [ ] **Step C1.2: Run CP4.1 inference**

```bash
cd ~/Humanoid
/home/zeus/miniconda3/envs/p4_env/bin/python -m programs.p4_cosmos_world_sim.cp41_inference \
    --model-dir ~/Humanoid/checkpoints/cosmos_base/ \
    --frame /tmp/initial_frame.npy \
    --out docs/results/cp41_inference.mp4 \
    --steps 16
```

Expected: `cp41_inference.mp4` saved. Open/verify it is a plausible video (not black, not garbage). **CP4.1 DONE.**

- [ ] **Step C1.3: Commit CP4.1 result**

```bash
cd ~/Humanoid
git add docs/results/cp41_inference.mp4
git commit -m "results: CP4.1 stock Cosmos inference baseline"
```

---

### Task C2: CP4.2 — Data collection (rollouts + format check)

**DoD:** HDF5 dataset created with (frame_t, action_t, frame_t+1) triplets; shapes printed and verified.

- [ ] **Step C2.1: Collect rollouts inside Isaac Lab Docker**

```bash
cd ~/Humanoid
docker exec \
  -e PYTHONPATH="/workspace:/workspace/my-humanoid-project:/workspace/isaaclab/source" \
  -e P3_CAM_H="64" -e P3_CAM_W="64" \
  isaac-lab-base python /workspace/programs/p4_cosmos_world_sim/collect_rollouts.py \
    --checkpoint /workspace/checkpoints/p3_vision_nav/run_300_l4/model_499.pt \
    --task Humanoid-G1-VisionNav-v0 \
    --num-envs 64 \
    --num-steps 500 \
    --out /workspace/datasets/g1_nav_cosmos.h5
```

Expected: progress prints every 50 steps. Final line: `Saved N triplets to datasets/g1_nav_cosmos.h5`.

This may take 10-20 min. A100 camera rendering is slower than L4 (no RT cores) but works.

- [ ] **Step C2.2: Verify data shapes**

```bash
cd ~/Humanoid
/home/zeus/miniconda3/envs/p4_env/bin/python -m programs.p4_cosmos_world_sim.cp42_verify_data \
    --data datasets/g1_nav_cosmos.h5
```

Expected output:
```
Dataset: datasets/g1_nav_cosmos.h5
  frame_t:   (N, 64, 64, 3)   uint8
  action_t:  (N, 29)          float32
  frame_t1:  (N, 64, 64, 3)   uint8
N >= 10000 triplets
Sample frame range: [0, 255]   ✓
Sample action range: [-π, π]   reasonable ✓
CP4.2 DONE
```

If N < 5000, re-run with more envs or steps. **CP4.2 DONE.**

---

### Task C3: CP4.3 — LoRA post-train (SMOKE GATE — must pass before full run)

**DoD (smoke):** 2 training steps complete without OOM or crash; loss printed. Two diff actions → two diff predicted frame distributions (visual check).

- [ ] **Step C3.1: SMOKE TEST (2 steps)**

```bash
cd ~/Humanoid
/home/zeus/miniconda3/envs/p4_env/bin/python -m programs.p4_cosmos_world_sim.cp43_train \
    --data datasets/g1_nav_cosmos.h5 \
    --model-dir ~/Humanoid/checkpoints/cosmos_base/ \
    --lora-rank 16 \
    --smoke \
    --max-steps 2 \
    --out /tmp/cosmos_smoke/
```

Expected:
```
Step 1/2  loss=X.XXXX
Step 2/2  loss=X.XXXX
Smoke test PASSED — VRAM used: XX GB / 80 GB
```

- [ ] **Step C3.2: Gate check — STOP HERE if smoke fails**

If OOM: reduce batch size or lora-rank in the command. If loss is NaN: check bf16 stability.
```bash
# If OOM at lora-rank=16, try lora-rank=8:
/home/zeus/miniconda3/envs/p4_env/bin/python -m programs.p4_cosmos_world_sim.cp43_train \
    --data datasets/g1_nav_cosmos.h5 \
    --model-dir ~/Humanoid/checkpoints/cosmos_base/ \
    --lora-rank 8 --smoke --max-steps 2 --out /tmp/cosmos_smoke/
```

**Do NOT proceed to the full training run until smoke passes.**

- [ ] **Step C3.3: Full training run (5000 steps, ~60-90 GPU-hr)**

```bash
cd ~/Humanoid
nohup /home/zeus/miniconda3/envs/p4_env/bin/python \
    -m programs.p4_cosmos_world_sim.cp43_train \
    --data datasets/g1_nav_cosmos.h5 \
    --model-dir ~/Humanoid/checkpoints/cosmos_base/ \
    --lora-rank 16 \
    --max-steps 5000 \
    --save-every 500 \
    --out checkpoints/p4_cosmos_lora/ \
    > p4_train.log 2>&1 &
echo "PID: $!"
```

- [ ] **Step C3.4: Monitor training**

```bash
tail -f p4_train.log
# Expected: loss decreasing from ~1.5 to <0.3 over 5000 steps
# Check every 500 steps — if loss is stuck for 1000 steps, stop and adjust LR
```

- [ ] **Step C3.5: Verify checkpoint saved**

```bash
ls checkpoints/p4_cosmos_lora/
# Expect: checkpoint_5000.pt (or similar), adapter_config.json
```

- [ ] **Step C3.6: Visual diff-actions test**

```bash
/home/zeus/miniconda3/envs/p4_env/bin/python -m programs.p4_cosmos_world_sim.cp43_train \
    --eval-only \
    --checkpoint checkpoints/p4_cosmos_lora/ \
    --out docs/results/cp43_action_diff.mp4
```

Expected: Two side-by-side videos, visually different futures for action A vs action B. **CP4.3 DONE.**

---

### Task C4: CP4.4 — K-step action-conditioned rollout

**DoD:** 8-step imagined rollout; fidelity (SSIM or pixel MSE) vs real env reported.

- [ ] **Step C4.1: Run K-step rollout**

```bash
cd ~/Humanoid
/home/zeus/miniconda3/envs/p4_env/bin/python -m programs.p4_cosmos_world_sim.cp44_rollout \
    --checkpoint checkpoints/p4_cosmos_lora/ \
    --data datasets/g1_nav_cosmos.h5 \
    --k-steps 8 \
    --out docs/results/cp44_rollout.mp4
```

Expected output:
```
K-step rollout fidelity (vs real frames):
  SSIM:  0.XX (higher is better, 1.0 = perfect)
  PSNR:  XX dB
  MSE:   X.XXX
Rollout video saved: docs/results/cp44_rollout.mp4
CP4.4 DONE
```

- [ ] **Step C4.2: Commit result**

```bash
git add docs/results/cp44_rollout.mp4
git commit -m "results: CP4.4 K-step rollout fidelity measured"
```

---

### Task C5: CP4.5 — CEM planning demo

**DoD:** CEM planner reaches goal in real env using only Cosmos model-predicted lookahead (no ground truth state at planning time).

- [ ] **Step C5.1: Run CEM planning**

```bash
cd ~/Humanoid
docker exec \
  -e PYTHONPATH="/workspace:/workspace/my-humanoid-project:/workspace/isaaclab/source" \
  isaac-lab-base python /workspace/programs/p4_cosmos_world_sim/cp45_plan.py \
    --cosmos-checkpoint /workspace/checkpoints/p4_cosmos_lora/ \
    --task Humanoid-G1-VisionNav-v0 \
    --num-envs 4 \
    --plan-steps 8 \
    --cem-samples 64 \
    --out /workspace/docs/results/cp45_planning.mp4
```

Expected output:
```
Episode 1: goal reached in 47 steps (planned)
Episode 2: goal reached in 52 steps (planned)
...
Success rate: X/10 = XX%
Planning video saved: docs/results/cp45_planning.mp4
CP4.5 DONE
```

---

### Task C6: Upload to HuggingFace and document results

- [ ] **Step C6.1: Upload LoRA checkpoint to HuggingFace**

```bash
cd ~/Humanoid
/home/zeus/miniconda3/envs/p4_env/bin/python -c "
from huggingface_hub import HfApi
api = HfApi(token='$HF_TOKEN')
api.upload_folder(
    folder_path='checkpoints/p4_cosmos_lora/',
    repo_id='mitvho09/humanoid-g1-nav',
    path_in_repo='checkpoints/p4_cosmos_lora/',
    repo_type='dataset'
)
print('Uploaded checkpoint')

# Upload videos
for fname in ['cp41_inference.mp4','cp43_action_diff.mp4','cp44_rollout.mp4','cp45_planning.mp4']:
    api.upload_file(
        path_or_fileobj=f'docs/results/{fname}',
        path_in_repo=f'videos/p4/{fname}',
        repo_id='mitvho09/humanoid-g1-nav',
        repo_type='dataset'
    )
    print(f'Uploaded {fname}')
"
```

- [ ] **Step C6.2: Write results doc**

```bash
cat > ~/Humanoid/docs/results/p4_cosmos_world_sim.md << 'EOF'
# P4 — Cosmos Predict World Simulator Results

**Status:** Complete  
**Date:** $(date +%Y-%m-%d)

## Checkpoints

| CP | DoD | Result |
|---|---|---|
| CP4.1 | Stock Cosmos generates mp4 | ✅ — [cp41_inference.mp4] |
| CP4.2 | Dataloader prints shapes | ✅ — N=XXXX triplets, (64,64,3) |
| CP4.3 | Loss decreases; diff actions → diff futures | ✅ — loss X→X |
| CP4.4 | K-step rollout fidelity reported | ✅ — SSIM=X.XX |
| CP4.5 | Planner reaches goal | ✅ — XX% success |

## Training Config
- Base model: Cosmos Predict 2.5 2B
- LoRA rank: 16, bf16, gradient checkpointing, 8-bit Adam
- Steps: 5000, data: XXXX triplets from P3 nav rollouts (64×64 G1 cam)

## Key Result
[Fill in key takeaway — e.g. "action-conditioned Cosmos distinguishes forward vs backward commands with SSIM X.XX vs X.XX"]
EOF
```

- [ ] **Step C6.3: Final commit and push**

```bash
cd ~/Humanoid
git add docs/results/p4_cosmos_world_sim.md docs/results/
git commit -m "results: P4 complete — Cosmos LoRA world model trained on G1 nav"
git push origin feat/planned-scripts
```

---

## Graceful Degradation Reference

If the full post-train (CP4.3 full) cannot complete:

| Situation | Fallback |
|---|---|
| Smoke OOM at lora-rank=8 | Inference-only: skip CP4.3 real, document as future work; still do CP4.1 + CP4.5 on stock model |
| CP4.3 loss stuck (doesn't decrease) | Check LR, try `--lr 5e-5`; document result anyway |
| CP4.5 planning fails | Report stock Cosmos planning result (weaker but still valid) |
| Machine time runs out during training | Save checkpoint at latest step, resume with `--resume` flag |

See `docs/vault/tasks/P4 - Cosmos Predict.md § Graceful Degradation` for full policy.

---

## Time / Cost Estimate

| Phase | Time | GPU Cost |
|---|---|---|
| Phase B (setup) | 30-40 min | $0 (CPU) |
| CP4.1 inference | 5-10 min | ~$0.5 |
| CP4.2 data collection | 20-30 min | ~$1-2 |
| CP4.3 smoke | 2 min | ~$0.1 |
| CP4.3 full train | 60-90 GPU-hr | ~$150-250 |
| CP4.4 rollout eval | 10 min | ~$0.5 |
| CP4.5 planning | 30-60 min | ~$2-5 |
| **Total** | **~4-5 days** | **~$155-260** |

A100-80G on-demand: ~$2.5-3.5/hr. Reserve the machine for CP4.3 full train block.
