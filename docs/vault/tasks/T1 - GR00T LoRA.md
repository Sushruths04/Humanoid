---
tags: [task, t1, manipulation, groot, lora, libero, vla, t-track]
---

# T1 — GR00T N1.7 LoRA Language Manipulation (T-Track)

## Summary

Plug NVIDIA's GR00T N1.7-3B Vision-Language-Action model into the LIBERO tabletop manipulation benchmark. Evaluate the pre-trained LIBERO_PANDA checkpoint, then optionally fine-tune with LoRA on LIBERO Spatial demos.

**Status: COMPLETE ✅ — 97.0% mean task success across all 10 LIBERO Spatial tasks (20 eps each)**

Matches NVIDIA paper result (~97.7%). +47 percentage points over T0 BC baseline (50%).

---

## Checkpoints

| ID | Deliverable | Target | Status |
|---|---|---|---|
| CPT1.1 | GR00T policy wrapper (`groot_policy.py`) | loads + infers | ✅ Written |
| CPT1.2 | Eval harness wired (`evaluate_groot_libero.py`) | runs 10 tasks | ✅ Written |
| CPT1.3 | Pre-trained checkpoint eval | task_success **97.0%** | ✅ Done |
| CPT1.4 | LoRA fine-tuning script | trains | ✅ Written (`run_finetune.sh`) |
| CPT1.5 | Fine-tuned checkpoint eval | task_success ≥97% | 🔜 Needs 40 GB GPU |

---

## Architecture

| Component | Detail |
|---|---|
| Base model | `nvidia/GR00T-N1.7-3B` — 3B param VLA |
| LIBERO checkpoint | `nvidia/GR00T-N1.7-LIBERO` (`libero_spatial` sub-folder) |
| Embodiment tag | `LIBERO_PANDA` |
| Obs — state | `eef_xyz(3) + euler_rpy(3) + gripper_qpos(2)` = 8 dims, shape `(B=1,T=1,8)` |
| Obs — video | `image` + `wrist_image`, shape `(B=1,T=1,H,W,3)` uint8 |
| Obs — language | `human.action.task_description: [["task text"]]` |
| Action | OSC_POSE delta (7-dim), action chunk size = 8 |
| Fine-tune method | LoRA via `gr00t.experiment.runner finetune` |
| Fine-tune dataset | `IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot` |

---

## What Was Built

```
programs/t1_groot_lora/
├── groot_policy.py            # GR00T policy wrapper (Gr00tPolicy adapter)
│   ├── load_groot_model()     # load once, share across tasks
│   ├── make_policy_fn()       # per-task closure (task language)
│   └── build_groot_policy()   # single-task convenience
├── evaluate_groot_libero.py   # evaluation harness (obs→action loop)
├── setup_t1.sh                # full env + checkpoint + dataset setup
├── run_finetune.sh            # LoRA fine-tuning launcher
├── run_eval.sh                # client-server eval (official GR00T protocol)
└── tests/test_groot_smoke.py  # 5 smoke tests (data pipeline, imports)
```

---

## Key Implementation Details

### Observation Format
LIBERO env provides quaternion (`robot0_eef_quat` = `[w, x, y, z]`). GR00T expects Euler angles. Conversion via scipy:
```python
xyzw = [quat[1], quat[2], quat[3], quat[0]]   # [w,x,y,z] → [x,y,z,w]
euler = Rotation.from_quat(xyzw).as_euler("xyz")
state = concat([eef_pos, euler, gripper_qpos])  # (8,)
```

### Multi-Task Model Reuse
GR00T is 3B params — reloading per task is expensive. The harness loads once:
```python
model = load_groot_model(checkpoint_path, ...)
for task in tasks:
    policy_fn, _ = make_policy_fn(model, task_language)  # new closure, same weights
```

### EmbodimentTag Access (gotcha)
```python
# WRONG — uses enum value, raises ValueError
EmbodimentTag("LIBERO_PANDA")
# CORRECT — looks up by name
EmbodimentTag["LIBERO_PANDA"]
```

### Module Path (gotcha)
```python
# WRONG
from gr00t.model.policy import Gr00tPolicy
# CORRECT
from gr00t.policy.gr00t_policy import Gr00tPolicy
```

---

## Setup (Lightning Studio)

```bash
# 1. Install groot_env (Python 3.10, PyTorch 2.7+cu128)
bash programs/t1_groot_lora/setup_t1.sh

# 2. Download pre-trained LIBERO checkpoint
huggingface-cli download nvidia/GR00T-N1.7-LIBERO \
    --local-dir programs/checkpoints/groot_n17/libero_spatial \
    --include "libero_spatial/**"

# 3. Copy modality.json for fine-tuning
cp /tmp/Isaac-GR00T/examples/LIBERO/modality.json \
    programs/t1_groot_lora/datasets/libero_spatial_no_noops/meta/
```

---

## Run Eval (needs ≥16 GB VRAM)

```bash
MUJOCO_GL=egl /home/zeus/miniconda3/envs/groot_env/bin/python \
    -m programs.t1_groot_lora.evaluate_groot_libero \
    --checkpoint programs/checkpoints/groot_n17/libero_spatial/libero_spatial \
    --task libero_spatial \
    --num-envs 20 \
    --out docs/results/t1_groot.md
```

---

## Run LoRA Fine-tuning (needs ≥40 GB VRAM)

```bash
# Single GPU (L40S, A100-80G)
bash programs/t1_groot_lora/run_finetune.sh

# 8× GPU for benchmark replication
NUM_GPUS=8 GLOBAL_BATCH_SIZE=640 MAX_STEPS=20000 \
    bash programs/t1_groot_lora/run_finetune.sh
```

---

## VRAM Requirements

| Operation | Required | Notes |
|---|---|---|
| Inference (eval) | ≥16 GB | T4 (15 GB) may OOM; use `--denoising-steps 4` to try |
| LoRA fine-tuning | ≥40 GB | L40S or A100-80G |
| Full fine-tune | ≥80 GB | A100-80G × 8 |

---

## Related

- [[T0 - ManipFoundation]] — LIBERO env setup, BC baseline (T1 builds on it)
- [[Frozen Text Encoder for Language Tasks]] — GR00T uses much larger VLM backbone
- [docs/results/t1_groot.md](../../results/t1_groot.md) — generated after eval run
- [isaac-gr00t repo](https://github.com/NVIDIA/Isaac-GR00T) — model source
