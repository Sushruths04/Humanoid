---
tags: [task, p4, cosmos, world-model, cosmos-predict, lora, planning, future]
---

# P4 — Cosmos as a Controllable World Simulator

**Status**: 🔜 Not Started — needs A100-80G  
**DoD**: Action-conditioned Cosmos rollout + planner reaches goal in real env using only model-predicted lookahead  
**GPU**: A100-80G (80 GB VRAM) — the **only** 80 GB need in the entire program  
**Estimated cost**: 80–150 GPU-hr (~$150–$300 on-demand)

---

## What It Does

Post-train **Cosmos Predict 2.5 (2B)** with LoRA on G1 navigation (frames, actions). The trained model becomes an action-conditioned world simulator: given a current frame and a planned action sequence, it predicts future frames. Use this for:
1. **Planning (CEM/MPC)**: sample K action sequences → score predicted outcomes → execute best
2. **Policy eval**: score P1/P3 policies via predicted rollouts (correlates with real success rate)

---

## Architecture

```
G1 nav rollouts (frames + actions)
         ↓
  Cookbook action-conditioned format
  (frame_t, action_t, frame_{t+1}) batches
         ↓
  LoRA post-train
  Cosmos Predict 2.5 (2B, bf16, grad ckpt, 8-bit optimizer)
         ↓
  Action-conditioned world model
         ↓
  ┌─ Rollout predictor: K-step imagined futures
  └─ Planner (CEM/MPC): pick action sequence → best predicted outcome
```

---

## Checkpoints

| ID | Deliverable | Target | Status |
|---|---|---|---|
| CP4.1 | Inference baseline | stock Cosmos Predict 2.5 generates mp4 from initial frame | 🔜 |
| CP4.2 | Data prep | dataloader yields (frame_t, action_t, frame_{t+1}); shapes printed | 🔜 |
| CP4.3 | LoRA post-train | training loss decreases; two diff actions → two diff predicted futures | 🔜 |
| CP4.4 | K-step rollout / distillation | K-step action-conditioned rollout; fidelity vs real env reported | 🔜 |
| CP4.5 | Planning + policy eval | planner reaches goal in real env; policy-eval correlates with real eval | 🔜 |

**Hard gate:** Run CP4.3 smoke (tiny subset, 2 iters) before committing GPU-hours. If 2B LoRA won't fit, fall back to inference-only + planning on stock Cosmos.

---

## VRAM Requirements

| Operation | Required | Notes |
|---|---|---|
| Stock inference (CP4.1) | ≥24 GB | L4 might work for inference only |
| LoRA post-train (CP4.3) | **≥40 GB comfortable, 80 GB safe** | L4 cannot do this |
| Distillation (CP4.4) | ≥40 GB | |

**Why A100-80G:** 2B LoRA with gradient checkpointing needs ~40–50 GB. The L4 (24 GB) cannot post-train — inference only if using L4. This is the only task in the entire P/T program that requires a proper 80 GB card.

---

## Efficiency Rules (apply every run)

1. **LoRA only** — never full fine-tune on one GPU
2. **bf16** + **gradient checkpointing** + **8-bit optimizer** (bitsandbytes)
3. **Smoke first**: tiny subset, 2 iters before real post-train
4. Short planning horizons: 8–16 steps; cache generations
5. CP4.3 is the budget-critical run — smoke gate hard before committing

---

## Setup Commands (to fill in on machine)

```bash
# 1. Install Cosmos Predict 2.5 (cosmos-cookbook)
# git clone https://github.com/nvidia-cosmos/cosmos-predict2.5
# Follow Robot/Policy post-training recipe in cosmos-cookbook

# 2. Data prep — export G1 nav rollouts
# python programs/p4_cosmos_world_sim/export_rollouts.py \
#   --checkpoint <p3_model_499.pt> \
#   --num-envs 512 --num-steps 1000 \
#   --out data/g1_nav_cosmos/

# 3. CP4.3 smoke test (before real post-train)
# python programs/p4_cosmos_world_sim/train_cosmos_lora.py \
#   --data data/g1_nav_cosmos/ \
#   --smoke --max-steps 2

# 4. Real post-train
# python programs/p4_cosmos_world_sim/train_cosmos_lora.py \
#   --data data/g1_nav_cosmos/ \
#   --lora-rank 16 --max-steps 5000 \
#   --out checkpoints/p4_cosmos/
```

---

## Deliverables

- `programs/p4_cosmos_world_sim/` — all scripts
- LoRA-post-trained Cosmos checkpoint on HF (`mitvho09/humanoid-g1-nav`)
- "Imagine the rollout" prediction videos (two actions → two different futures)
- Planning demo: planner reaches goal using only model-predicted lookahead
- `docs/results/p4_cosmos_world_sim.md`

---

## Graceful Degradation

If the post-train won't fit/finish, present:
- **Stock Cosmos inference** on nav frames (no action conditioning)
- **Planning on stock Cosmos** (limited but still shows the WFM interface)
- Document LoRA post-train as future work
This is still a strong portfolio result (shows WFM familiarity).

---

## Related

- [[P2 - World Model]] — Dreamer-mini WM (conceptual foundation for understanding this)
- [[P3 - VisionNav]] — the nav policy whose rollouts feed P4 data
- [[C5 - Loco-Manipulation Capstone]] — P4 world model is used here for lookahead
- [GPU VRAM Requirements](../../GPU_VRAM_REQUIREMENTS.md)
- [Master Roadmap §4](../../MASTER_ROADMAP_CONVERGED.md)
