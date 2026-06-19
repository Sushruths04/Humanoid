---
tags: [moc, index, wakeboarding]
---

# 🏄 Wakeboarding RL — Vault Index (LIVE)

> The single hub for this project. Status badges below are updated by whoever runs experiments (see [[DOC_PROTOCOL]] in the project root).

## 📍 Status (update me)
- **Stage:** 🟢 **Full codebase written** (pure-PyTorch core CPU-verified; Isaac-Lab parts untested on GPU) — ready for a GPU smoke pass
- **Compute:** Modal (main, L40S) / Lightning AI (fallback) — see project README & PLAN §8
- **Best checkpoint:** _none yet (no GPU run)_
- **Best success @30 km/h:** _TBD_
- **Next action:** follow `../PRE_GPU_CHECKLIST.md` on a Lightning box (G1 joint/link names already pre-filled → fewer markers to fix), then `10_train_stage1.sh`
- **Pre-GPU prep done:** G1 names resolved from IsaacLab; Day-1 checklist written; Docker reuses existing `humanoid-isaaclab` image
- **Last updated:** 2026-06-19

## 🗺️ Map of Content

### Concept (theory)
- [[Wakeboard_Start_Biomechanics]] — what a correct start *is* (the 5 rules)
- [[RL_Method_HumanUP_AMP]] — two-stage discovery→deployable + AMP style
- [[Environment_and_Rope_Model]] — G1 + board + sand + the 30 km/h rope
- [[Reward_Design]] — every reward/penalty term and why

### Implementation (kept current)
- [[Scripts]] — every script: what it does + how to run *(live)*

### Results (kept current)
- [[Results_Live]] — success/fall/time tables, ablations, Stage I vs II *(live)*

### Log
- [[Experiment_Log]] — dated record of every run *(append-only)*

## 🎯 The goal in one line
Teach a G1 humanoid the biomechanics of a wakeboard deep-water start (crouch → stable ride under a 30 km/h pull), with results polished enough to show recruiters.

## 💻 Compute
**Modal vs Lightning decision rule + Docker machine-switching:** `../COMPUTE_GUIDE.md`.
- Quick rule: *short + parallel + compute-only → **Modal**. long + interactive + rendering → **Lightning**.*
- First GPU smoke / fixing `# VERIFY` markers → **Lightning** (interactive). Then push image, run on Modal.
- One image (`ghcr.io/sushruths04/wakeboard-isaaclab`) for both → `./docker/run.sh pull` to switch machines.

## 🔗 Plan
Full executor plan: `../PLAN.md` (sections referenced throughout these notes).
