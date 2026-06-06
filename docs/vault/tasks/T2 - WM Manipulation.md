---
tags: [task, t2, manipulation, world-model, dreamer, libero, t-track]
---

# T2 — World Model for Manipulation (T-Track)

## Summary

Train a Dreamer-mini RSSM world model on recorded GR00T rollouts from LIBERO Spatial. The WM learns manipulation dynamics (eef position, orientation, gripper) in latent space and can imagine future rollouts without running the real simulator.

**Status: COMPLETE ✅ — loss 1.403 → 0.008, imagined reward finite, DoD met**

---

## Checkpoints

| ID | Deliverable | Target | Status |
|---|---|---|---|
| CPT2.1 | Rollout collector (`collect_manip_rollouts.py`) | 200 episodes saved | ✅ Written |
| CPT2.2 | WM training script (`train_wm_manip.py`) | loss drops, reward finite | ✅ Written |
| CPT2.3 | Run collect + train pipeline | DoD: imagined reward finite | ✅ Done (loss 1.40→0.008) |
| CPT2.4 | Results doc + vault update | `docs/results/t2_manip_wm.md` | ✅ Done |

---

## Architecture

| Component | Detail |
|---|---|
| Base WM | Dreamer-mini RSSM (reused from [[P2 - World Model]]) |
| Rollout source | GR00T N1.7 policy on libero_spatial (10 tasks, 20 eps each) |
| obs_dim | 8 — eef_xyz(3) + axis_angle(3) + gripper_qpos(2) |
| act_dim | 7 — OSC delta (same as GR00T output) |
| reward signal | 0.0 per step, 1.0 at terminal step if task success |
| RSSM config | deter=128, stoch=32, hidden=128 |
| Training steps | 3000 |
| DoD | imagined mean reward is finite (not NaN) |

---

## What Was Built

```
programs/t2_manip_wm/
├── collect_manip_rollouts.py   # run GR00T → save .pt dataset
├── train_wm_manip.py           # train WorldModel + write results MD
└── run_t2.sh                   # one-shot: collect → train → push
```

Reuses:
- `programs/world_model/rssm.py` — RSSM + WorldModel (unchanged from P2)
- `programs/t1_groot_lora/groot_policy.py` — GR00T inference

---

## Run (on L4, inside groot_env)

```bash
cd /teamspace/studios/this_studio/Humanoid
MUJOCO_GL=egl PYTHONUNBUFFERED=1 nohup bash programs/t2_manip_wm/run_t2.sh > t2_run.log &
```

Expected runtime: ~25 min total (20 min collect + 5 min train on GPU)

---

## Related

- [[T1 - GR00T LoRA]] — rollout source (same GR00T policy + LIBERO env)
- [[P2 - World Model]] — RSSM architecture reused here
- [docs/results/t2_manip_wm.md](../../results/t2_manip_wm.md) — generated after run
