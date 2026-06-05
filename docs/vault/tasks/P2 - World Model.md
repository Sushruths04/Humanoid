---
tags: [task, p2, world-model, dreamer, rssm, pytorch]
---

# P2 — World Model (Dreamer-mini)

## Summary

A world model lets the agent imagine future states without running the real simulator — useful for model-based planning, data augmentation, and latent-space training. This project builds a minimal Dreamer-style RSSM from scratch.

**Status:** fully built and CPU-verified. Toy point-mass training: loss 2.7 → 0.11. **Not yet trained on real Isaac rollouts** (remaining GPU task).

---

## Architecture

See [[World Models (Dreamer-mini)]] for the full concept explanation.

| Component | Dims |
|---|---|
| GRU deterministic state | 64 |
| Stochastic latent (z) | 16 |
| MLP hidden | 64 |

```python
from programs.world_model.rssm import RSSM, WorldModel
wm = WorldModel(obs_dim=16, action_dim=4, deter=64, stoch=16, hidden=64)
# wm.observe(obs, actions) → sequence of posterior states
# wm.loss(obs, actions, rewards) → (recon_loss, kl_loss)
# wm.imagine(policy, initial_state, horizon) → imagined rollout
```

---

## Files

```
programs/world_model/
  rssm.py          ← RSSM, WorldModel classes
  agent.py         ← Actor, Critic, imagine_returns()
  train_wm.py      ← toy point-mass trainer (CPU, verifies code)
  tests/test_world_model.py ← unit tests (overfit tiny batch)
```

---

## Running the Toy Training

```bash
cd /teamspace/studios/this_studio/Humanoid
/home/zeus/miniconda3/envs/cloudspace/bin/python programs/world_model/train_wm.py
# Loss should go from ~2.7 to ~0.11 in 1000 iters
# No GPU needed — pure PyTorch
```

---

## Remaining GPU Tasks

1. **Collect rollouts** from the trained CommandNav/ObstacleNav policy (save obs, actions, rewards to disk)
2. **Train Dreamer-mini** on real nav rollouts
3. **Show imagination-trained agent > random** (the P2 DoD)
4. Generate `docs/results/p2_world_model.md`

VRAM: world model training is pure PyTorch, no Isaac Sim → 8–16 GB is plenty (even T4 or laptop GPU). See [GPU VRAM table](../../docs/GPU_VRAM_REQUIREMENTS.md).

---

## Related

- [[World Models (Dreamer-mini)]]
- [[Training Recipe]]
- [programs/world_model/rssm.py](../../programs/world_model/rssm.py)
- [docs/PHYSICAL_AI_6MONTH_PLAN.md](../../docs/PHYSICAL_AI_6MONTH_PLAN.md)
