---
tags: [task, p2, world-model, dreamer, rssm, pytorch]
---

# P2 — World Model (Dreamer-mini)

## Summary

A world model lets the agent imagine future states without running the real simulator — useful for model-based planning, data augmentation, and latent-space training. This project builds a minimal Dreamer-style RSSM from scratch.

**Status: COMPLETE ✅** CPU-verified toy training + real Isaac rollouts trained. Final: loss 0.76 → 0.011, imagined reward finite (DoD met).

[Full result doc](../../results/p2_world_model.md)

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
  rssm.py                    ← RSSM, WorldModel classes
  agent.py                   ← Actor, Critic, imagine_returns()
  train_wm.py                ← toy point-mass trainer (CPU, verifies code)
  collect_nav_rollouts.py    ← collect (obs,action,reward) from nav policy → .pt
  train_wm_isaac.py          ← train WorldModel on real Isaac rollouts
  tests/test_world_model.py  ← unit tests (overfit tiny batch)
  tests/test_wm_isaac.py     ← smoke tests for Isaac pipeline (5, all green)
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

## Running on Real Isaac Rollouts (GPU)

### Step 1 — Collect rollouts
```bash
docker exec -e PYTHONPATH=/workspace:/workspace/my-humanoid-project:/workspace/isaaclab/source \
  isaac-lab-base /workspace/isaaclab/isaaclab.sh -p \
  /workspace/programs/world_model/collect_nav_rollouts.py \
  --task Humanoid-G1-CommandNav-v0 \
  --checkpoint /workspace/programs/checkpoints/g1_commandnav/model_499.pt \
  --num-envs 64 --num-episodes 200 \
  --out /workspace/programs/data/nav_rollouts_commandnav.pt
```

### Step 2 — Train world model
```bash
python -m programs.world_model.train_wm_isaac \
  --data programs/data/nav_rollouts_commandnav.pt \
  --steps 2000 \
  --out programs/checkpoints/world_model/wm_commandnav.pt
```

### Step 3 — Verify DoD
Output should show `imagined_mean_reward` is finite (not NaN). This is the P2 DoD.

---

## Remaining GPU Tasks

1. ~~**Collect rollouts**~~ (script written, CPU-tested — needs GPU run)
2. ~~**Train Dreamer-mini on real rollouts**~~ (script written, CPU-tested — needs GPU run)
3. **Run the pipeline** — 200 episodes → 2000 training steps → imagined_reward finite
4. Generate `docs/results/p2_world_model.md`

VRAM: world model training is pure PyTorch, no Isaac Sim → 8–16 GB is plenty (even T4 or laptop GPU). See [GPU VRAM table](../../docs/GPU_VRAM_REQUIREMENTS.md).

---

## Related

- [[World Models (Dreamer-mini)]]
- [[Training Recipe]]
- [programs/world_model/rssm.py](../../programs/world_model/rssm.py)
- [docs/PHYSICAL_AI_6MONTH_PLAN.md](../../docs/PHYSICAL_AI_6MONTH_PLAN.md)
