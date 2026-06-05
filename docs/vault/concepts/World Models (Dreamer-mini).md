---
tags: [concepts, world-model, dreamer, rssm, p2]
---

# World Models (Dreamer-mini)

## What Is a World Model?

A world model learns to predict what happens next: given the current state and an action, what will the next state and reward be? Once you have a world model, you can **imagine** trajectories without running the real environment — useful for planning, data augmentation, and learning from fewer real interactions.

This project implements a minimal Dreamer-style world model from scratch (no external Dreamer library dependency).

---

## The Architecture (RSSM)

The Recurrent State-Space Model (RSSM) combines:
- **Deterministic (recurrent) state** `h_t` — a GRU hidden state; captures temporal context
- **Stochastic state** `z_t` — a sampled latent; captures uncertainty about the world

```
h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])   # recurrent update
z_t ~ Normal(mu(h_t), sigma(h_t))          # sample stochastic state
y_t_hat = decode(h_t, z_t)                 # predict observation
r_t_hat = reward_head(h_t, z_t)            # predict reward
```

**Dimensions (Dreamer-mini, tuned for CPU feasibility):**
- `deter` (GRU hidden) = 64
- `stoch` (latent dim) = 16
- `hidden` (MLP hidden) = 64

Learned loss = reconstruction + KL divergence on the stochastic state.

---

## Implementation in This Repo

```
programs/world_model/
  rssm.py        ← RSSM class: observe(), loss(), imagine()
  agent.py       ← Actor, Critic, imagine_returns()
  train_wm.py    ← toy point-mass trainer (verifies code without Isaac Sim)
  tests/test_world_model.py ← CPU unit tests (overfit tiny batch)
```

**`WorldModel.observe(obs, actions)`** — runs the RSSM over a sequence, returns states  
**`WorldModel.loss(obs, actions, rewards)`** — reconstruction + KL loss  
**`WorldModel.imagine(policy, initial_state, horizon)`** — unrolls imagined trajectories

**Verified:** toy point-mass training loss goes 2.7 → 0.11 in 1000 iterations. CPU-tested.

---

## Current Status and What's Left

✅ Architecture built and CPU-verified  
✅ All tests pass  
❌ Not yet trained on real Isaac Lab nav rollouts  

**Remaining (P2 GPU task):**
1. Collect rollouts from the trained CommandNav/ObstacleNav policy
2. Train the world model on those rollouts
3. Show imagination-trained agent > random (the P2 DoD)

---

## Related

- [[P2 - World Model]]
- [[PPO with RSL-RL]]
- [programs/world_model/rssm.py](../../programs/world_model/rssm.py)
- [programs/world_model/train_wm.py](../../programs/world_model/train_wm.py)
- [docs/PHYSICAL_AI_6MONTH_PLAN.md](../../docs/PHYSICAL_AI_6MONTH_PLAN.md)
