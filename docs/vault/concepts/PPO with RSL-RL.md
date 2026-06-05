---
tags: [concepts, ppo, rsl-rl, training, algorithm]
---

# PPO with RSL-RL

## What RSL-RL Is

RSL-RL is a fast, vectorized PPO implementation built for Isaac Lab. It handles:
- Collecting rollouts across thousands of parallel envs
- Computing advantages (GAE)
- Updating the policy + value network
- Checkpointing and logging to TensorBoard

All nav tasks in this project use `G1FlatPPORunnerCfg` — the pre-tuned PPO config for the G1 flat task. You don't tune PPO hyperparameters; you tune the **env config** (rewards, resets, observations).

---

## Training Command

```bash
docker exec \
  -e PYTHONPATH="/workspace:/workspace/my-humanoid-project:/workspace/isaaclab/source" \
  isaac-lab-base \
  /workspace/isaaclab/isaaclab.sh -p /workspace/my-humanoid-project/custom_train.py \
  --task Humanoid-G1-CommandNav-v0 \
  --headless \
  --num_envs 4096 \
  --max_iterations 500
```

---

## Training Parameters (all tasks)

| Parameter | Value | Why |
|---|---|---|
| num_envs | 4096 | More envs = more diverse experience per update; fits on L4 at ~4.6 GB peak |
| max_iterations | 500 | Empirically sufficient; CommandNav converges ~iter 200, reward stabilizes 300–500 |
| Iteration time | ~2.4 s/iter | Measured on L4 |
| Total training time | ~20 min | 500 × 2.4 s |
| Checkpoint frequency | every 50 iters | `model_*.pt` files in log dir |

---

## What to Monitor (reading the training log)

Every iteration prints:
```
Learning iteration 221/500
  Mean reward: 4.17                          ← TOTAL reward (all terms combined)
  Episode_Reward/nav_command: 1.93           ← navigation term specifically
  Episode_Reward/track_lin_vel_xy_exp: 0.85  ← base locomotion tracking
  Episode_Termination/time_out: 0.977        ← fraction of eps ending by timeout
  Episode_Termination/base_contact: 0.024    ← fraction ending by fall
  Iteration time: 2.38s
  ETA: 00:16:33
```

**Key diagnostics:**
- `Mean reward` goes from ≈ −5 to positive around iter 150–250 for a healthy nav task
- Your task-specific term (e.g. `nav_command`) should climb alongside total reward
- If `nav_command ≈ 0` while `track_lin_vel_xy_exp ≈ 0.85`: the policy is farming base locomotion, not navigating → see [[Reward Shaping & Progress Rewards]] and [[SeqNav Stand-Still Local Optimum]]
- High `base_contact` = high fall rate; not a blocker for nav tasks but worth fixing (P0 follow-up)

---

## Checkpoints

```
/workspace/isaaclab/logs/rsl_rl/g1_flat/<YYYY-MM-DD_HH-MM-SS>/
  model_0.pt
  model_50.pt
  ...
  model_499.pt    ← final checkpoint (always use this for eval)
```

> **These live inside the container in a Docker volume — not on the host filesystem.** To persist them: `docker cp isaac-lab-base:/workspace/.../model_499.pt ./` or upload to HF directly.

---

## The `handle_deprecated_rsl_rl_cfg` Fix

When loading a checkpoint for evaluation, RSL-RL 2.x changed the config format. Without this call you get `KeyError: 'class_name'`:

```python
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
import importlib.metadata as _metadata
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _metadata.version("rsl-rl-lib"))
runner = OnPolicyRunner(env, agent_cfg.to_dict(), ...)
```

Both `evaluate.py` and `evaluate_seq.py` already include this.

---

## VRAM Usage (measured)

On NVIDIA L4 (23 GB), 4096 envs:
- **Peak VRAM: ~4.6 GB** — only 20% of the card
- **T4 (16 GB) is sufficient** for all state-based nav RL
- Only vision phases (P3/T3) need 24 GB+ due to camera rendering

---

## Related

- [[Isaac Lab Manager-Based RL]]
- [[Training Recipe]]
- [[Evaluation Harness]]
- [[Reward Shaping & Progress Rewards]]
- [docs/GPU_VRAM_REQUIREMENTS.md](../../docs/GPU_VRAM_REQUIREMENTS.md)
