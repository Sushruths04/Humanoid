---
tags: [workflow, training, recipe, commands]
---

# Training Recipe

## The One-Liner

```bash
bash programs/scripts/train_eval_nav.sh <TASK_ID> 4096 500 256
```

This trains (4096 envs, 500 iters) then evaluates (256 envs) and writes a result markdown to `docs/results/`. For a single task it takes ~20–25 min on L4.

---

## Prerequisites

1. Container is up: `docker ps | grep isaac-lab-base` shows Up
2. On the right branch: `git rev-parse --abbrev-ref HEAD` → `feat/planned-scripts`
3. `programs/results/` is host-writable (chown if container just wrote to it)

---

## Available Task IDs

| Task ID | What it trains |
|---|---|
| `Humanoid-G1-CommandNav-v0` | baseline nav to commanded target |
| `Humanoid-G1-LangNav-v0` | language-conditioned nav (text embedding) |
| `Humanoid-G1-ObstacleNav-v0` | nav + obstacle avoidance |
| `Humanoid-G1-SeqNav-v0` | sequential two-subgoal nav |

---

## Run All Four at Once

```bash
bash programs/scripts/batch_test_nav.sh
# or for a quick 2-iter wiring check first:
SMOKE=1 bash programs/scripts/batch_test_nav.sh
```

---

## Detached Long Run (recommended for 20-min jobs)

```bash
nohup docker exec \
  -e PYTHONPATH="/workspace:/workspace/my-humanoid-project:/workspace/isaaclab/source" \
  isaac-lab-base \
  /workspace/isaaclab/isaaclab.sh -p /workspace/my-humanoid-project/custom_train.py \
  --task Humanoid-G1-ObstacleNav-v0 --headless --num_envs 4096 --max_iterations 500 \
  > _runlogs/obstaclenav.log 2>&1 &
echo "PID=$!"
# Monitor:
tail -f _runlogs/obstaclenav.log
```

---

## Reading the Log

```
Learning iteration 221/500
  Mean reward: 4.17                          ← should rise from ~-5 to positive by ~iter 200
  Episode_Reward/nav_command: 1.93           ← WATCH THIS — should climb, not stay ~0
  Episode_Reward/track_lin_vel_xy_exp: 0.85  ← base locomotion (will always be ~0.7-0.9)
  Episode_Termination/base_contact: 0.024    ← fall rate (known follow-up)
  Iteration time: 2.38s
  ETA: 00:16:33
```

**Red flag:** `nav_command ≈ 0.001` while total reward is positive → stand-still failure. See [[Reward Shaping & Progress Rewards]] and [[SeqNav Stand-Still Local Optimum]].

---

## Checkpoints

Land in the container at:
```
/workspace/isaaclab/logs/rsl_rl/g1_flat/<YYYY-MM-DD_HH-MM-SS>/model_499.pt
```

Copy out immediately after training (container is ephemeral):
```bash
# Find the latest run dir:
docker exec isaac-lab-base bash -lc "ls -td /workspace/isaaclab/logs/rsl_rl/g1_flat/*/ | head -1"
# Copy checkpoint:
docker cp isaac-lab-base:/workspace/isaaclab/logs/rsl_rl/g1_flat/<timestamp>/model_499.pt ./
# Upload to HF:
hf upload mitvho09/humanoid-g1-nav ./model_499.pt checkpoints/g1_<task>/model_499.pt
```

---

## Related

- [[Isaac Sim Docker Container]]
- [[PYTHONPATH & Python Interpreters]]
- [[Evaluation Harness]]
- [[PPO with RSL-RL]]
- [programs/scripts/train_eval_nav.sh](../../programs/scripts/train_eval_nav.sh)
- [docs/PLANNED_SCRIPTS.md](../../docs/PLANNED_SCRIPTS.md)
