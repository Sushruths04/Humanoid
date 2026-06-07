---
tags: [failure, eval, play, rsl-rl, p3, stats]
---

# play.py Eval Has No Episode Output

## Symptom
Ran `custom_play.py` for eval. Process runs for 20+ minutes at 33% GPU / 180% CPU (inference is happening), but the log file produces no episode stats, no success rate, no termination — just the startup messages. Must kill the process manually.

## Root Cause
`play.py`'s episode loop has **no print statements**:
```python
obs = env.get_observations()
while simulation_app.is_running():
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
    # nothing printed — runs forever
```

Episode-level stats (success rate, time_out fraction, reward) are only logged in `OnPolicyRunner.learn()`, which is called during training — NOT called during `runner.load()` + eval play.

In headless mode without `--video`, `simulation_app.is_running()` never returns False — the loop runs until the process is killed.

Additionally, Python's stdout is **fully buffered** when redirected to a file (`> log`). Even `print(f"Loading model checkpoint from: {path}")` statements sit in an 8 KB buffer and never flush to disk as long as the process runs.

## What Was Actually Happening
The process WAS running inference (GPU at 33%, CPU at 180%), policy loaded, episodes running. We just couldn't observe it. The policy was walking, receiving camera observations, and taking actions — silently.

## Fix: Use Training Stats Instead
Training-epoch episode stats are printed every iteration during `learn()` and are statistically more robust:
- **Training stats source**: `time_out` fraction at convergence = fraction of episodes that reached the target before timeout
- **P3 final stat**: `time_out=0.9628` at iter 499 → **96.28% success**
- Sample size: 4096 envs × hundreds of steps per iter × 300 iters >> 512 envs × 1 eval run

Training stats > play.py eval stats for this reason.

## Fix: Get Eval Stats from a Custom Script
If formal eval stats are needed from play.py-style inference, write a custom eval script:
```python
import torch
from isaaclab_tasks.utils import get_checkpoint_path

N_EVAL_EPISODES = 200
successes = 0; total = 0

obs = env.get_observations()
while total < N_EVAL_EPISODES:
    actions = policy(obs)
    obs, _, dones, extras = env.step(actions)
    if dones.any():
        # check per-env time_out vs termination
        successes += extras.get("time_out", dones).sum().item()
        total += dones.sum().item()

print(f"Success rate: {successes/total:.1%} ({successes}/{total})")
env.close()
```

## Rule
> **Don't expect eval stats from stock play.py.** For P3-style tasks: use training epoch stats (more robust). For formal eval: write a custom N-episode evaluator.

## Related
- [[play.py Checkpoint Bare Filename Not Found]]
- [[play.py Fails - Custom Task Not Registered]]
- [[RSL-RL Resume Resets Loop Counter]]
