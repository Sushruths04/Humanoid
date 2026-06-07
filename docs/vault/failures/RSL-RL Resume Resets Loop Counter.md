---
tags: [failure, rsl-rl, resume, checkpoint, ppo, p3, iteration-counter]
---

# RSL-RL Resume Resets Loop Counter

## Symptom
Ran training with `--resume --load_run run_final --checkpoint model_200.pt` expecting 100 more iterations (200→300). The log showed:
- ETA: **1:49:11** (should be ~37 min for 100 iters)
- First reward: **-0.04** (should be near +27.74 if weights loaded correctly)
- New checkpoints named: `model_200.pt`, `model_205.pt`, `model_210.pt` ← wait, correct names?

## What Actually Happened
RSL-RL's `OnPolicyRunner.learn()` does:
```python
for it in range(self.current_learning_iteration, 
                self.current_learning_iteration + num_learning_iterations):
    ...
```

Where:
- `self.current_learning_iteration = 200` (read from `model_200.pt`'s `iter` field) ✓
- `num_learning_iterations = max_iterations = 300` (from config)

So the loop runs **`range(200, 500)`** = **300 new iterations** on top of the loaded checkpoint. Final checkpoint: `model_499.pt`.

The ETA of 1:49 = 300 × 21.91s confirmed this. The "100 remaining" assumption was wrong — `max_iterations` means "how many new iterations to run", not "what iteration to stop at".

## Why Reward Started Low (-0.04)
The checkpoint loaded the policy weights AND obs normalizer statistics. However, the first few iterations update the normalizer with fresh environment data, temporarily distorting the normalized observations while the running mean/std recalibrates. This makes the policy behave suboptimally for ~10-20 iterations even with correct weights.

By iteration 215 reward was already +10.94 (faster than cold start's iter 100 = -6.76).

## Outcome: Better Than Expected
Running 300 new iterations on top of model_200.pt (which had +27.74 reward) produced:
- Iter 260: +109.73 (4× better than any previous checkpoint)
- Iter 499: **+141.35** (final)
- Success rate: **96.28%**

The "failure" to do only 100 iters was actually optimal — more training on a warm start produced a dramatically better policy.

## The Lesson
**`max_iterations` in RSL-RL = number of NEW iterations to run, not the final iteration number.**

If you want to run exactly to iter 300 when resuming from iter 200:
```python
# Set max_iterations = 100 (the delta), not 300 (the target)
-e P3_MAX_ITERS="100"
```

Or just embrace the full 300 new iterations — warm-started training converges much faster, so the extra time is often worth it.

## Checkpoint Naming
RSL-RL names checkpoints by `current_learning_iteration + loop_index`:
- Loaded from `model_200.pt` → `current_learning_iteration = 200`
- After 5 new iters: saves `model_205.pt`
- After 299 new iters: saves `model_499.pt` (= 200 + 299)

Final checkpoint for `--resume` from 200 + 300 new iters = `model_499.pt`, NOT `model_500.pt`.

## Related
- [[PPO with RSL-RL]]
- [[P3 - VisionNav]]
- [[Training Recipe]]
