---
tags: [concepts, reward, shaping, rl, progress]
---

# Reward Shaping & Progress Rewards

## The Fundamental Rule

**Always inspect per-term episode rewards, not just total reward.**

A rising total reward with a flat task-specific term = the policy is farming an easy reward while ignoring the hard one. This is exactly what happened in [[SeqNav Stand-Still Local Optimum]].

---

## The Progress Reward Shape

`commanded_target_reward` in `programs/common/rewards.py`:

```
reward = progress_term + wrong_marker_penalty + reach_bonus

progress_term     = (prev_dist_to_target - cur_dist_to_target) * progress_scale
                  → positive when closing distance, negative when moving away

wrong_marker_penalty = -sum(approach to non-commanded markers) * wrong_penalty_scale
                  → prevents gaming by approaching the wrong marker "accidentally"

reach_bonus       = +10.0 when within reach_radius of the commanded target
                  → the strong sparse signal that locks in the goal
```

**Why `prev_dist - cur_dist` (delta-distance) instead of `-dist`?**
- Delta-distance is dense: every step you get a signal for whether you made progress.
- Raw `-dist` gives no info between steps; the gradient is dominated by the initial distance.
- Delta-distance is frame-independent; it works whether the target is 1 m or 5 m away.

---

## The Stand-Still Local Optimum

When a task is too hard to reach the first goal before episode timeout, the policy can find a different optimum:

```
Actual reward landscape:
  nav reward (navigating well) = +1 to +3 per episode
  base locomotion reward (standing still) = ~+8 per episode (just tracks zero vel command)

If the policy CAN'T reach the navigation reward (targets too far, episode too short):
  it settles on the locomotion reward instead.
  Result: nav_command term stays ≈ 0, robot stands still, total reward looks "good."
```

**Fix:** make the first success achievable. For SeqNav: reduce `RADIUS_RANGE` from `(2.0, 5.0)` to `(1.0, 2.5)`.

---

## Parameter Tuning Table

| Task | progress_scale | wrong_penalty_scale | reach_bonus | reward weight |
|---|---|---|---|---|
| CommandNav | 1.0 | 1.0 | 10.0 | 1.0 |
| LangNav | 1.0 | 1.0 | 10.0 | 1.0 |
| ObstacleNav | 1.0 | 1.0 | 10.0 | 1.0 (nav) + 1.0 (collision) |
| SeqNav (final) | **2.0** | **1.0** | 10.0 | 1.0 |

SeqNav needed `progress_scale=2.0` because the task is harder (two subgoals) and the policy needs a stronger signal to overcome the locomotion-farming tendency once it starts navigating.

---

## Collision Penalty (ObstacleNav)

```
collision_penalty = -sum(smooth_proximity to obstacles)
                  where proximity = max(0, 1 - dist/collision_radius)^2
```

This is smooth (no discontinuity at the boundary) so gradients flow nicely. `collision_radius=0.4`, `penalty_scale=1.0`. After training: episode collision penalty ≈ −0.0008 (negligible) → obstacle avoidance is working.

---

## Reach Bonus vs. Progress

The reach bonus (+10) is the **sparse** signal that tells the policy "you made it to the goal." Progress reward alone can lead to sub-optimal policies that approach but never commit (since approach gives reward continuously). The large reach bonus makes actually arriving highly rewarding.

For sequential tasks, there's an advance bonus too (same magnitude as reach bonus) per subgoal intermediate completion.

---

## Related

- [[Command-Conditioned Navigation]]
- [[SeqNav Stand-Still Local Optimum]]
- [[PPO with RSL-RL]]
- [programs/common/rewards.py](../../programs/common/rewards.py)
