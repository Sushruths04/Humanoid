---
tags: [concepts, reward, rl, theory, interview, formulation]
---

# Reward Engineering Deep Dive

Every reward term used across the project, with mathematical formulation, motivation, and lessons learned.

---

## The Full Reward Stack — Navigation

The total reward per timestep for the G1 humanoid navigation tasks:

```
r_total = w_nav × r_nav + w_upright × r_upright + w_base × r_base
```

### Term 1: Navigation Reward — Euclidean Distance Progress

**Formulation:**
```
r_nav = (d_prev - d_curr) × scale + Σ wrong_approach_penalty + reach_bonus
```

Where `d = ||robot_xy - target_xy||₂` (Euclidean distance in 2D).

**Breaking it down:**

**(a) Progress term — delta-distance:**
```
r_progress = (d_t-1 - d_t) × progress_scale
```
- Positive when getting closer (d decreases)
- Negative when moving away (d increases)
- Scale = 1.0 for P0/P1.2/P1.3; scale = **2.0** for P1.4 (SeqNav, harder task)

**Why delta-distance, not -distance?**
| Reward | Signal | Problem |
|---|---|---|
| `-d` (raw distance) | Sparse at each step | Gradient dominated by initial distance; no dense feedback |
| `d_prev - d_curr` | Dense every step | Every step tells you if you improved |

**(b) Wrong-marker penalty:**
```
r_penalty = -Σᵢ≠target max(0, d_prev,i - d_curr,i) × wrong_penalty_scale
```
Without this, the policy could "accidentally" approach the wrong marker to farm the reach bonus. This penalty specifically discourages approach to non-commanded markers.

**(c) Reach bonus (sparse):**
```
r_reach = reach_bonus × 𝟙[d_curr < reach_radius]
          = 10.0 × 𝟙[d < 0.5m]
```
This large sparse bonus (+10 vs typical step reward ~0.01-0.1) is the **key signal** that locks in goal-reaching. Without it, the policy might approach within 0.6m but never commit to fully reaching.

**Implementation:**
```python
dists = ||robot_xy.unsqueeze(1) - markers_xy||₂  # (N, M)
target_approach = (prev_dists - dists)[env_idx, target_id]
progress_term = target_approach × progress_scale
penalty_term = wrong_approach.clamp(min=0) × wrong_penalty_scale
bonus_term = (target_dist < reach_radius).float() × reach_bonus
r_nav = progress_term - penalty_term + bonus_term
```

---

### Term 2: Upright Reward — Quaternion-Based CoM Proxy

**Formulation:**
```
r_upright = max(0, 1 - 2(qx² + qy²))
```

Where [qw, qx, qy, qz] is the root body quaternion in world frame.

**Derivation:** The z-component of the body's "up" unit vector, expressed in world frame:
```
up_z_world = R(q) × [0, 0, 1]_body
           = 1 - 2(qx² + qy²)
```
- = 1.0 when perfectly upright (qx = qy = 0)
- = 0.0 when lying horizontal
- < 0 when inverted (clamped to 0)

**Why not use CoM directly?**
- CoM computation requires iterating over all 23+ body links and their masses
- Root orientation is a sufficient proxy: if the root tilts, the robot is about to fall
- Root quaternion is directly available as observation data

**Critical discovery — P0-stable (F-rate fix):**

| Config | Fall Rate | Upright Weight |
|---|---|---|
| P0 original | 28.1% | 0.0 (not used) |
| P0-stable | **7.8%** | **0.5** |

Adding `upright_reward_weight=0.5` provided a continuous gradient against tilting, which is the precursor to falling. With no upright reward, the policy had no incentive to maintain posture unless it was already falling (too late for a recovery signal).

---

### Term 3: Base Locomotion Reward (From Isaac Lab)

Isaac Lab's built-in locomotion reward terms (active during all nav tasks):
```
r_base = lin_vel_tracking + ang_vel_tracking - action_rate - joint_acc
```

- `lin_vel_tracking`: tracks commanded linear velocity (Gaussian: e^(-||v_cmd - v||²/σ²))
- `ang_vel_tracking`: tracks commanded angular velocity
- `action_rate`: -||aₜ - aₜ₋₁||² penalizes jittery actions
- `joint_acc`: -||θ̈||² penalizes rapid joint accelerations

**These exist to maintain stable locomotion even when no navigation target is near.**

---

## The Full Reward Stack — Obstacle Navigation

ObstacleNav adds a collision penalty on top of navigation:

```
r_total = r_nav + r_upright + r_base + r_collision
```

**Collision penalty:**
```
r_collision = -penalty_scale × Σₖ max(0, 1 - dist(robot, obstacle_k) / radius)
```

**Properties:**
- Smooth (no discontinuity at boundary → gradients flow)
- Increases as robot gets closer (linear ramp from 0 at `radius=0.4m` to `-scale` at zero distance)
- Summed over all K obstacles in the environment

**Result:** Episode collision penalty ≈ -0.0008 after training (negligible → obstacle avoidance learned well).

---

## The Full Reward Stack — Sequential Navigation

SeqNav extends navigation with subgoal ordering:

```
r_seqnav = r_nav_current_subgoal + advance_bonus × 𝟙[subgoal_i_reached]
```

- `r_nav_current_subgoal`: same progress reward but tracking current subgoal only
- `advance_bonus`: +10.0 when a subgoal is reached AND the sequence advances
- The policy must reach subgoals in order: A → B → C

**Key insight:** We initially had `progress_scale=1.0` and the policy found a **stand-still local optimum** (farmed locomotion reward). Fix: increased to `progress_scale=2.0`, making the navigation reward larger than the locomotion component → policy was forced to navigate.

---

## Manipulation Reward (GR00T / LIBERO)

For T1 (GR00T eval), we didn't define our own reward — LIBERO provides a binary terminal reward:
```
r = 1.0 if task_success else 0.0
```

**`task_success` = `env.check_success()`** — LIBERO's own metric (object at goal position + gripper state correct).

For T2 (World Model), we used:
```
r_t = 0.0 for all steps except terminal
r_T = 1.0 if task_success else 0.0
```
Dreamer-mini's RSSM learned to predict this sparse terminal reward and imagined expected return = 0.0108 (finite → WM works).

---

## Reward Reduction: How We Reduced Redundancy

### Problem 1: Too many similar terms cancel each other out
Early ObstacleNav had both:
- `distance_to_goal` (dense progress)
- `negative_velocity_toward_obstacle` (penalty)

These conflicted when the goal was behind an obstacle. **Fix:** Separate concerns — progress reward is only for target approach, collision reward only for proximity.

### Problem 2: Stand-still local optimum (SeqNav)
The base locomotion reward was large relative to the navigation reward. The policy found it easier to maximize locomotion (stand still, track zero-velocity command) than navigate. **Fix:**
- Increase `progress_scale` from 1.0 → 2.0 to outweigh locomotion component
- Reduce first waypoint radius to ensure first success is achievable

### Problem 3: Reach bonus scale wrong
Initial `reach_bonus=1.0` was too small. The dense progress reward accumulated to ~5 by the time the robot reached the goal, making the sparse bonus irrelevant. **Fix:** `reach_bonus=10.0` — makes goal arrival clearly the most rewarding event.

### Redundancy in the observation space
We removed `robot_state` (ground truth position) from nav observations and replaced with:
- `projected_gravity` (tilt signal)
- `velocity_commands` (what the policy should do)
- `base_lin/ang_vel` (what the robot is doing)

This prevents position-hacking (policy memorizes arena positions) and forces generalization.

---

## Reward Design Principles (Interview Q&A format)

**Q: What is reward shaping and why is it needed?**
A: Sparse rewards (only at goal) are hard to learn from — the policy rarely reaches the goal at random, so it gets no training signal. Reward shaping adds intermediate dense rewards (like distance progress) that guide the policy toward the goal. The risk is potential reward hacking (the Ng & Russell shaping theorem says a shaped reward with the right form doesn't change the optimal policy, but wrong shaping can create shortcuts).

**Q: What is the difference between dense and sparse rewards?**
A:
- **Dense:** signal at every timestep (progress reward). Fast learning, but risk of reward hacking.
- **Sparse:** signal only at episode end (task success). Robust but requires exploration.
- Our system uses **both**: dense progress + sparse reach bonus. Dense gets the policy close, sparse confirms goal achievement.

**Q: What is Euclidean distance used for in your project?**
A: Two places:
1. Navigation progress reward: `d = ||robot_xy - target_xy||₂`. We use the delta `d_prev - d_curr` as a dense reward signal.
2. Reach detection: `d < reach_radius (0.5m)` triggers the reach bonus.
We use 2D Euclidean distance (XY plane only) since navigation is a planar task — height is irrelevant for horizontal navigation.

**Q: Why did you use quaternion for upright reward instead of computing CoM?**
A: Center of mass computation requires knowing all link masses and positions — it's expensive and adds complexity. The root body orientation quaternion is a sufficient proxy: if the robot's root tilts significantly, it's about to fall. The formula `1 - 2(qx² + qy²)` directly extracts the cos²(tilt) without any trigonometry, which has clean gradients.

**Q: How did you detect and fix reward hacking?**
A: We always plot per-term rewards separately. When we saw `nav_term ≈ 0.0` while `total_reward` was rising, that revealed the stand-still exploit. Fix: scale the navigation reward relative to the locomotion reward so the easy exploit doesn't dominate.

---

## Related

- [[Reward Shaping & Progress Rewards]] — concise reference
- [[Robot Kinematics & Dynamics]] — CoM, quaternions explained
- [[PPO with RSL-RL]] — how rewards are consumed by the learning algorithm
- [[Interview Prep - Master Guide]] — all interview Q&As
- [programs/common/rewards.py](../../programs/common/rewards.py) — source code
