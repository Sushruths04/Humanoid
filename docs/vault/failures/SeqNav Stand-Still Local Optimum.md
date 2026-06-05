---
tags: [failure, seqnav, training, bootstrap, local-optimum, debugging, important]
---

# SeqNav Stand-Still Local Optimum ⭐

> This is the most instructive failure in the program. Read it fully.

---

## The Symptom

SeqNav trained for 500 iterations. Training logs looked healthy:
- Mean reward: **+8 to +14** (rising and positive)
- Episode terminations: 97% by timeout (robot surviving)

But evaluation showed:
- `full_sequence_success`: **0.4%** (basically zero)
- `first_subgoal_rate`: **6%**
- `mean_robot_displacement`: **0.5 m** (robot barely moved an entire episode)
- `Episode_Reward/nav_command`: **≈ 0.001** (the navigation term was essentially zero)

---

## The Investigation (step by step — this is the method)

### Step 1: Build a proper evaluator
The original `evaluate.py` crashed on SeqNav with `AttributeError: _nav_target_ids` because SeqNav originally used different buffers. Built `evaluate_seq.py` with `sequence_eval_metrics` to track per-subgoal first-reach times and compute full-sequence + ordering metrics. ← **Lesson: match the evaluator to the task.**

### Step 2: Instrument the eval
Added diagnostics to the evaluator:
- `mean_robot_displacement` — how far did the robot actually travel?
- `mean_cmd_vel_norm` — how strong was the steering command?
- `mean_min_dist_subgoal_k` — did the robot even get close?

Results: displacement=0.5 m, cmd_vel_norm=0.248 m/s, min_dist_subgoal0=2.3 m.

**The robot was barely moving, and the steering command was weak (~0.25 vs expected ~1.0).**

### Step 3: Rule out code bugs (the expensive step)
Hypothesized: maybe the SeqNav reimplementation had a subtle bug.

**Rebuilt SeqNav completely** on the proven CommandNav core (reuse `steer_velocity_to_target`, `nav_command_obs`, `nav_command_reward` verbatim, add only a `seq_advance` event). **Retrained. Same result — nav reward still ≈ 0.**

This was expensive (2 extra GPU training runs) but proved: **the code was not the problem.**

### Step 4: Ground-truth diagnostic
Ran a direct env diagnostic (no policy, just reset + call the steer function):
```
target_dist_mean = 3.443 m    (target is sanely placed)
cmd_vel_norm = 0.307 m/s
vx_max = 0.994                (when facing the target, full forward speed is available)
wz_absmean = 0.690            (robot turns significantly)
```

**The steering command is correct** — it CAN command full speed when facing the target. So target placement, steering law, and reward code are all verified correct.

### Step 5: Understand the training dynamics

The real landscape revealed by the per-term breakdown:
```
Episode_Reward/track_lin_vel_xy_exp: 0.85   ← BASE LOCOMOTION reward
Episode_Reward/nav_command:          0.001  ← navigation reward (FLAT)
```

**The policy is farming locomotion reward.** Standing still and tracking a near-zero velocity command earns ~+8 total reward. Navigating to a marker 2–5 m away in a sequence earns potentially +1–3 nav reward but requires actually reaching the target, which is **hard** when targets are far away and you need TWO in sequence.

Since the policy never reaches a subgoal, it never experiences the reach bonus, so it never has a gradient to learn navigation. It gets stuck in a local optimum.

---

## Root Cause

**Training bootstrap failure** — not a code bug.

With two sequential subgoals 2–5 m apart:
- The expected number of steps to reach subgoal-0 by random/early policy ≈ episode length
- The policy never triggers the reach bonus (+10) in early training
- Without that sparse signal, there's no gradient to learn navigation
- The policy finds the easier path: stand still, track near-zero velocity = ~+8 locomotion reward
- This optimum is stable — once found, there's no gradient to escape it

---

## The Fix

**Closer targets: `RADIUS_RANGE` from `(2.0, 5.0)` to `(1.0, 2.5)` m.**

With targets 1–2.5 m away, the robot reaches subgoal-0 within the first ~100 training iterations of an early policy, triggering the reach bonus. This starts the reach→bonus→progress gradient loop. Once the policy learns to reach one target, it can be fine-tuned to reach the sequence.

Also: `progress_scale` 1.0 → 2.0 (stronger navigation signal relative to locomotion).

**Result after fix:** nav reward jumped from 0.001 to **2.9–3.5** at convergence. Full-sequence success: **80.9%**. Ordering accuracy: **94.5%**.

---

## The Generalizable Lesson

> **If a task-specific reward term stays flat near zero while total reward rises, you have a BOOTSTRAP / EXPLORATION problem, not a reward-magnitude problem.**

1. Scaling near-zero progress by 10× is still near-zero.
2. The policy needs to *see* the reward at least once to start learning from it.
3. Make the **first success** easy: closer targets, curriculum, auxiliary rewards, etc.
4. Only after the first success is routinely achieved should you increase difficulty.

This pattern shows up everywhere in RL: if the reward signal requires a long sequence of actions to trigger for the first time, the random policy will never see it and gradient-based methods can't learn from it.

---

## What Didn't Work (also instructive)

- **Increasing reward weight (×2, ×5, ×10):** scaling ≈0 signal gives ≈0 gradient
- **Rebuilding the env on proven code:** the code was always correct; it was a dynamics problem
- **Hypothesis that "iter 91 is too early to judge":** at iter 300, reward was still +7 but nav still ≈0 — confirmed it wasn't just slow bootstrapping

---

## Related

- [[P1.4 - SeqNav]]
- [[Sequential Subgoal Navigation]]
- [[Reward Shaping & Progress Rewards]]
- [[Eval Crash - Missing Buffer]]
- [[00 - Failure Index]]
