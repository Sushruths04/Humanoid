---
tags: [guide, rewards, isaac-lab, training, p0, p3]
---

# Reward Engineering Guide

> **Full version with code examples**: `docs/guides/reward_engineering_guide.md`

---

## How Rewards Work

```
total_reward = Σ (term_weight × term_value)
```
- Positive weight → encourages the behavior  
- Negative weight → penalizes the behavior  
- weight = 0 → term is disabled

Rewards fire every physics step, accumulated over `num_steps_per_env=24` steps before a PPO update.

---

## All Active Reward Terms (P0 / P3)

### Custom Navigation (`g1_command_nav_cfg.py`)

| Term | Weight | Purpose |
|---|---|---|
| `nav_command` | **+1.0** | Progress toward target + reach bonus (+10 at 0.5 m) |
| `upright` | **+0.5** | Torso vertical — prevents falling without freezing movement |

### Base Locomotion (inherited from `G1FlatEnvCfg → G1RoughEnvCfg`)

| Term | Weight | Purpose |
|---|---|---|
| `track_lin_vel_xy_exp` | **+1.0** | Match commanded XY velocity |
| `track_ang_vel_z_exp` | **+1.0** | Match commanded yaw rate |
| `feet_air_time` | **+0.75** | Proper bipedal gait (alternating air time) |
| `termination_penalty` | **−200** | Massive penalty for falling |
| `flat_orientation_l2` | **−1.0** | Penalize torso tilt |
| `lin_vel_z_l2` | **−0.2** | Penalize vertical velocity (bouncing) |
| `ang_vel_xy_l2` | **−0.05** | Penalize roll/pitch angular velocity |
| `action_rate_l2` | **−0.005** | Smooth joint targets (no jerking) |
| `dof_acc_l2` | **−1e-7** | Smooth joint acceleration |
| `dof_torques_l2` | **−2e-6** | Energy efficiency |
| `dof_pos_limits` | **−1.0** | Ankle joint limit violations |
| `feet_slide` | **−0.1** | Foot sliding when in contact |
| `joint_deviation_hip` | **−0.1** | Hip yaw/roll from default |
| `joint_deviation_arms` | **−0.1** | Shoulder/elbow from default |
| `joint_deviation_fingers` | **−0.05** | Fingers from default |
| `joint_deviation_torso` | **−0.1** | Torso joint from default |

---

## Where to Edit

**File:** `my-humanoid-project/my_humanoid_project/tasks/g1_command_nav_cfg.py`  
**Location:** `CommandConditionedG1NavCfg.__post_init__`

```python
# Change weight
self.rewards.term_name.weight = NEW_VALUE

# Disable a term
self.rewards.lin_vel_z_l2.weight = 0.0

# Change a parameter
self.rewards.nav_command.params["reach_bonus"] = 25.0

# Add a new term
self.rewards.my_term = RewTerm(func=my_function, weight=1.0, params={})
```

---

## Quick Recipes

### Jump
```python
# 1. Add jump rewards
self.rewards.jump_up = RewTerm(func=jump_reward, weight=3.0)
self.rewards.both_feet_off = RewTerm(func=both_feet_off_reward, weight=2.0)

# 2. MUST zero out terms that fight jumping
self.rewards.lin_vel_z_l2.weight = 0.0        # was -0.2 — kills vertical velocity
self.rewards.feet_air_time.weight = 0.0       # designed for walking gait
self.rewards.flat_orientation_l2.weight = -0.3 # allow lean during jump
```

### Run faster
```python
self.commands.base_velocity.ranges.lin_vel_x = (0.0, 2.5)
self.events.steer_velocity.params["speed"] = 2.0
self.rewards.track_lin_vel_xy_exp.weight = 2.0
```

### Spin in place
```python
self.rewards.spin = RewTerm(func=spin_reward, weight=2.0)
self.rewards.ang_vel_xy_l2.weight = -0.01   # relax yaw penalty
self.events.steer_velocity = None           # remove nav steering
```

### Navigate faster (bigger target, shorter episode)
```python
self.episode_length_s = 10.0
self.rewards.nav_command.params["reach_bonus"] = 25.0
self.rewards.nav_command.params["reach_radius"] = 1.0
```

### Increase stability (reduce fall rate)
```python
self.rewards.upright.weight = 1.5
self.rewards.termination_penalty.weight = -500.0
```

---

## Common Mistakes

| Mistake | Effect | Fix |
|---|---|---|
| Adding jump reward without zeroing `lin_vel_z_l2` | Robot hovers slightly instead of jumping | Zero `lin_vel_z_l2` |
| `upright` weight too high | Robot stands still — maximizes upright by not moving | Keep `upright` ≤ 0.5 × nav weight |
| Only progress reward, no reach bonus | Robot circles target forever | Keep `reach_bonus` ≥ 5× per-step max |
| Removing `termination_penalty` | Robot learns to fall if it avoids a worse state | Always keep at −200 or larger |
| `action_rate_l2` weight too large | Robot freezes — optimal action is no movement | Keep ≥ −0.01 |

---

## Related
- [[P0 - CommandNav]]
- [[P3 - VisionNav]]
- [[SeqNav Stand-Still Local Optimum]] — real example of upright weight causing stand-still
- [[Decorative Navigation Defect]] — real example of wrong reward signal

Full guide with reward functions: `docs/guides/reward_engineering_guide.md`
