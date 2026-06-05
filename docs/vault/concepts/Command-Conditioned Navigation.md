---
tags: [concepts, navigation, command-conditioning, design-pattern]
---

# Command-Conditioned Navigation

## The Core Design Pattern

All four nav tasks ([[P0 - CommandNav]], [[P1.2 - LangNav]], [[P1.3 - ObstacleNav]], [[P1.4 - SeqNav]]) are built on the same pattern. Understanding it fully is the key to understanding the whole P0–P1 track.

**The problem with vanilla locomotion RL:** if you just add a velocity-tracking reward, the robot learns to walk fast but not *where* to go. The command is random. You need the behavior to be *causally* conditioned on the command (which target to walk toward), otherwise it's [[Decorative Navigation Defect]].

**The solution:**
1. Sample a **commanded target** per episode (random marker position, random id)
2. Every step, compute a steering command that points the robot at THAT target
3. Write it into the velocity command — so the robot's locomotion reward now aligns with actually reaching the target
4. Give a reward for *progress toward the commanded target specifically* (not any target)
5. Expose the command in observations so the policy can be conditioned on it

---

## The Three Required Components

### Component 1: Reset event (per-episode randomization)

```python
def reset_nav_command(env, env_ids, num_markers=2, radius_range=(2.0, 5.0)):
    _ensure_buffers(env, num_markers)   # creates _nav_target_ids, _nav_markers_xy, _nav_prev_xy
    k = len(env_ids)
    env._nav_target_ids[env_ids] = sample_target_ids(k, num_markers, device=env.device)
    env._nav_markers_xy[env_ids] = sample_marker_positions(k, num_markers, radius_range, device=env.device)
    env._nav_prev_xy[env_ids] = _robot_xy(env, env_ids)
```

Every episode gets: a random commanded target id (which marker to walk toward), random marker positions (where each marker is in local frame), and a baseline for computing progress rewards.

### Component 2: Per-step steer event (fires every step)

```python
def steer_velocity_to_target(env, env_ids, num_markers=2, speed=1.0, yaw_gain=0.5, max_yaw_rate=1.0):
    cmd = velocity_command_to_target(_robot_xy(env), _robot_yaw(env), _commanded_target_xy(env),
                                     speed=speed, yaw_gain=yaw_gain, max_yaw_rate=max_yaw_rate)
    env.command_manager.get_term("base_velocity").vel_command_b[:] = cmd
```

Registered as `mode="interval", interval_range_s=(0.0, 0.0)` so it fires every physics step. See [[Velocity-Command Steering Law]].

### Component 3: Observation term

```python
def nav_command_obs(env, num_markers=2):
    onehot = target_id_to_onehot(env._nav_target_ids, num_markers)  # which target
    rel = _commanded_target_xy(env) - _robot_xy(env)                # relative position
    return torch.cat([onehot, rel], dim=-1)
```

The policy sees: a one-hot encoding of which marker it should go to, plus a 2D relative vector pointing at it. This is what lets the behavior *change* with the command.

---

## How to Verify It Is Working (Instruction-Swap Probe)

Train a policy, then evaluate: if you swap the command (e.g. tell it "go to marker 0" when it was trained on "go to marker 1"), does it go to the correct one? If yes, the conditioning is genuine. If it always goes the same direction regardless of command, it's decorative.

In practice: per-command success rates in eval (e.g. `success_by_command = [83.7%, 88.0%]`) must both be well above the baseline — that means the behavior genuinely changes with the command.

---

## What Makes This Extensible

All harder tasks layer on top of this base:
- **LangNav** — adds a frozen text embedding of the command color name as extra obs
- **ObstacleNav** — swaps the steer function for a potential-field avoiding version
- **SeqNav** — adds a phase-advance event that re-points `_nav_target_ids` to the next subgoal when one is reached

The policy sees the same pattern (one-hot + relative vector), just with a target that moves (SeqNav) or with an embedded language command (LangNav).

---

## Related

- [[Isaac Lab Manager-Based RL]]
- [[Velocity-Command Steering Law]]
- [[Reward Shaping & Progress Rewards]]
- [[Decorative Navigation Defect]]
- [[P0 - CommandNav]]
- [programs/common/commands.py](../../programs/common/commands.py)
- [programs/common/rewards.py](../../programs/common/rewards.py)
