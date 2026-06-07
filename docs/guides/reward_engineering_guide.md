# Reward Engineering Guide — G1 Humanoid (Isaac Lab)

How to read, understand, and modify every reward term used in P0/P3. Includes worked examples for custom behaviors (jumping, running, spinning, etc.).

---

## 1. How Rewards Work in Isaac Lab

Each training step returns a scalar reward per environment:

```
total_reward = Σ (term_weight × term_value)
```

- **Positive weight** → encourages the behavior
- **Negative weight** → penalizes the behavior
- **weight = 0** → term is disabled (computed but ignored)

The reward is computed every physics step, then accumulated over `num_steps_per_env` steps before a PPO update. So a term with weight `+1.0` that fires every step has roughly 24× more impact per update than a bonus with weight `+10.0` that fires once per episode.

---

## 2. Full Reward Table — P0 CommandNav / P3 VisionNav

These are ALL active reward terms after the inheritance chain resolves:
`G1VisionNavEnvCfg → CommandConditionedG1NavCfg → G1FlatEnvCfg → G1RoughEnvCfg → LocomotionVelocityRoughEnvCfg`

### Navigation rewards (custom — `g1_command_nav_cfg.py`)

| Term | Weight | What it measures | Effect |
|---|---|---|---|
| `nav_command` | **+1.0** | Progress toward commanded target + reach bonus | Core navigation objective |
| `upright` | **+0.5** | How vertical the robot's torso is (`cos(tilt_angle)`) | Prevents falling; reduced from default to allow dynamic motion |

**`nav_command` breakdown:**
- `+progress_scale × Δdist_to_target` per step (reward proportional to how much closer you got)
- `+reach_bonus = 10.0` one-time when within `reach_radius = 0.5 m`
- `−wrong_penalty_scale × dist_to_wrong_target` if moving toward a wrong marker

### Locomotion rewards (from `G1Rewards` / `G1FlatEnvCfg`)

| Term | Weight | What it measures | Effect |
|---|---|---|---|
| `track_lin_vel_xy_exp` | **+1.0** | exp(−‖cmd_vel − actual_vel‖² / 0.5²) | Track commanded XY velocity |
| `track_ang_vel_z_exp` | **+1.0** (flat) | exp(−‖cmd_yaw − actual_yaw‖² / 0.5²) | Track commanded yaw rate |
| `feet_air_time` | **+0.75** (flat) | Seconds each foot spends airborne per step | Encourages proper bipedal gait |
| `termination_penalty` | **−200.0** | 1 if episode terminates (robot falls) | Strong penalty for falling |
| `flat_orientation_l2` | **−1.0** | ‖torso tilt from vertical‖² | Penalizes leaning |
| `lin_vel_z_l2` | **−0.2** (flat) | Vertical velocity² | Discourages bouncing |
| `ang_vel_xy_l2` | **−0.05** | Roll+pitch angular velocity² | Discourages pitching/rolling |
| `action_rate_l2` | **−0.005** | ‖action[t] − action[t−1]‖² | Smooth joint targets (no jerking) |
| `dof_acc_l2` | **−1e-7** | Joint acceleration² (hip+knee only) | Smooth motion |
| `dof_torques_l2` | **−2e-6** (flat) | Joint torque² (hip+knee) | Energy efficiency |
| `dof_pos_limits` | **−1.0** | Ankle joint limit violations | Protects ankle joints |
| `feet_slide` | **−0.1** | Foot velocity when in contact with ground | Prevents foot sliding |
| `joint_deviation_hip` | **−0.1** | Hip yaw + roll deviation from default | Keeps hips aligned |
| `joint_deviation_arms` | **−0.1** | Shoulder/elbow deviation from default | Keeps arms still |
| `joint_deviation_fingers` | **−0.05** | Finger deviation from default | Keeps fingers relaxed |
| `joint_deviation_torso` | **−0.1** | Torso joint deviation from default | Keeps torso stable |

---

## 3. Where to Change Rewards

All custom reward changes go in **`my-humanoid-project/my_humanoid_project/tasks/g1_command_nav_cfg.py`** (for P0/P3) inside `CommandConditionedG1NavCfg.__post_init__`.

**Syntax:**
```python
# Change weight of existing term
self.rewards.term_name.weight = NEW_WEIGHT

# Disable a term
self.rewards.term_name.weight = 0.0

# Change a parameter of an existing term
self.rewards.term_name.params["param_key"] = new_value

# Add a brand-new term
from isaaclab.managers import RewardTermCfg as RewTerm
self.rewards.my_new_term = RewTerm(func=my_function, weight=1.0, params={...})
```

---

## 4. Worked Examples — Custom Behaviors

### 4a. Make the robot run faster

The robot's commanded speed is capped at 1.0 m/s. To get faster locomotion:

```python
# in g1_command_nav_cfg.py → CommandConditionedG1NavCfg.__post_init__
# 1. Widen the command range
self.commands.base_velocity.ranges.lin_vel_x = (0.0, 2.5)  # was (0.0, 1.0)

# 2. Also update the steer event speed
self.events.steer_velocity.params["speed"] = 2.0  # was 1.0

# 3. Reward higher speed more
self.rewards.track_lin_vel_xy_exp.weight = 2.0  # was 1.0
```

---

### 4b. Make the robot jump

Jumping requires rewarding upward velocity and feet leaving the ground simultaneously. Add these in `__post_init__`:

```python
import torch

def jump_reward(env):
    """Reward for having positive vertical velocity (launching phase)."""
    robot = env.scene["robot"]
    vz = robot.data.root_lin_vel_w[:, 2]           # vertical velocity
    height = robot.data.root_pos_w[:, 2]           # CoM height above ground
    # reward when going up AND above standing height (~0.8 m for G1)
    return torch.clamp(vz, min=0.0) * (height > 0.85).float()

def both_feet_off_reward(env):
    """Reward for having both feet airborne simultaneously."""
    # contact_forces sensor — zero contact = in the air
    contacts = env.scene["contact_forces"].data.net_forces_w_history
    # contacts shape: [N, history, bodies, 3]
    # ankle_roll_link indices are [6, 13] (left, right)
    left_contact = contacts[:, 0, 6, :].norm(dim=-1)
    right_contact = contacts[:, 0, 13, :].norm(dim=-1)
    both_airborne = (left_contact < 1.0) & (right_contact < 1.0)
    return both_airborne.float()

# Add to rewards:
self.rewards.jump_up = RewTerm(func=jump_reward, weight=3.0, params={})
self.rewards.both_feet_off = RewTerm(func=both_feet_off_reward, weight=2.0, params={})

# CRITICAL: reduce penalties that fight jumping
self.rewards.lin_vel_z_l2.weight = 0.0       # was -0.2; this penalizes upward velocity
self.rewards.flat_orientation_l2.weight = -0.3  # was -1.0; allow some lean during jump
self.rewards.feet_air_time.weight = 0.0      # designed for walking, not jumping
```

**Why it works:** `lin_vel_z_l2` penalizes any vertical motion — you MUST zero it out, otherwise it fights your jump reward. `feet_air_time` rewards alternating air time (walking gait) and conflicts with simultaneous airborne.

---

### 4c. Make the robot spin in place

```python
def spin_reward(env):
    """Reward high yaw angular velocity."""
    robot = env.scene["robot"]
    yaw_rate = robot.data.root_ang_vel_w[:, 2]    # world-frame yaw rate
    return torch.abs(yaw_rate)                    # reward magnitude of spin

# Add spin reward
self.rewards.spin = RewTerm(func=spin_reward, weight=2.0, params={})

# Relax angular velocity penalty (fights spinning)
self.rewards.ang_vel_xy_l2.weight = -0.01   # was -0.05; keep pitch/roll damping

# Fix the commanded yaw to a high value so steer_velocity doesn't zero it out
# Or disable the steer event entirely for pure spinning:
self.events.steer_velocity = None  # remove navigation steering
```

---

### 4d. Make the robot crouch / stay low

```python
def low_height_reward(env):
    """Reward for keeping CoM below a target height."""
    target_height = 0.6  # crouched position for G1 (~0.9m is standing)
    actual_height = env.scene["robot"].data.root_pos_w[:, 2]
    return torch.exp(-torch.abs(actual_height - target_height) / 0.1)

self.rewards.crouch = RewTerm(func=low_height_reward, weight=2.0, params={})

# Also relax joint deviation penalties for knees (need to bend)
self.rewards.dof_acc_l2.weight = -5e-8  # allow more aggressive joint motion
```

---

### 4e. Make the robot stop falling (increase upright stability)

The P0 `upright_reward` was tuned to 0.5 (reduced from higher values that caused the robot to stand still). To get the least-fall-rate possible:

```python
# In g1_command_nav_cfg.py, change the env variable default:
UPRIGHT_REWARD_WEIGHT = float(os.environ.get("COMMANDNAV_UPRIGHT_WEIGHT", "1.0"))  # was 0.5

# Or override directly:
self.rewards.upright.weight = 1.5

# Also make termination penalty larger (very strong signal to not fall):
self.rewards.termination_penalty.weight = -500.0  # was -200.0
```

---

### 4f. Make the robot navigate to targets faster (shrink episode)

```python
# Shorter episodes → robot learns urgency
self.episode_length_s = 10.0  # was 20.0

# Larger reach bonus → bigger signal for arriving
self.rewards.nav_command.params["reach_bonus"] = 25.0   # was 10.0

# Larger reach radius → easier success, good for curriculum
self.rewards.nav_command.params["reach_radius"] = 1.0   # was 0.5
```

---

### 4g. Make the robot go to the WRONG target (invert objective — for ablation)

```python
# Invert the navigation reward weight
self.rewards.nav_command.weight = -1.0   # was +1.0
```

---

## 5. Anti-patterns to Avoid

| Mistake | What happens | Fix |
|---|---|---|
| Adding jump reward without zeroing `lin_vel_z_l2` | Reward farming — robot learns to hover slightly (tiny vz = no penalty) instead of jumping | Zero `lin_vel_z_l2` when rewarding vertical motion |
| High `upright` weight | Robot stands still — upright reward is maximized without moving | Keep `upright` ≤ 0.5 × navigation reward |
| Only reward tracking, no feet_air_time | Robot slides along the ground instead of walking | Keep `feet_air_time` for any walking task |
| Reward delta-progress only (no reach bonus) | Robot circles the target at `reach_radius + ε` — never arrives | Keep `reach_bonus` at least 5× the per-step max progress |
| Negative weight on `action_rate_l2` too large | Robot freezes — optimal action is to not move | Keep `action_rate_l2` weight ≥ −0.01 |
| Remove `termination_penalty` | Robot learns to fall on purpose if it avoids a worse state | Always keep at −200 or larger |

---

## 6. Reward Debugging Workflow

### Check if a new term is being learned
```bash
# Watch per-term reward in TensorBoard:
tensorboard --logdir /teamspace/studios/this_studio/Humanoid/IsaacLab/logs/rsl_rl/

# Look for: Episode/Reward/<term_name>
# A flat line = term not changing behavior
# A rising line = term being optimized
```

### Print reward breakdown during eval
```python
# In your custom_play.py, after env.step():
if hasattr(env.unwrapped, 'reward_manager'):
    for name, val in env.unwrapped.reward_manager._episode_sums.items():
        print(f"  {name}: {val.mean().item():.3f}")
```

### Sanity-check your reward function
```python
# Add a one-step test before training:
import torch
obs, _ = env.reset()
for _ in range(5):
    actions = torch.zeros(env.num_envs, env.action_space.shape[0], device=env.device)
    obs, reward, done, _, info = env.step(actions)
    print(f"reward={reward.mean():.3f}  done={done.sum()}")
```

---

## 7. File Locations

| File | Purpose |
|---|---|
| `my-humanoid-project/my_humanoid_project/tasks/g1_command_nav_cfg.py` | **Custom P0/P3 rewards** — edit here |
| `my-humanoid-project/my_humanoid_project/tasks/g1_vision_nav_cfg.py` | P3 camera config + CNN runner |
| `IsaacLab/source/isaaclab_tasks/.../config/g1/rough_env_cfg.py` | Base G1 rewards (G1Rewards class) |
| `IsaacLab/source/isaaclab_tasks/.../config/g1/flat_env_cfg.py` | Flat-terrain overrides |
| `IsaacLab/source/isaaclab_tasks/.../velocity_env_cfg.py` | Root locomotion rewards |
| `IsaacLab/source/isaaclab_tasks/.../mdp/rewards.py` | All built-in reward functions |
| `programs/common/rewards.py` | Custom reward functions (upright, commanded_target) |
