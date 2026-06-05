---
tags: [concepts, isaac-lab, rl, manager-based]
---

# Isaac Lab Manager-Based RL

## What It Is

Isaac Lab's `ManagerBasedRLEnv` lets you build RL environments by registering **named terms** into managers (observations, rewards, events, commands, terminations). You subclass a base config, override `__post_init__`, and add your terms. The simulator handles vectorized physics across thousands of parallel environments.

All nav tasks in this project subclass `G1FlatEnvCfg` — the pre-built Unitree G1 flat-terrain locomotion environment. You don't have to build a walking robot from scratch; you inherit it and add navigation on top.

---

## The Four Managers You Touch

### 1. `commands` — what the policy is commanded to do
Controls the base velocity target. In the stock G1FlatEnvCfg, the command manager randomly samples a velocity direction every few seconds. For navigation we **disable this resampling** and instead write to it ourselves from a steer event.

```python
self.commands.base_velocity.resampling_time_range = (1e9, 1e9)  # effectively never resample
self.commands.base_velocity.heading_command = False
```

### 2. `events` — things that fire at specific moments
Two modes matter:
- `mode="reset"` — fires when an episode resets (use to randomize positions, commands).
- `mode="interval", interval_range_s=(0.0, 0.0)` — fires **every physics step** (use for per-step steering).

**Critical gotcha:** event function signatures MUST have `env_ids` as a required parameter (no default). If you write `def my_event(env, env_ids=None)`, Isaac Lab will error at runtime.

```python
def my_reset_event(env, env_ids):  # CORRECT
    ...

def my_step_event(env, env_ids):   # CORRECT
    ...
```

### 3. `observations.policy` — what the policy sees
Each `ObsTerm` calls a function that returns a tensor; all terms are concatenated into the policy's observation vector. The total observation dimension must match what PPO expects.

### 4. `rewards` — the training signal
Each `RewardTerm` calls a function that returns a scalar reward per env. The `weight` parameter scales it before summing. **Always inspect per-term episode rewards in the training log** — not just the total.

---

## The Navigation Pattern (used by all tasks)

The key insight: G1FlatEnvCfg already teaches the G1 to **track a base-velocity command** (forward speed, turning rate). Navigation is just about **pointing that command at the target**. You don't retrain locomotion from scratch — you hijack the command.

```
Every step:
  1. Compute desired heading to target (atan2)
  2. Compute steering command (vx, vy=0, wz) that faces and walks toward target
  3. Write it to the command manager: env.command_manager.get_term("base_velocity").vel_command_b[:] = cmd
  4. The G1 locomotion policy tracks this command → robot walks toward target
```

See [[Command-Conditioned Navigation]] and [[Velocity-Command Steering Law]] for details.

---

## Task Registration

```python
# In tasks/__init__.py — registers with gymnasium
gym.register(
    id="Humanoid-G1-CommandNav-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "my_humanoid_project.tasks.g1_command_nav_cfg:CommandConditionedG1NavCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
    disable_env_checker=True,
)
```

The `rsl_rl_cfg_entry_point` points to the PPO config. All nav tasks share `G1FlatPPORunnerCfg` (experiment name `g1_flat`).

---

## Related

- [[Command-Conditioned Navigation]]
- [[PPO with RSL-RL]]
- [[Velocity-Command Steering Law]]
- [[P0 - CommandNav]]
