# How the Nav Policy Works

## The RL Loop (every 20ms)

The robot lives in a physics simulation. Every 20ms:
1. **Observe** — sensors read 127 numbers describing the robot's state
2. **Act** — neural network (MLP) maps 127 → 37 joint targets
3. **Step** — physics simulates result (torques applied, robot moves)
4. **Reward** — scalar score: did the robot move toward the goal?
5. **Learn** — after 24 steps × 4096 parallel environments, PPO updates weights

## The 127-Dim Observation Breakdown

```
base_lin_vel        3   velocity of the pelvis (x, y, z)
base_ang_vel        3   rotation rate of the pelvis (roll, pitch, yaw)
projected_gravity   3   gravity direction in robot frame — tells if tilting
velocity_commands   3   commanded vx, vy, yaw_rate
joint_pos          37   current joint angles minus default angles
joint_vel          37   current joint velocities
actions            37   last action output — helps policy be smooth
nav_command         4   which marker (one-hot 2D) + relative xy to marker
─────────────────────
TOTAL             127
```

## Why 4096 Environments in Parallel?

Training one robot at 0.02s/step would take years.
Isaac Sim simulates 4096 identical robots simultaneously on one GPU.
Every 0.48 seconds of simulation time = one learning batch.
500 epochs × 24 steps × 4096 envs = ~1.2 billion robot-seconds of experience.

## What "96.28% Success" Means

In the P3 evaluation run:
- 500 episodes were run (not 4096 — just 1 env for eval)
- Each episode: robot spawned at origin, commanded to one of 2 markers
- Success = robot center within 0.5m of the commanded marker after ≤1000 steps
- 481 / 500 episodes succeeded = 96.28%

## Why Arms Stay Near Default

The reward only cared about reaching the target.
Arms at default position are stable — low joint velocity means low reward noise.
PPO converged to "keep arms still" as the lowest-effort strategy.

This is both a feature (stable, arms don't flail while walking) and the limitation
we're fixing: arms need a NEW policy with a NEW reward to learn to pick things up.

## The Neural Network (MLP)

```python
MLP:
  Linear(127, 256) + ELU
  Linear(256, 128) + ELU
  Linear(128, 128) + ELU
  Linear(128, 37)           ← no activation — outputs raw position targets
```

Parameters: 127×256 + 256×128 + 128×128 + 128×37 = ~81K weights.
Inference time: <1ms on CPU, <0.1ms on GPU — negligible compared to physics simulation.
