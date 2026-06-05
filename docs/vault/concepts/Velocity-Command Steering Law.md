---
tags: [concepts, steering, navigation, math]
---

# Velocity-Command Steering Law

## What It Computes

Given: robot's current XY position, robot's current yaw, and target XY position.
Returns: a `(N, 3)` velocity command `[vx, vy, wz]` in the robot's **base frame**.

- `vx` = forward speed (positive = walk forward)
- `vy` = sideways (always 0 — robot doesn't strafe)
- `wz` = turning rate (positive = turn left)

## The Math

```python
def velocity_command_to_target(robot_xy, robot_yaw, target_xy, speed=1.0, yaw_gain=1.0, max_yaw_rate=1.0):
    delta = target_xy - robot_xy                         # vector to target
    desired_heading = torch.atan2(delta[:, 1], delta[:, 0])  # angle to face target
    
    diff = desired_heading - robot_yaw
    yaw_err = torch.atan2(torch.sin(diff), torch.cos(diff))  # wrapped to [-pi, pi]
    
    vx = speed * torch.clamp(torch.cos(yaw_err), min=0.0, max=1.0)
    vy = torch.zeros_like(vx)
    wz = torch.clamp(yaw_gain * yaw_err, min=-max_yaw_rate, max=max_yaw_rate)
    return torch.stack([vx, vy, wz], dim=-1)
```

**Intuition:**
- If the robot is facing directly at the target: `yaw_err=0`, `cos(0)=1` → `vx=speed` (full forward), `wz=0` (no turning).
- If the robot is facing 90° away: `yaw_err=π/2`, `cos(π/2)=0` → `vx=0` (stop walking), `wz=yaw_gain*π/2` (turn hard).
- This ensures the robot turns first, then walks forward once aligned — avoids walking sideways into obstacles.

---

## Parameters Tuned Per Task

| Task | speed | yaw_gain | max_yaw_rate | Notes |
|---|---|---|---|---|
| CommandNav | 1.0 | 0.5 | 1.0 | baseline |
| LangNav | 1.0 | 0.5 | 1.0 | same |
| ObstacleNav | 1.0 | 0.5 | 1.0 | but uses avoiding variant (see below) |
| SeqNav | 1.0 | 0.5 | 1.0 | same as base |

---

## The Obstacle-Avoiding Variant

`velocity_command_to_target_avoiding` adds a potential field: obstacles push the command sideways. Used in ObstacleNav.

```python
def velocity_command_to_target_avoiding(robot_xy, robot_yaw, target_xy, obstacles_xy,
                                         speed=1.0, avoid_radius=1.5, avoid_gain=2.0, ...):
    cmd = velocity_command_to_target(...)  # base steering
    # potential field: sum of repulsion vectors from obstacles within avoid_radius
    diff = robot_xy.unsqueeze(1) - obstacles_xy   # (N, num_obs, 2)
    dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-3)
    inside = dist.squeeze(-1) < avoid_radius
    repulsion = (diff / (dist ** 2)) * inside.unsqueeze(-1)
    cmd[:, :2] += avoid_gain * repulsion.sum(dim=1)
    # renormalize so speed stays bounded
    return cmd
```

---

## Why This Works (instead of just outputting actions directly)

The G1 locomotion policy was pre-trained to track velocity commands accurately. By hijacking the command with a steering law, we're composing with an already-trained skill. We don't need to learn locomotion from scratch — we just need to learn *where* to point the command.

---

## What Happens When Command Magnitude is Low (~0.25 m/s)

This is the [[SeqNav Stand-Still Local Optimum]] failure mode. When `vx` averages only 0.25 m/s instead of ~1.0 m/s, the robot barely moves. The cause is the robot continuously misaligned to its target — it turns (wz is non-zero) but rarely faces the target long enough to get full forward speed. This was a **training bootstrap problem**, not a bug in the steering law.

---

## Related

- [[Command-Conditioned Navigation]]
- [[SeqNav Stand-Still Local Optimum]]
- [programs/common/commands.py](../../programs/common/commands.py)
