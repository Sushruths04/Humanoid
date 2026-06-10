# State Machines in RL

## What Is a State Machine?

A state machine is a system that is always in exactly ONE of a fixed set of states.
It transitions between states based on conditions you define.

For loco-manipulation:
```
States:   WALK → APPROACH → GRASP → PLACE → DONE
Triggers: distance to object
          object height (lifted?)
```

## Why Use a State Machine for Multi-Policy Control?

You have two neural networks:
- Nav policy: knows how to walk
- Arm policy: knows how to grab

Neither knows about the other. The state machine is the "conductor" that decides
which one runs at each timestep.

```
if state == WALK:
    action[:12] = nav_policy(obs)   # leg joints
    action[15:] = default_arm_pose  # arm joints stay still
elif state == GRASP:
    action[:12] = hold_stance       # leg joints hold position
    action[15:] = arm_policy(obs)   # arm joints reach for object
```

## The Four Transitions

```
WALK ──(dist < 1.5m)──► APPROACH
                              │
                         (dist < 0.5m)
                              │
                              ▼
                           GRASP ──(obj_height > 0.2m)──► PLACE ──► DONE
```

- **1.5m**: "getting close" — robot slows down, starts orienting toward object
- **0.5m**: "within arm's reach" — legs stop, arms take over
- **0.2m height**: "object is lifted" — transition to placing phase

These thresholds were chosen based on G1 arm reach (~0.6m extended) and typical table height.

## Why Not Just Use One Policy?

Training one policy that does BOTH walking and grasping requires:
1. A single observation space covering both tasks (127 nav + ~50 manip = 177 dims)
2. A reward function that balances locomotion and manipulation (hard to tune)
3. ~10× more training time (curriculum: first learn to walk, then add arms)

The state machine approach reuses two already-trained policies.
It's not as elegant as one unified policy, but it works TODAY with zero additional training.
