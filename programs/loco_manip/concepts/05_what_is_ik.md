# What Is IK (Inverse Kinematics)?

## The Problem IK Solves

You want the robot's hand to be at position (x=0.5, y=0.2, z=0.8) in 3D space.
But the robot's joints only understand angles: shoulder_pitch, elbow, wrist_roll, etc.

**Forward kinematics (FK):** Given joint angles → compute where the hand ends up.
**Inverse kinematics (IK):** Given where you want the hand → compute the joint angles.

IK solves the reverse problem.

## Why IK for Manipulation?

Without IK (joint-space control):
```
action = [0.3, -0.5, 0.1, 0.8, 0.0, 0.2, -0.1]   # shoulder, elbow, wrist...
```
You'd need to manually figure out what angles put the hand near the block.
The policy has to learn this mapping from scratch — slow and brittle.

With IK (task-space control):
```
action = [0.5, 0.2, 0.8]   # WHERE should the hand be? (x, y, z)
```
The IK solver computes the joint angles automatically.
The policy only needs to learn "move hand toward block" — much simpler.

## How Isaac Lab Uses IK

Isaac Lab's `DifferentialInverseKinematicsActionCfg` does:
1. Policy outputs: `[dx, dy, dz, d_roll, d_pitch, d_yaw]` — 6-dim delta pose
2. IK solver: compute joint angles that achieve this new end-effector pose
3. Joint commands: send computed angles to PD controller

The G1 upper body IK config controls BOTH arms simultaneously:
```
G1_UPPER_BODY_IK_ACTION_CFG:
  left arm:  6-dim delta pose → 7 joint angles
  right arm: 6-dim delta pose → 7 joint angles
  Total action dim: 12
```

## The Reach Envelope

The G1's arm reach (from shoulder to fingertip) is approximately 0.6m.
Objects further than 0.6m can't be grasped even with perfect IK.
This is why the approach phase stops at 0.5m — just inside reliable reach.

## Practical: What the Arm Policy Will Output

In Phase 2, the arm policy will output 6 numbers per arm (or 12 total):
```
[left_Δx, left_Δy, left_Δz, left_Δroll, left_Δpitch, left_Δyaw,
 right_Δx, ...]
```

The IK layer in Isaac Lab converts this to actual joint torques automatically.
You don't need to implement IK yourself.
