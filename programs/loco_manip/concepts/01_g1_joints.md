# G1 Joint Architecture

## The G1 Has 29 Degrees of Freedom (DOF)

A "degree of freedom" is one independently movable joint. The G1's 29 joints are split into four groups:

```
LEGS (12 joints, bilateral — same 6 joints on each side):
  left_hip_pitch_joint    ← hip swing forward/backward (walking)
  left_hip_roll_joint     ← hip tilt side-to-side (lateral balance)
  left_hip_yaw_joint      ← hip rotation around vertical axis (turning)
  left_knee_joint         ← knee bend
  left_ankle_pitch_joint  ← ankle forward/backward (tiptoeing, push-off)
  left_ankle_roll_joint   ← ankle lateral tilt (balance on uneven ground)
  right_* (same 6 joints)

TORSO (3 joints):
  torso_joint             ← upper body twist left/right
  waist_yaw_joint         ← waist rotation
  waist_roll_joint        ← waist lateral lean

ARMS (14 joints, bilateral — 7 per side):
  left_shoulder_pitch_joint  ← arm swing forward/back
  left_shoulder_roll_joint   ← arm raise sideways (like a T-pose)
  left_shoulder_yaw_joint    ← upper arm rotation
  left_elbow_joint           ← elbow bend
  left_wrist_roll_joint      ← wrist roll (twist forearm)
  left_wrist_pitch_joint     ← wrist up/down flex
  left_wrist_yaw_joint       ← wrist side-to-side deviation
  right_* (same 7 joints)
```

## Why model_499.pt Has 37 Actions With Only 29 Joints

The Isaac Lab G1 env used for nav training included extra hand/finger joints on top of the 29 DOF.
The exact 37 = 12 (legs) + 3 (torso) + 14 (arms) + 8 (hands/fingers).

Run `programs/loco_manip/audit_joints.py` on Lightning AI to see the exact layer shapes.

## What "Position Target" Means

The nav policy outputs 37 **position targets** — not forces or velocities.
Each target says: "joint X should be at angle Y."
A PD (Proportional-Derivative) controller on the robot converts this to torque:

```
torque = kp × (target_angle - current_angle) + kd × (0 - current_velocity)
```

- `kp` = spring stiffness (how hard to push toward target)
- `kd` = damping (how much to resist velocity)

The policy never sees torques — it only outputs desired angles.

## The Key Insight for Loco-Manipulation

The nav policy outputs targets for ALL 37 joints every step.
For leg joints: the targets encode a walking gait.
For arm joints: the targets are always near the default "arms-at-side" pose.

The arms weren't frozen — the policy just never got a reward for moving them.
That's the bug we fix in Phase 2: train a second policy that DOES get a reward for moving arms.
