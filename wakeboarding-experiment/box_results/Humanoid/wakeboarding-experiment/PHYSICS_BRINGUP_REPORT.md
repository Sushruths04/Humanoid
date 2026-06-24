# Wakeboarding Physics Bring-up Report

Date: 2026-06-20
Branch: gpu-l4-bringup
GPU: NVIDIA L4
Scope: Section 1 physics verification and fixes only. Stage I and Stage II training were not started.

## Summary

The original smoke run was misleading. The pipeline trained, but the rope force was not actually being applied to the robot, and the board was only a free rigid object near the robot rather than a real foot-bound wakeboard.

Final state after this bring-up:

- Rope force lands: yes.
- Board is bound to the feet: yes, via two USD fixed joints per environment.
- Smoke probe runs 30 iterations on L4 without traceback.
- No Stage I or Stage II training was launched.

Final commits made:

- be66b8a Set smoke probe to 30 iterations
- 9e1d1c8 Apply rope force in manager env step
- 11efc26 Bind wakeboard to G1 feet

## Original Probe Result

The first real 30-iteration probe used configs/smoke.yaml with max_iterations: 30 and was launched through:

```bash
cd ~/Humanoid/wakeboarding-experiment
setsid bash docker/run.sh train smoke > ~/_setup_logs/probe.log 2>&1 < /dev/null & disown
```

Important runner detail: docker/run.sh smoke calls scripts/00_smoke.sh, which overrides the YAML with its own MAX_ITERS default of 2. To respect the edited YAML, use docker/run.sh train smoke.

Before fixes, the final table showed:

```text
Episode_Reward/forward_glide: 0.0000
Episode_Reward/board_positive_angle: 0.0013
Episode_Reward/pen_pull_against_rope: 0.0000
Episode_Termination/board_range: 0.0000
Episode_Termination/fell: 1.0000
```

There was no traceback. forward_glide flickered at tiny non-zero values earlier, but pen_pull_against_rope stayed exactly zero. That was not convincing evidence of rope physics.

## Rope Failure

### Root Cause

WakeboardStartEnv inherits from ManagerBasedRLEnv, but the rope force code was originally placed in _pre_physics_step. That hook belongs to Isaac Lab direct environments. ManagerBasedRLEnv.step does not call _pre_physics_step, so _apply_handle_force was never reached during the training loop.

There was a second issue in _apply_handle_force: set_external_force_and_torque was wrapped in a bare except/pass, so any Isaac Lab API mismatch would have been hidden.

The original force shape was also wrong for two hand bodies. It passed shape (N, 1, 3) while body_ids contained two palm links. Isaac Lab 5.1 expects (num_envs, num_bodies, 3) matching len(body_ids).

### Failed Attempt

First attempt: only fix _apply_handle_force by removing the silent exception, splitting force across both hands, and using is_global=True. The run still did not print the diagnostic line and pen_pull_against_rope stayed zero. That proved _apply_handle_force was not in the active step path.

### Final Rope Fix

The rope update was moved into an override of ManagerBasedRLEnv.step before super().step(action), because super().step writes scene data to sim after action processing. This ensures the external wrench buffer is populated before Isaac steps physics.

The final rope application:

- resolves palm links as [28, 29], corresponding to left and right palm links;
- computes the rope force in world coordinates;
- expands it to (N, 2, 3);
- splits force across both palms;
- uses is_global=True;
- raises on errors instead of swallowing them;
- prints one confirmation line.

Evidence from fixed probe:

```text
[wakeboard] rope force configured: force_shape=(16, 2, 3) body_ids=[28, 29] mean_norm=600.000
```

Final rope-fix probe table before board binding:

```text
Learning iteration 29/30
Episode_Reward/forward_glide: 0.0054
Episode_Reward/board_positive_angle: 0.0059
Episode_Reward/pen_pull_against_rope: -0.0293
Episode_Termination/board_range: 0.1562
Episode_Termination/fell: 0.8438
```

Interpretation: rope force was now landing physically. The non-zero pen_pull_against_rope is the key signal.

## Board Binding Failure

### Root Cause

The repo had comments saying the board should be welded to the feet, but no implementation. src/board.py only spawned a standalone RigidObjectCfg at {ENV_REGEX_NS}/Board. There was no _bind_feet_to_board implementation despite comments pointing to it.

This meant the board could appear stable in smoke metrics while not actually being constrained to the robot.

### Live USD Findings

A one-env Isaac diagnostic showed these relevant prims:

```text
/World/envs/env_0/Board                         Xform, RigidBodyAPI=true
/World/envs/env_0/Robot/left_ankle_roll_link    Xform, RigidBodyAPI=true
/World/envs/env_0/Robot/right_ankle_roll_link   Xform, RigidBodyAPI=true
```

It also showed a board placement issue:

```text
Board root z: about 0.05
Ankle-roll link z: about 0.02 to 0.03
```

The board was spawned above the foot binding frame.

### Failed Board Attempt

First attempt: create fixed joints without changing board spawn height. PhysX emitted warnings like:

```text
PhysicsUSD: CreateJoint - found a joint with disjointed body transforms, the simulation will most likely snap objects together
```

The smoke probe got worse immediately, with board_range around 0.75 to 0.85. That attempt was stopped and removed. It was not committed.

### Final Board Fix

The board spawn height was changed from hard-coded z=0.05 to:

```python
p.thickness * 0.5
```

With the default thickness of 0.04, this places the board center at z=0.02, aligned with the ankle-roll link frame and with the board bottom on the ground plane.

Then _bind_feet_to_board was added. It creates two UsdPhysics.FixedJoint prims per environment:

- left ankle-roll link to board;
- right ankle-roll link to board.

It computes the joint frame using live USD transforms:

- localPos0 = (0, 0, 0) in the foot link frame;
- localPos1 = board_inv.Transform(foot_world) in the board frame.

Evidence from fixed board probe:

```text
[wakeboard] board fixed joints configured: count=32
[wakeboard] rope force configured: force_shape=(16, 2, 3) body_ids=[28, 29] mean_norm=600.000
```

For 16 envs, 32 joints means 2 joints per env.

The final board-binding probe had no Traceback and no CreateJoint snap warnings. A later clean verification run also showed board_range staying at zero through the final iteration.

Final clean verification table:

```text
Learning iteration 29/30
Episode_Reward/pelvis_height: 0.0647
Episode_Reward/height_progress: 0.0000
Episode_Reward/uprightness: 0.0427
Episode_Reward/survival: 0.0173
Episode_Reward/forward_glide: 0.0132
Episode_Reward/success_bonus: 0.0000
Episode_Reward/board_positive_angle: 0.0299
Episode_Reward/arms_straight: 0.0023
Episode_Reward/handle_at_hips: 0.0137
Episode_Reward/lean_back_moderate: 0.0013
Episode_Reward/knee_bend_maintained: 0.0277
Episode_Reward/pen_stand_too_fast: -0.0028
Episode_Reward/pen_pull_against_rope: -0.0345
Episode_Reward/pen_torque: -2.3423
Episode_Reward/pen_action_rate: -0.0572
Episode_Reward/pen_action_accel: -0.0057
Episode_Reward/pen_dof_pos_limits: -0.2216
Episode_Termination/timeout: 0.0000
Episode_Termination/board_range: 0.0000
Episode_Termination/fell: 1.0000
```

## How to Interpret board_range

After the board is actually fixed to the feet, board_range is high under a random/untrained policy. This is expected: the board now follows the ankle motion and pitch during falls instead of remaining as an independent stable block.

So board_range not equal to zero after the weld is not automatically a weld failure. In fact, the earlier board_range equal to zero was suspicious because the board was not constrained to the robot.

The stronger indicators that the binding exists are:

- fixed joint creation count equals 2 * num_envs;
- no PhysX CreateJoint disjoint-frame warnings;
- board_positive_angle is non-zero under tow;
- board range/fall dynamics change once rope and board are real.

## Commands Used

Typical verification command:

```bash
cd ~/Humanoid/wakeboarding-experiment
rm -f ~/_setup_logs/board_fix_probe.log
setsid bash docker/run.sh train smoke > ~/_setup_logs/board_fix_probe.log 2>&1 < /dev/null & disown
```

Important: docker/run.sh train smoke reads configs/smoke.yaml. docker/run.sh smoke goes through scripts/00_smoke.sh, which still has its own MAX_ITERS override path.

## Current Status

Committed and verified:

- Smoke probe runs for 30 iterations.
- Rope force is applied in the actual ManagerBasedRLEnv.step path.
- Rope force is split across both palm links with the correct shape and global frame.
- Board is spawned at foot-link height instead of above the feet.
- Board fixed joints are created for both feet in every environment.
- Latest smoke probe had no traceback, no fixed-joint snap warnings, and final board_range was 0.0000.

Not done:

- Stage I training was not started.
- Stage II training was not started.
- Reset pose is still stock reset_scene_to_default, not the planned cannonball crouch.
- The external force API emits Isaac Lab deprecation warnings recommending permanent_wrench_composer.set_forces_and_torques; current API still works.

## Recommendation

The physics is now real enough to begin a short Stage I sanity run, but expect very poor early rewards and high board_range/fell rates because the board and rope are now actually coupled to the robot. If Stage I still fails to improve, the next highest-leverage fix is not more rope/board plumbing; it is replacing the default reset pose with the planned crouched wakeboard start pose.
