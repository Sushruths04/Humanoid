# Wakeboard Start — Complete System Reference

How the real sport works, how we simulate it, and exactly what is implemented.

---

## Part 1 — Real Wakeboarding Physics

### The Deep-Water Start

A wakeboard deep-water start is one of the hardest beginner skills. The rider starts floating in water, completely submerged from the chest down, in a tight cannonball. A boat then accelerates and literally drags them out of the water onto the board.

**What the rider looks like at the start:**
- Floating in water, only head above surface
- Knees pulled tight to chest (cannonball crouch)
- Board is horizontal in front of them, slightly nose-up (~15°)
- Both feet strapped into bindings on the board
- Arms extended forward holding the rope handle at hip height
- Body is SIDEWAYS to the direction of travel — left shoulder toward the boat (regular stance) or right shoulder (goofy)

**The rope and boat:**
- Rope length: ~23 m (75 ft) — this matters because a longer rope = gentler pull ramp-up
- Rope attaches to a PYLON on the boat (a vertical pole), NOT a pulley. One straight cable.
- The pylon is elevated (~1m above water), so the rope pulls at a slight upward angle (~5-10°)
- This upward angle is critical: it lifts the rider's hands, which tilts the board nose-up, which makes the board plane on the surface instead of diving
- Max tension during start: ~500-700 N (roughly the weight of a person)
- Boat reaches ~25-30 km/h during the start phase

**The physics of the pull:**
1. Rope goes taut → tension yanks the handle forward and slightly upward
2. Rider's arms are straight → tension transfers through arms to torso
3. Rider's heels are weighted → board nose tips up (positive pitch)
4. Board hits water at a positive angle → hydrodynamic lift pushes board up to the surface
5. As board planes up, rider gradually extends legs → stands up
6. Rope is TENSION ONLY — it cannot push. If rider leans too far forward, rope goes slack → face-plant

**What goes wrong for beginners:**
- Standing up too fast → weight shifts forward → board dives → face-plant
- Bending elbows to "pull themselves up" → fighting the boat → thrown forward → face-plant
- Board nose down (heels not weighted) → board dives under water → face-plant
- Leaning too far back → board goes vertical → falls backward

**What a correct start looks like:**
1. Rope tightens → stay crouched, arms straight, let the rope do the work
2. Board pitches up 10-20° → rider feels the board start to plane (rise to surface)
3. Gradually extend knees while keeping handle at hips, slight backward lean
4. After ~3-5 seconds → standing in a balanced crouch riding at 25-30 km/h

---

## Part 2 — How We Simulate This

### The "Sand Instead of Water" Abstraction

Real water physics (buoyancy, hydrodynamic lift, drag) is extremely complex. We use a simpler approximation:

- **Water surface → frictional ground plane** (μ = 0.4, sand-like)
- **Board planing on water → board sliding on ground** under rope pull
- **Hydrodynamic lift → not modeled** (board just slides; robot rises by extending legs)
- **Buoyancy → not modeled** (robot stands on board from the start)

This loses realism but keeps the RL problem tractable. The robot still has to learn the same muscle coordination: stay crouched, arms straight, handle at hips, lean back, rise gradually.

### Coordinate System

```
+X = direction of travel (rope pulls this way)
+Y = sideways (direction the rider FACES)
+Z = up

Robot yaw = 90° (faces +Y, sideways to travel) ← correct wakeboard stance
Head faces +Y  ← NOTE: head should face +X toward the boat, but G1 neck
                         may be fixed; this is a known open issue for when
                         vision / human-coaching comparison is added
```

### The Robot: Unitree G1

- 23 degrees of freedom (DoF)
- Arms are ACTUATED (unlike locomotion tasks where arms are often frozen)
  - The robot must hold the handle → arms must be free to move
- Both feet are rigidly welded to the board (simulates foot bindings)
- Relevant joints:
  - `left/right_hip_pitch_joint` — forward/backward hip flex
  - `left/right_knee_joint` — knee bend
  - `left/right_ankle_pitch_joint` — foot angle
  - `left/right_shoulder_pitch_joint` — arm forward/backward
  - `left/right_elbow_pitch_joint` — elbow bend (should stay near 0 = straight)
  - `torso_joint` — torso lean

---

## Part 3 — The Initial Position (Cannonball Pose)

This is the most critical part to get right. If the initial pose is wrong, physics explodes (NaN crashes) and training learns nothing.

### What the Cannonball Pose Is

```
CANNONBALL_ROOT_Z = 0.50 m     ← pelvis height above ground
Robot yaw = 90°                 ← facing +Y (sideways to travel)

Joint angles at spawn:
  hip_pitch      = -0.8 rad   (deep forward flex — knees to chest)
  knee           = +1.4 rad   (deep bend)
  ankle_pitch    = +0.3 rad   (feet relatively flat)
  shoulder_pitch = +0.9 rad   (arms extended forward toward handle)
  elbow_pitch    = +1.0 rad   (elbows bent to hold handle initially)
  torso          = -0.3 rad   (torso reclined backward)
```

### The Board

```
Size: 0.4m (X) × 1.4m (Y) × 0.04m (Z)
  - Long axis along Y (same direction rider faces — perpendicular to travel)
  - Short axis along X (direction of travel)
  - This is correct: real wakeboards go ACROSS the direction of travel

Mass: 3 kg
Friction: μ = 0.4 (sand approximation)
Position: center at Z = 0.02m (sits flat on ground, top surface at Z = 0.04m)
```

### The Foot-Board Weld

At spawn time, a fixed joint is created between each ankle roll link and the board:
- `left_ankle_roll_link` → Board
- `right_ankle_roll_link` → Board

The joint anchor is offset +0.04m upward in the ankle link's local frame. This ensures the ankle mesh BOTTOM (not the joint origin) aligns with the board TOP surface, preventing the feet from sinking visually into the board.

**IMPORTANT**: The weld is created ONCE at spawn. Every reset uses `write_root_pose_to_sim` to teleport the robot back to the cannonball pose. The weld must match the spawn pose exactly — if reset pose ≠ spawn pose, PhysX will violently snap the bodies and create NaN explosions. This is why spawn pose = reset pose = same CANNONBALL_JOINT_POS values.

### Known Open Issue: Head Direction

After the 90° yaw rotation, the robot's HEAD faces +Y (sideways) instead of +X (toward the boat). In real wakeboarding the rider's head turns to look forward toward the boat while the torso stays sideways. The G1's neck/head joints may be fixed, so this cannot be corrected without adding a separate head yaw target. **Deferred until vision observations or human-coaching comparison is added.**

---

## Part 4 — The Rope Model

### What We Have

A software-only spring model (no physical rope in the USD scene):

```python
model = "spring"  # default
v_pull = 30 km/h = 8.33 m/s   # boat speed
k_p    = 800 N/m               # spring stiffness
k_d    = 50 N·s/m              # damping
f_max  = 600 N                 # max force (realistic rope tension cap)
pull_dir = (1.0, 0.0, 0.0)    # purely horizontal +X
```

**How it works each step:**
1. A virtual anchor moves forward along +X at `v_pull`
2. Force = `k_p*(anchor - handle) + k_d*(anchor_vel - handle_vel)`, clamped to 600N
3. Force is split equally across both palm links (left + right hands)

### What's Missing vs Real Physics

| Real rope | Our model | Impact |
|---|---|---|
| ~5-10° upward pull angle (pylon elevated) | Purely horizontal (0°) | Robot gets no upward lift assist — harder to learn |
| Tension ONLY (goes slack if rider gets ahead) | Spring can push backward | Slightly unphysical if robot overshoots |
| 23m rope length (gradual tension ramp) | No length limit | Tension ramps up too fast at episode start |

The upward pull angle is the most impactful gap. Real riders rely on this lift to tip the board nose-up. Without it, the board pitch reward is harder to achieve.

---

## Part 5 — The Rewards (5 Biomechanical Rules)

Every reward term maps directly to a real coaching rule:

| Rule | What it means | Reward term | Weight |
|---|---|---|---|
| #1 Stay crouched, rise gradually | Never jump to your feet | `height_progress` (phase-gated) | 1.5 |
| #2 Board nose up 10-20° | Weight heels, let board plane | `board_positive_angle` | 1.5 |
| #3 Arms straight, handle at hips | Let rope tow you, don't pull | `arms_straight` + `handle_at_hips` | 1.0 + 0.8 |
| #4 Don't pull against the rope | Bending elbows = face-plant | `pen_pull_against_rope` | -1.0 |
| #5 Lean slightly back, knees soft | Balance, never lock knees | `lean_back_moderate` + `knee_bend_maintained` | 0.7 + 0.8 |

Plus:
- `pelvis_height` (2.0) — reward being upright
- `uprightness` (2.0) — penalize tilting over
- `forward_glide` (1.0) — board speed tracks boat speed
- `success_bonus` (50.0) — sparse: stable ride for 1.5+ seconds
- `survival` (0.5) — reward staying alive each step
- Various action smoothness penalties

### Success Condition

Episode is a success when ALL hold for 1.5+ consecutive seconds:
- Pelvis height ≥ 0.55m (robot is standing/crouching, not collapsed)
- Uprightness ≥ 0.85 (torso near vertical)
- Board pitch < 30° (not tipped too far forward or back)

### Termination Conditions

Episode ends early if:
- **Timeout**: 8 seconds elapsed
- **Board out of range**: board pitch < -40° or > 60°
- **Fell over**: uprightness < 0.3 (torso nearly horizontal)

---

## Part 6 — The Training Pipeline

### Phase Structure (Curriculum)

```
Stage 1: v_pull = 10 km/h   ← start slow, easier to stand up
  ↓ (when success rate > 60%)
Stage 2: v_pull = 20 km/h
  ↓ (when success rate > 60%)
Stage 3: v_pull = 30 km/h   ← real boat speed
```

### Algorithm

PPO (Proximal Policy Optimization) via RSL-RL:
- Policy: MLP 512→256→128, ELU activations
- 4096 parallel environments
- 200 steps per env per iteration
- Resilient training loop: saves checkpoint every 50 iters, recovers from NaN crashes

### Observation Space (what the robot sees)

```
joint_pos (23)     — current joint angles relative to default
joint_vel (23)     — joint velocities
base_ang_vel (3)   — pelvis angular velocity
proj_gravity (3)   — which way is down (in robot's local frame)
last_action (23)   — previous action (for smoothness)
board_pitch (1)    — current board nose angle
rope_force (3)     — current force vector being applied at hands
handle_rel (3)     — handle position relative to pelvis
v_pull (1)         — current boat speed (curriculum input)
phase (1)          — how far through the episode we are (0→1)
Total: 81 dims
```

---

## Part 7 — What Is and Isn't Implemented

### Implemented and Working
- G1 robot with actuated arms
- Cannonball initial pose (spawn = reset = same pose, prevents NaN)
- Board as rigid body, welded to both feet
- Rope spring model applying force to both hands
- All 5 biomechanical reward rules
- Curriculum over pull speed
- Resilient training loop with NaN recovery
- Robot sideways stance (90° yaw) ← fixed 2026-06-22

### Fixed Recently (2026-06-22)
- Robot now spawns facing +Y (sideways to travel) — was incorrectly facing +X
- Board long axis now along Y — was incorrectly along X
- Pelvis height adjusted 0.55→0.50 so feet land on board surface
- Ankle weld anchor +0.04m offset so feet don't sink into board

### Known Gaps (deferred)
- Head faces +Y instead of +X toward boat (no visual/physics impact for now)
- Rope pull is purely horizontal — missing the ~8° upward lift angle from boat pylon
- No rope slack detection (spring can push backward, unphysical)
- No hydrodynamic lift (board doesn't actually plane — sand friction approximation)
- No buoyancy (robot sits on board from start, not floating in water)

---

## Quick Reference: Key File Locations

```
src/tasks/wakeboard_start_cfg.py   ← main env config, initial pose, weld, reset
src/rope_model.py                  ← rope spring model
src/board.py                       ← board rigid body config
src/rewards/wakeboard_rewards.py   ← all reward functions
configs/stage1.yaml                ← training config (PPO hyperparams, weights)
train.py                           ← training entry point
```
