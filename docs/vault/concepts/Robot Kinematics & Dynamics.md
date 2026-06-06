---
tags: [concepts, kinematics, dynamics, robotics, interview, theory]
---

# Robot Kinematics & Dynamics — Interview Reference

Everything from this project that maps to classical robotics theory. Use for interviews.

---

## 1. Forward Kinematics (FK)

**What it is:** Given joint angles θ = [θ₁, θ₂, ..., θₙ], compute the end-effector pose (position + orientation) in world frame.

**In this project (Navigation):**
The G1 humanoid's **root position** is directly in world frame (Isaac Lab tracks it). We never explicitly compute FK for the full chain — we use Isaac Lab's `asset.data.root_pos_w` directly.

**In this project (Manipulation — GR00T):**
LIBERO gives us `robot0_eef_pos` (3D position of Franka end-effector) directly. This IS the result of running FK on Franka's 7-DOF chain. We receive the final result, not the intermediate computation.

**Mathematical formulation:**
```
T_ee = T₀₁(θ₁) · T₁₂(θ₂) · ... · Tₙ₋₁,ₙ(θₙ)

where T_i,i+1 = DH transform matrix:
    [ cos(θ)  -sin(θ)cos(α)   sin(θ)sin(α)   a·cos(θ) ]
    [ sin(θ)   cos(θ)cos(α)  -cos(θ)sin(α)   a·sin(θ) ]
    [   0       sin(α)         cos(α)           d       ]
    [   0         0              0               1       ]
```

**Interview angle:** "FK is a closed-form chain multiplication. It's always solvable. The hard part is IK."

---

## 2. Inverse Kinematics (IK)

**What it is:** Given target end-effector pose, find joint angles. The inverse of FK.

**In this project:**
We don't solve IK directly. GR00T's action space is **OSC delta** (Operational Space Control) — it outputs Cartesian velocity/force targets. Isaac Lab / MuJoCo converts these to joint torques using Jacobian-based control internally.

**Why IK is hard:**
- Non-linear (many solutions or no solution)
- For redundant robots (7-DOF Franka, 37-DOF G1): infinitely many solutions
- Numerical methods: damped-least-squares Jacobian pseudoinverse

**Key formula:**
```
θ̇ = J⁺ ẋ_ee + (I - J⁺J) z

where:
  J⁺ = Jᵀ(JJᵀ + λI)⁻¹  (damped pseudoinverse)
  J⁺ẋ_ee = minimum-norm solution
  (I - J⁺J)z = null-space motion (redundancy resolution)
```

**Interview angle:** "We avoided explicit IK by using OSC — the controller handles it internally using Jacobian pseudoinverse."

---

## 3. Jacobian Matrix

**What it is:** J maps joint velocities to end-effector velocity:
```
ẋ_ee = J(θ) · θ̇
```

**Dimensions:** For Franka (7-DOF), J is 6×7 (6 = 3 position + 3 orientation, 7 = joints).

**In this project:**
OSC (Operational Space Control) uses the Jacobian internally. GR00T outputs 7-dim OSC delta actions. The LIBERO simulator converts these to joint torques via:
```
τ = Jᵀ F + (I - Jᵀ(Jᵀ)⁺) τ_null
```

**Singularities:** When J loses rank (arm fully extended, etc.), Jacobian-based control breaks. Damping (λI) prevents this.

---

## 4. Center of Mass (CoM) & Upright Stability

**What it is:** CoM = weighted average position of all body links:
```
r_CoM = Σ(mᵢ rᵢ) / Σmᵢ
```

**In this project — Upright Reward:**
We don't compute CoM explicitly. Instead, we use the **root orientation quaternion** as a proxy for whether the humanoid is upright:

```python
def upright_reward(root_quat_w: torch.Tensor) -> torch.Tensor:
    x, y = root_quat_w[:, 1], root_quat_w[:, 2]
    up_z = 1.0 - 2.0 * (x.pow(2) + y.pow(2))
    return up_z.clamp(min=0.0)
```

**Mathematical derivation:**
For quaternion [w, x, y, z], the z-component of the "up" unit vector after rotation is:
```
up_z = 1 - 2(x² + y²)
```
This equals `cos²(tilt_angle)` approximately. When perfectly upright: up_z = 1.0. When tilted 90°: up_z = 0.0.

**Why this works:** G1 humanoid falls if CoM projects outside support polygon. Keeping the root orientation upright (up_z ≈ 1.0) prevents this without needing to compute actual CoM.

**The P0-stable fix:** We found fall rate was 28.1% when `upright_reward_weight = 0.0`. Setting it to `0.5` reduced falls to **7.8%**. This is because the upright reward provides a continuous gradient signal against tilting, which is the precursor to falling.

---

## 5. Degrees of Freedom & Redundancy

**G1 Humanoid:**
- 37 DOF total (joint position targets)
- Legs (12), arms (14), waist (3), hands (8)
- For navigation: most DOFs are "redundant" for the end-goal (reaching waypoints)

**Franka Panda (LIBERO manipulation):**
- 7-DOF arm + 2-DOF gripper = 9 total
- For end-effector control: 6 DOF needed (3 position + 3 orientation)
- 1 DOF is redundant → null-space motion possible

**How we handled redundancy:**
For G1 navigation, we don't resolve redundancy explicitly. Instead:
- **Observation space** is 45-dim (velocity commands + joint pos/vel + gravity projection + base lin/ang vel)
- PPO policy outputs 37-dim joint **position targets** (not joint torques)
- Isaac Lab's built-in PD controller converts position targets to torques: `τ = kp(θ_target - θ) - kd(θ̇)`
- The policy learns which joints to use for locomotion vs. upright stability implicitly

**Interview angle:** "We reduced effective complexity by using joint position targets + PD controller rather than joint torques directly. The PD layer absorbs low-level redundancy."

---

## 6. Contact Dynamics & Foot Contact

**Why it matters for humanoid locomotion:**
The G1 walks — this requires managing ground contact forces. In RL, the policy must learn gait patterns that maintain stable contact.

**Isaac Lab handles:**
- Rigid body contact simulation (PhysX)
- Contact impulse computation
- Ground reaction forces

**Our observation space includes:**
- `projected_gravity` (3D vector) — tells policy how gravity is oriented relative to robot → captures body tilt
- `base_lin_vel` + `base_ang_vel` — root velocity in robot frame
- `joint_pos` + `joint_vel` — all 37 joints

**What the policy learns from these signals:**
- When projected_gravity[2] ≈ -1: upright
- When projected_gravity deviates: tilting → upright reward kicks in
- The policy implicitly learns heel-toe contact patterns through RL

---

## 7. Euler Angles vs. Quaternions vs. Axis-Angle

**Three representations of rotation:**

| Format | Dims | Pros | Cons |
|---|---|---|---|
| Euler (roll/pitch/yaw) | 3 | Human-readable | Gimbal lock at ±90° pitch |
| Quaternion [w,x,y,z] | 4 | No gimbal lock, smooth interpolation | Non-intuitive, requires normalization |
| Axis-angle | 3 | Compact, differentiable | Discontinuity at 2π |

**In this project (GR00T bug F-11):**
GR00T was trained with **axis-angle** rotation in observations. We initially passed Euler angles from scipy:
```python
# WRONG
euler = Rotation.from_quat(xyzw).as_euler("xyz")  # 0% success

# CORRECT  
den = np.sqrt(1.0 - quat[3]**2)
rpy = (quat[:3] * 2.0 * np.arccos(quat[3])) / den  # 97% success
```
This single bug caused 0% task success. **Rotation convention mismatch is one of the most common robot ML bugs.**

**Upright reward uses quaternion:**
`up_z = 1 - 2(x² + y²)` — direct quaternion math, no conversion needed.

---

## 8. OSC — Operational Space Control

**What it is:** Control in end-effector (task) space rather than joint space.

**GR00T action format:** 7-dim = [dx, dy, dz, droll, dpitch, dyaw, gripper]
These are **OSC delta targets** — desired displacement in end-effector frame.

**LIBERO converts to torques via:**
```
F_ee = K_p · (x_desired - x_current) + K_d · ẋ_ee
τ = Jᵀ F_ee
```

**Why OSC for manipulation:**
1. Language-conditioned actions are more natural in task space ("move right 2cm")
2. Avoids joint-space singularities for manipulation
3. More transferable across robot morphologies (same δx means same thing)

---

## 9. Parallel Environments & Sample Efficiency

**Isaac Lab key insight:** Run 4096 parallel environments simultaneously.

**Effect on training:**
- 4096 envs × 24 steps per update = 98,304 transitions per PPO iteration
- Each PPO update uses all 4096 rollouts simultaneously
- Wall-clock training time: ~20 min despite millions of samples collected

**Why this matters:**
- Single env RL for humanoid walking: would take days
- 4096 envs: 20 minutes
- This is the key advantage of GPU-accelerated simulation (Isaac Lab / MuJoCo MJX)

**Interview angle:** "Massively parallel simulation is what makes modern robot RL practical. Without it, 37-DOF humanoid locomotion learning would be computationally infeasible in a research setting."

---

## 10. PD Controller (Joint-Level)

**Formula:**
```
τ_joint = kp × (θ_target - θ_current) - kd × θ̇_current
```

**Isaac Lab defaults for G1:**
- kp = 200 (stiffness — how hard to reach target)
- kd = 10 (damping — how much to resist velocity)

**Role in our system:**
PPO policy → joint position targets (37-dim) → PD controller → joint torques → physics simulation

This abstraction keeps the policy's action space simple (position targets) while handling the low-level torque control automatically.

---

## Related

- [[Reward Shaping & Progress Rewards]]
- [[Reward Engineering Deep Dive]]
- [[PPO with RSL-RL]]
- [[Interview Prep - Master Guide]]
