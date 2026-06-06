---
tags: [reference, interview, theory, robotics, rl, manipulation, concepts]
---

# Interview Prep — Master Guide

All interview topics from this project. Organised by domain. Each section: what we did → the theory behind it → likely interview questions.

---

## 1. Reinforcement Learning Fundamentals

### What we did
Trained G1 humanoid locomotion + navigation using PPO (Proximal Policy Optimization) via RSL-RL in Isaac Lab, 4096 parallel environments, ~20 minutes per task.

### Theory

**MDP (Markov Decision Process):**
- State s, Action a, Reward r, Transition P(s'|s,a), Discount γ
- Policy π(a|s) → maximise E[Σ γᵗ rₜ]

**PPO key idea:**
```
L_CLIP = E[min(r(θ) Â, clip(r(θ), 1-ε, 1+ε) Â)]
where r(θ) = π_θ(a|s) / π_θ_old(a|s)
```
The clip prevents large policy updates that might destabilise training. ε=0.2 typical.

**Advantage estimation (GAE):**
```
Âₜ = δₜ + (γλ)δₜ₊₁ + (γλ)²δₜ₊₂ + ...
where δₜ = rₜ + γV(sₜ₊₁) - V(sₜ)
```
GAE (λ=0.95 typically) balances bias vs. variance in advantage estimates.

**Value function:** V(s) = expected cumulative reward from state s. PPO trains both policy π and value function V simultaneously.

### Interview Q&A

**Q: Why PPO over other algorithms?**
A: PPO is the go-to for continuous control: stable training (clip prevents large updates), on-policy (good for sim-to-real where we care about current policy distribution), and well-tuned defaults that work across many robot tasks. We used RSL-RL's implementation which is optimised for Isaac Lab's batched environments.

**Q: What is the curse of dimensionality in RL?**
A: State and action spaces grow exponentially with dimensions. G1 has 37-DOF = 37-dim action space. We handled this by: (1) using position targets instead of torques (PD controller absorbs 1 level), (2) shared policy across all 4096 envs (massive parallelism), (3) dense rewards to guide exploration.

**Q: What's the difference between model-free and model-based RL?**
A: Model-free (PPO) learns a policy directly from experience. Model-based (Dreamer-mini, P2/T2) learns a world model first, then plans using imagined rollouts. Our P2/T2 used Dreamer-mini RSSM — 3000 steps to train, vs. PPO which needed millions of environment steps.

---

## 2. Reward Engineering

### What we did
Designed 3 reward terms for navigation: (1) Euclidean progress, (2) quaternion-based upright, (3) base locomotion from Isaac Lab. Added collision penalty for ObstacleNav.

### Key formulas

```
r_progress = (||p_prev - p_target|| - ||p_curr - p_target||) × scale
r_upright = max(0, 1 - 2(qx² + qy²))
r_collision = -Σₖ max(0, 1 - dist(robot, obs_k) / radius)
r_reach = 10.0 × 𝟙[dist < 0.5m]
```

### Interview Q&A

**Q: What is the upright reward and why do you use quaternions?**
A: Rewards the humanoid for staying vertical. We derive the world-frame up-direction from the root body quaternion: `up_z = 1 - 2(qx² + qy²)`. This equals 1.0 when upright, 0.0 when horizontal. We use quaternions directly (not Euler angles) to avoid gimbal lock and keep gradients smooth. Without this reward, the fall rate was 28.1%. Adding it with weight=0.5 reduced falls to 7.8%.

**Q: Why Euclidean distance for navigation reward, not Manhattan distance?**
A: Euclidean distance is rotation-invariant and gives the shortest path length. The robot can approach from any direction, and we want the reward to be symmetric. Manhattan distance would bias toward grid-aligned paths which don't exist in continuous environments.

**Q: What is reward shaping? Can it break your policy?**
A: Reward shaping adds intermediate rewards to help exploration. It can break the policy (reward hacking) if the shaped reward conflicts with the true objective. The Ng & Russell potential-based shaping theorem says `r'(s,a,s') = r(s,a,s') + γΦ(s') - Φ(s)` preserves optimal policy. We didn't use potential-based shaping; we used progress rewards, which generally work but require careful scaling to prevent local optima.

---

## 3. Robot Kinematics & Control

### What we did
G1 humanoid: 37-DOF, PPO outputs joint position targets, PD controller converts to torques. Franka (LIBERO): 7-DOF, GR00T outputs 7-dim OSC delta, LIBERO converts to torques internally.

### Key concepts

**Forward Kinematics:** Joint angles → end-effector pose. Always solvable (chain multiplication).

**Inverse Kinematics:** End-effector pose → joint angles. Non-unique for redundant robots. Solved numerically via damped pseudoinverse: `θ̇ = J⁺ẋ`.

**PD Controller:** `τ = kp(θ_target - θ) - kd·θ̇`. Stiffness (kp) pulls to target, damping (kd) prevents oscillation.

**OSC (Operational Space Control):** Control in Cartesian space. `τ = Jᵀ(Kp·Δx + Kd·ẋ)`. GR00T outputs Δx (7-dim: 3 position + 3 orientation + 1 gripper).

### Interview Q&A

**Q: What is the Jacobian and why does it matter?**
A: The Jacobian J maps joint velocities to end-effector velocity: `ẋ = J·θ̇`. It's a 6×7 matrix for Franka (6 task DOF, 7 joints). Used in: IK via pseudoinverse, OSC torque computation, singularity detection (when J loses rank → robot can't move in some directions → control becomes unstable).

**Q: What is the difference between joint space and task space control?**
A: Joint space: commands in joint angles/torques. Simple but unintuitive for manipulation tasks. Task space (OSC): commands in end-effector Cartesian coordinates. Intuitive ("move 2cm forward") but requires Jacobian computation. GR00T uses task space — the 7-dim OSC delta is what makes the model transferable: "move right 1cm" means the same thing regardless of which joints achieve it.

**Q: How do you handle a 37-DOF humanoid without explicit kinematics?**
A: We don't. PPO policy directly maps proprioceptive observations (joint positions, velocities, gravity, velocity commands) to joint position targets. The policy implicitly learns which joints to use for walking vs. balancing through millions of simulation steps. This is end-to-end learning — no explicit kinematic model required.

**Q: What is null-space motion in redundant robots?**
A: A 7-DOF robot (Franka) has more DOFs than needed for 6-DOF end-effector control. The extra DOF can be used for secondary objectives (obstacle avoidance, joint limit avoidance) without disturbing the end-effector: `θ̇_null = (I - J⁺J)·z`. This is the "null space" of the Jacobian.

---

## 4. Vision-Language-Action Models (GR00T)

### What we did
Evaluated NVIDIA GR00T N1.7 (3B params) on LIBERO Spatial: 97.0% success, 10 tasks, 20 eps each. Fixed 3 critical observation/action bugs that caused 0% success.

### Architecture
```
Input: agentview_image (256×256) + wrist_image (256×256) + state (8-dim) + language
       ↓
Qwen3-VL (2.7B) — vision-language encoder (from NVIDIA pre-training)
       ↓
DiT (Diffusion Transformer, 0.3B) — denoising action head
       ↓
Output: 7-dim OSC delta action chunk (8 steps per inference)
```

**DiT = Diffusion Transformer:**
- Action prediction as denoising: start from Gaussian noise, predict clean action
- Denoising steps: 10-20 during inference
- Conditioning: visual features + language embeddings

**Action Chunking:**
- Output 8 actions at once, execute all before next inference
- Effective control frequency: ~10 Hz (1 inference every 0.8 seconds)
- Prevents jerky behaviour from inference latency

### Interview Q&A

**Q: What is a VLA (Vision-Language-Action model) and how does GR00T work?**
A: A VLA ingests visual observations + language instructions and outputs actions. GR00T's backbone (Qwen3-VL) processes images and text jointly. The DiT action head treats action prediction as a denoising problem — conditioned on the visual/language context, it denoises a noise vector into a clean action chunk.

**Q: What is diffusion for robot actions? Why not just regress?**
A: Regression predicts a single action (unimodal output). Diffusion predicts a distribution over actions — better for multimodal manipulation (e.g., "grasp this object" could involve reaching from left OR right). Diffusion models capture the full distribution, not just the mean. For robotics, this prevents "averaging" between two valid strategies which would give an invalid strategy.

**Q: What rotation convention did GR00T use and why did it matter?**
A: GR00T was trained with axis-angle rotation in the observation state (3D vector = axis × angle). LIBERO provides quaternions. Converting quaternion to Euler angles (our initial approach) gave wrong input to the model → 0% success. Converting to axis-angle gave correct input → 97%. This was the hardest bug: the policy received plausible-looking (but wrong) rotation values, so errors were subtle.

**Q: Why does GR00T need a wrist camera?**
A: The wrist camera provides close-up view of the grasp — crucial for precise manipulation. The agentview camera sees the whole scene for spatial context, but can't resolve fine finger-object contact. GR00T uses both: agentview for where to move, wrist for how to grasp.

---

## 5. Behaviour Cloning & Imitation Learning

### What we did
T0: MLP BC policy on LIBERO state observations (12-dim) → 50% success task 0. T3: ResNet18+MLP BC policy on pixel observations (128×128×3) → ~20-35% expected.

### Theory

**Behaviour Cloning:**
```
min_θ E_{(s,a)~D} [||π_θ(s) - a_demo||²]
```
Supervised learning on (observation, action) pairs from expert demonstrations. Simple but has the **compounding error problem**.

**Compounding errors (distribution shift):**
- Policy trains on s ~ D_demo (expert states)
- At test time, policy visits s ~ D_π (policy's own states)
- If policy makes a mistake, it enters unseen states where it makes more mistakes
- Error compounds: ε per step → O(ε × T²) total error over T steps

**DAgger (Dataset Aggregation) — not used but interview-relevant:**
```
1. Train on D₀ = demo data
2. Run policy, query expert for corrective labels
3. Aggregate: D_{i+1} = D_i ∪ new data
4. Retrain on D_{i+1}
```
Fixes distribution shift by training on states the policy actually visits.

**T3 approach — pixel-only BC:**
- ResNet18 (ImageNet pretrained) as visual encoder
- MLP head: 512 → 256 → 256 → 7-dim action
- Trained on 61,750 (image, action) pairs from all 10 LIBERO Spatial tasks
- Uses agentview_rgb from HDF5 demos, maps to OSC delta actions

### Interview Q&A

**Q: What is the compounding error problem in behaviour cloning?**
A: At training time, the policy sees the expert's state distribution. At test time, it visits states caused by its own (imperfect) actions. Slight deviations lead to states where the policy has no training data. Each mistake makes the distribution drift further, causing cascading failures. T0 (50% success) vs T1 GR00T (97% success) partly reflects BC vs. NVDIA's full training with robust demonstrations.

**Q: Why use ResNet18 pretrained on ImageNet for robot manipulation?**
A: ImageNet features capture general visual patterns (edges, textures, shapes) that are still useful for robot perception. Fine-tuning from ImageNet converges much faster than training from scratch, especially with limited robot data. This is transfer learning — leverage a well-trained feature extractor for a new task.

**Q: What is the advantage of pixel observations vs state observations?**
A: Pixels generalise across embodiments and environments — a camera always produces a 128×128×3 image regardless of the robot. State observations (joint angles, EEF position) require careful calibration and don't transfer. Pixel policies can potentially run on any robot with a camera. The downside: harder to train (much higher dimensional input), requires visual encoder.

---

## 6. World Models (Dreamer-mini RSSM)

### What we did
P2: Trained Dreamer-mini RSSM on Isaac Lab G1 rollouts (3000 steps, loss 0.76 → 0.011). T2: Same RSSM on LIBERO manipulation rollouts (3000 steps, loss 1.40 → 0.008).

### Theory

**RSSM (Recurrent State Space Model):**
```
State: zₜ = [hₜ, sₜ]   (deterministic + stochastic)
Prior:  sₜ ~ p(sₜ | hₜ)
Posterior: sₜ ~ q(sₜ | hₜ, oₜ)
Transition: hₜ₊₁ = f(hₜ, sₜ, aₜ)
Decoder: oₜ ~ p(oₜ | hₜ, sₜ)
Reward: rₜ ~ p(rₜ | hₜ, sₜ)
```

**Why RSSM?**
- Deterministic part (GRU hₜ) handles temporal dependencies
- Stochastic part (sₜ) handles environmental uncertainty
- "Dreamer" learns to plan in latent space — no actual simulation needed

**Training loss:**
```
L = E[-log p(oₜ|zₜ)] + E[-log p(rₜ|zₜ)] + β × KL[q(sₜ|hₜ,oₜ) || p(sₜ|hₜ)]
   reconstruction     reward prediction       regularisation
```

**Our results:**
- P2: Initial loss 0.76 → final 0.011 (69× reduction). Imagined reward finite ✅
- T2: Initial loss 1.40 → final 0.008 (175× reduction). Imagined reward finite ✅

### Interview Q&A

**Q: What is a world model in RL?**
A: A learned model of environment dynamics: given (state, action), predict next state and reward. Allows planning and imagination without interacting with the real environment. Dreamer-mini generates imaginary rollouts in latent space and trains a policy on those imagined experiences → more sample efficient than model-free methods.

**Q: What is KL divergence in the RSSM training loss?**
A: KL divergence measures the difference between the posterior q(s|h,o) (what the model infers from actual observation) and the prior p(s|h) (what the model predicts without seeing the observation). Minimising KL makes the prior match the posterior → the model learns to predict the future state distribution correctly from latent state alone.

**Q: Why does the loss matter for your T2 DoD?**
A: DoD was "imagined reward must be finite". If the world model diverges (unbounded loss), imagined trajectories become nonsensical and the reward prediction is unstable. Loss 1.40 → 0.008 means the model converged, and imagined_reward = 0.0108 (finite → PASS).

---

## 7. Language-Conditioned Navigation

### What we did
P1.2 LangNav: G1 navigates to one of 3 markers based on natural language command. 98.8% success. Used frozen sentence transformer to encode commands.

### Theory

**Frozen text encoder approach:**
```
language_command → sentence-transformers/all-MiniLM-L6-v2 → 384-dim embedding
                                                            ↓ (frozen during RL)
                                                    concatenated to policy obs
```

**Why freeze the text encoder during RL?**
- RL training can corrupt text encodings (gradient-based updates optimise for reward, not semantic meaning)
- Pretrained embeddings already capture semantic similarity well
- Freezing reduces parameter count (fewer things to learn)

**Why does it work?**
- "Go to the red marker" and "navigate to the red marker" → similar embeddings → similar policy behaviour
- "Go to the red marker" and "Go to the blue marker" → different embeddings → different navigation behaviour
- The policy learns: "when embedding looks like X, navigate toward target_id=0"

### Interview Q&A

**Q: How do you encode language for RL policies?**
A: We use a pretrained sentence transformer (all-MiniLM-L6-v2, 384-dim). The command is encoded once at episode start and concatenated to the observation at every timestep. The encoder is frozen during PPO training to preserve semantic structure.

**Q: What is the advantage of using pretrained language models vs. one-hot encoding?**
A: One-hot: discrete, no semantic structure, doesn't generalise to unseen commands. Sentence embeddings: continuous, semantically similar commands map to similar vectors, can handle paraphrases and novel phrasings. Our policy generalised to "please navigate toward the red marker" even though training used "go to red marker".

---

## 8. GPU-Accelerated Simulation

### What we did
Isaac Lab: 4096 parallel G1 environments on a single L4 GPU. 20 million timesteps per 20 minutes of training.

### Key concepts

**Why parallel simulation matters:**
- PPO sample efficiency: 4096 envs × 24 steps = 98,304 samples per update
- Wall-clock time is dominated by simulation, not gradient computation
- GPU-resident simulation (no CPU-GPU transfer) → eliminates bottleneck

**Isaac Lab architecture:**
- All environment tensors (joint states, forces, rewards) on GPU
- Physics via NVIDIA PhysX running on GPU
- Python interface: vectorised operations on `torch.Tensor` directly

**Comparison:**
| Simulation | Envs | Steps/sec | Platform |
|---|---|---|---|
| MuJoCo (CPU) | 1 | ~1,000 | CPU |
| Isaac Lab | 4096 | ~500,000 | GPU |
| MuJoCo MJX | 4096 | ~200,000 | GPU |

### Interview Q&A

**Q: Why use Isaac Lab instead of MuJoCo for G1 training?**
A: Isaac Lab supports GPU-parallelised physics with 4096 environments simultaneously. This gives 500× more samples per second than single-env CPU simulation. For complex 37-DOF humanoid locomotion, this parallelism is what makes 20-minute training feasible.

**Q: What are the limits of sim-to-real transfer?**
A: Simulation has perfect state estimation, no sensor noise, idealized contact models, and zero latency. Real robots have: noisy observations, joint friction, latency between sensing and actuation, and complex contact dynamics (soft contacts, deformable objects). Mitigations: domain randomisation (mass, friction, observation noise), careful actuator modelling, and deployment-specific fine-tuning.

---

## 9. Project-Level: Architecture Decisions

**Q: Why ResNet18 for T3 and not a Transformer?**
A: ResNet18 is well-understood, fast to train, and provides strong ImageNet-pretrained features. For a BC baseline with limited data (50 demos × 10 tasks = 61,750 images), a transformer would overfit. ResNet18 provides the right inductive biases for spatial visual features.

**Q: What was your biggest debugging challenge?**
A: The GR00T rotation convention bug (F-11). The model input looked correct (valid numbers in the right range) but used the wrong 3D rotation parameterisation. Diagnosis took hours of comparing our implementation against NVIDIA's reference. Lesson: always compare against the official reference implementation before debugging the model.

**Q: What would you change if you had unlimited compute?**
A: (1) Train GR00T LoRA fine-tune (need L40S for 40GB VRAM) to beat the 97% pretrained baseline. (2) Run P3 Vision Nav with full 4096-env parallelism + cameras. (3) C5 Capstone: full loco-manipulation (walk to object, pick it up).

**Q: How does your work compare to the state of the art?**
A: GR00T N1.7 at 97% on LIBERO Spatial matches NVIDIA's reported 97.7%. Our contribution was building the eval harness and diagnosing the 3 critical obs/action convention bugs that blocked reproduction. For navigation, our G1 policies are competitive with IsaacGym baselines (~90%+ success) trained in similar compute budgets.

---

## 10. Quick-Fire Concepts

| Topic | One-line answer |
|---|---|
| Euclidean distance | `||p₁ - p₂||₂ = √(Σ(xᵢ-yᵢ)²)`. Used for nav progress and reach detection. |
| Center of mass | `r_CoM = Σmᵢrᵢ/Σmᵢ`. We proxy it via root quaternion (much simpler). |
| Quaternion | 4D rotation: `[w, x, y, z]`, `||q||=1`. No gimbal lock. Smooth interpolation. |
| Gimbal lock | When Euler angles reach ±90° pitch, 2 axes align and a DOF is lost. |
| Sample efficiency | How much data/environment interaction is needed to learn a policy. |
| Sim-to-real gap | Performance difference between simulation training and real deployment. |
| Action chunking | Predict and execute N actions at once before re-querying policy. Reduces jerking. |
| Null space motion | Motion in redundant joints that doesn't affect end-effector pose. |
| Grasp pose | Orientation + position of gripper for successful object pickup. |
| Domain randomisation | Randomise physics parameters during training to improve sim-to-real transfer. |
| BDDL | Behaviour Domain Definition Language — LIBERO uses it to define task goals formally. |
| OSC | Operational Space Control: control end-effector in Cartesian frame using Jacobian. |
| DiT | Diffusion Transformer: predicts action by iteratively denoising from Gaussian noise. |

---

## Related

- [[Robot Kinematics & Dynamics]] — detailed kinematics theory
- [[Reward Engineering Deep Dive]] — all reward formulations with code
- [[World Models (Dreamer-mini)]] — RSSM architecture
- [[PPO with RSL-RL]] — PPO implementation details
- [[00 - Failure Index]] — every bug, root cause, fix
