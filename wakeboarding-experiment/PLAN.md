# Wakeboarding Start — Humanoid RL Project Plan

> **Status:** PLAN (no code yet). This document is written to be **executed by a Sonnet-class coding model** end-to-end. Every section is concrete: env design, MDP, reward weights, curriculum, training config, checkpoints, metrics, and a numbered task list. Reproduce on the existing Humanoid Isaac Lab + RSL-RL + Lightning AI stack.

---

## 0. One-paragraph summary

Train a **Unitree G1 humanoid** in **Isaac Lab** to perform a **wakeboard "deep-water start"** — but in a **dry/sand environment** instead of water. The robot begins **crouched and seated**, feet bound to a board, arms straight holding a rope handle. A **rope force ramps the robot forward to an initial pull speed of 30 km/h (8.33 m/s)**. Using **PPO (RSL-RL)**, the robot must learn the correct biomechanics — **stay crouched, keep arms straight, keep a positive board angle (~10–20°), weight the heels, lean slightly back without pulling against the rope, then gradually extend the legs to a stable standing ride.** This is methodologically a **getting-up-under-external-pull** task; we follow the **HumanUP two-stage (discovery → deployable) recipe** plus an optional **AMP style reward** for a natural wakeboard stance. Later phases add a **computer-vision coaching loop** that detects a real human's knee/hip angles and compares them to the robot's learned optimal pose.

---

## 1. Why this is the right RL formulation (prior art)

- **HumanUP — "Learning Getting-Up Policies for Humanoid Robots"** ([arxiv 2502.12152](https://arxiv.org/html/2502.12152v1)): two-stage RL on the **G1**: Stage I "discovery" finds *any* feasible crouch→stand trajectory under weak regularization; Stage II "deployable" tracks an 8×-slowed version under strict smoothness/torque penalties + domain randomization. 78.3% real-world success. **We copy this structure** — the wakeboard start is a getting-up motion with a horizontal rope disturbance and a board constraint.
- **AMP — Adversarial Motion Priors** ([arxiv 2104.02180](https://arxiv.org/pdf/2104.02180)): a learned **style reward** from a small reference-motion dataset (a discriminator that rewards motions that look like the demos). Lets us get a **natural-looking wakeboard stance** without hand-tuning every joint. Optional but high-value for the "show recruiters" demo.
- **Whole-Body Control under external force** (Isaac Lab 2.3, Booster Gym, Humanoid-Gym): standard PPO + heavy domain randomization + perturbation forces is the proven way to get balance under disturbances. The rope is exactly such a disturbance.

**Biomechanics grounding** (so the reward matches reality), from wakeboard coaching sources:
- Start: knees pulled to chest, board perpendicular to pull, tip just above ground. ([Monster Tower](https://monstertower.com/blog/post/how-to-get-up-on-a-wakeboard), [Axis](https://blog.axiswake.com/beginner-wakeboarding-tips))
- **Positive board angle ≥ 10°**, heels weighted, so the board planes up instead of digging in. ([Monster Tower – getting up](https://monstertower.com/blog/post/getting-up-staying-up-on-a-wakeboard))
- **Arms straight, handle low at the hips**; do **not** bend elbows / pull against the rope (the #1 failure → face-plant). ([WaterVolleyball](https://watervolleyball.com/wakeboarding-for-beginners/))
- **Let the pull do the work** — standing up too fast throws you off balance; extend legs **gradually**. ([Monster Tower – proper positioning](https://monstertower.com/blog/post/proper-positioning-for-wakeboarding))
- Slight knee bend at all times (never lock knees); lean slightly back but **not against the rope**.

These five rules become explicit reward/penalty terms in §5.

---

## 2. Success definition (what "done" means)

A single episode is a **SUCCESS** if, within `T_success = 6 s` of the pull starting, the robot reaches and holds (for ≥ 1.5 s) a **stable riding stance**:
- pelvis height ≥ `0.55 m` (G1 standing-ish, crouched-ride),
- torso uprightness (gravity-z projection) ≥ `0.85`,
- both feet still bound to the board, board still moving forward,
- no fall (no non-foot/non-hand body contact with ground),
- board angle within `[5°, 30°]`.

Project-level "done": Stage II policy reaches **≥ 70% success at 30 km/h** with a saved checkpoint, a results table across rope speeds, ≥ 2 ablations, and a rollout video. (Mirrors HumanUP's bar on a harder, novel task.)

---

## 3. Environment design (Isaac Lab)

### 3.1 Robot
- **Unitree G1** (reuse the URDF/config already used by `LanguageConditionedG1*` tasks). 23–37 DoF depending on hand model; **keep arms actuated** (needed to hold the handle) — do **not** freeze them as the locomotion tasks do.

### 3.2 The board
- A **rigid-body wakeboard** (~1.4 m × 0.4 m thin box, mass ~3 kg) spawned under the robot's feet.
- **Feet → board binding:** fixed joints (or a welded constraint) attaching each G1 foot to the board, so the board and feet move as one. (Wakeboard bindings are rigid.)
- **Board ↔ ground contact:** model the **sand/dry surface** as a frictional plane. Because real boarding is low-friction-glide, use a **moderate-low dynamic friction** (μ ≈ 0.3–0.5) so the board slides forward under the rope rather than sticking. (This is the "sand instead of water" abstraction: a sliding board on ground.)

### 3.3 The rope / pull model
Two interchangeable models (implement both behind a config flag; default = **B**):
- **(A) Constant-force handle:** apply a constant horizontal force `F_pull` at the handle body the hands hold. Simple, but doesn't cap speed.
- **(B) Velocity-target spring (recommended):** the rope's far end is a **virtual anchor moving forward at the target speed** `v_pull` (ramped per curriculum). Apply a PD/spring force at the handle toward the anchor: `F = k_p*(x_anchor - x_handle) + k_d*(v_anchor - v_handle)`, capped at `F_max`. This naturally produces "boat pulls you to 30 km/h" dynamics and is robust.
- **Handle:** a small graspable body; the hands are attached to it via fixed joints during the start (assume grip; grasp learning is out of scope for v1).
- **`v_pull` target = 30 km/h = 8.33 m/s**, applied along +x.

### 3.4 Initial state (the crouched float, dry version)
Reset the G1 into a **seated/crouched "cannonball" pose**: hips & knees deeply flexed (knees toward chest), torso reclined ~30–45° back, board in front roughly perpendicular to +x with the tip up ~15°, arms extended straight forward holding the handle near the hips. Add small randomization (§7). This replaces "floating on your back in water."

### 3.5 Terrain
Flat sand plane (v1). Later: small ripples / friction patches for robustness (mirrors the existing rough-terrain DR work).

---

## 4. MDP definition

### 4.1 Observations (policy input)
- **Proprioception:** joint positions & velocities (all actuated DoF), base orientation (quaternion or gravity vector in base frame), base linear & angular velocity, previous action.
- **Task/board:** board pitch angle, board forward velocity, foot-board contact flags.
- **Rope:** handle position relative to pelvis (3-vec), rope force vector on handle (3-vec), current `v_pull` (scalar, so the policy "knows" the pull intensity).
- **Phase:** a normalized phase/clock variable `t/T` (helps with the non-periodic getting-up motion, per HumanUP).
- *(Optional, ties to Part-1 language work)* a command embedding slot (e.g. "start", "stay low") for future language conditioning — reuse `language_commands.embedding_for_text`.

### 4.2 Actions
- Target joint positions for a **PD controller** (standard Isaac Lab / RSL-RL action space), all actuated DoF. `num_steps_per_env = 24`.

### 4.3 Episode / termination
- **Horizon:** 8 s (`episode_length_s = 8`).
- **Terminate (failure)** on: torso/pelvis/head/knee ground contact (fall), feet leaving the board, board pitch outside `[-20°, 45°]`, or torso uprightness < 0.3.
- **Early-success bonus** when §2 criteria are met (but keep running to test "staying up").

---

## 5. Reward design (the core)

Total reward = **task** + **biomechanics-shaping** + **style (AMP, optional)** − **penalties**. Weights are starting points to tune. Use Isaac Lab manager-based reward terms (one `RewTerm` each), so each is independently ablatable.

### 5.1 Task rewards (get up & stay up) — from HumanUP
| Term | Definition | Weight |
|---|---|---|
| `pelvis_height` | `clip(h_pelvis, 0, h_target)` rising toward 0.6 m | +2.0 |
| `height_progress` | positive Δ pelvis height per step (phase-gated: rewarded only after t > 0.5 s, to discourage standing too fast) | +1.5 |
| `uprightness` | gravity-vector z-projection of torso (1 = vertical) | +2.0 |
| `survival` | +1 per step alive (board moving, not fallen) | +0.5 |
| `forward_glide` | board forward speed tracking toward `v_pull` (gaussian around target) | +1.0 |
| `success_bonus` | one-time +50 when §2 stance reached & held 1.5 s | +50 (sparse) |

### 5.2 Biomechanics-shaping rewards (make it a *correct* wakeboard start)
| Term | Encodes the rule | Definition | Weight |
|---|---|---|---|
| `board_positive_angle` | "board ≥10°, heels weighted, plane up" | reward board pitch in `[10°,20°]`, taper outside | +1.5 |
| `arms_straight` | "arms straight, don't pull the handle in" | reward elbow extension (penalize flexion); compare handle-to-shoulder distance vs. arm length | +1.0 |
| `handle_at_hips` | "handle low near hips" | negative distance of handle from a hip-height target point | +0.8 |
| `lean_back_moderate` | "lean slightly back, not against rope" | reward torso back-lean in `[10°,25°]`; **penalize** if back-lean correlates with elbow flexion (= pulling against rope) | +0.7 |
| `knee_bend_maintained` | "never lock knees; gradual extension" | reward knee angle staying within a bent band early, relaxing the band as phase→1 | +0.8 |

### 5.3 Style reward (AMP, optional but recommended)
- A discriminator trained on a **small wakeboard-stance reference dataset** (see §9) outputs a per-step style reward. Weight **+1.0**, blended with task reward as in AMP. Turn on in Stage II / as an ablation.

### 5.4 Penalties / regularization — from HumanUP
| Term | Weight |
|---|---|
| `stand_too_fast` (pelvis vertical velocity above cap early in episode) | −1.0 |
| `pull_against_rope` (elbow flexion torque while rope force high) | −1.0 |
| `torque` / `energy` (Σ τ²) | −1e-4 |
| `action_rate` / `action_accel` (smoothness) | −0.01 / −1e-3 |
| `dof_pos_limits` | −5.0 |
| `dof_vel`, `base_ang_vel` excess | −1e-3 |
| `fall` (terminal) | −20 |

**Stage I vs Stage II:** Stage I uses mostly §5.1 + §5.2 with **weak** §5.4 (discovery). Stage II adds **strong** §5.4 + dense tracking of the slowed Stage-I trajectory + §5.3 AMP (deployable, smooth, natural).

---

## 6. Curriculum (two nested curricula)

### 6.1 Pull-speed curriculum (the headline difficulty)
Ramp the **target pull speed** as success rate clears a threshold (auto-curriculum):
`v_pull`: **10 → 15 → 20 → 25 → 30 km/h** (advance when rolling success ≥ 60% at current level). Also ramp the **force ramp rate** (how violently the boat yanks): gentle → sharp. The 30 km/h sharp-yank is the final boss (where real beginners fail).

### 6.2 Stage curriculum (HumanUP)
- **Stage I — Discovery:** weak regularization, canonical init, simplified collision, fixed dynamics, fast motion allowed. Goal: *find a way up at all*.
- **Stage II — Deployable:** track 8×-slowed Stage-I trajectory, strict smoothness/torque, 20k randomized init poses, full collision, domain randomization, AMP style. Goal: *smooth, natural, robust*.

---

## 7. Domain randomization (reuse existing infra where possible)
- **Pull:** `v_pull` ±15%, force ramp rate, pull direction yaw jitter ±10°, occasional mid-start tug perturbation.
- **Surface:** sand friction μ ∈ [0.25, 0.6].
- **Board:** mass ±20%, length ±10%.
- **Robot:** joint stiffness/damping ±25% (reuse `randomize_joint_parameters` from the robust task), link mass ±10%, added sensor noise, action latency.
- **Init pose:** randomize crouch depth, torso lean, board pitch, lateral offset.

---

## 8. Training configuration
- **Algorithm:** PPO via **RSL-RL** (reuse `RslRlOnPolicyRunnerCfg` patterns from `g1_vla_vision_cfg.py`). MLP policy `[512, 256, 128]` ELU (bigger than locomotion — harder task).
- **PPO HPs (start):** clip 0.2, entropy 0.005, lr 5e-4 adaptive (desired_kl 0.01), γ 0.99, λ 0.95, 5 epochs, 4 minibatches, 24 steps/env.
- **Envs:** 4096 (Stage I) / 8192 (Stage II). **Iterations:** Stage I ~3–5k, Stage II ~5–10k.
- **Hardware:** Lightning AI **L40S** (reuse Docker image `ghcr.io/sushruths04/humanoid-isaaclab:latest`). Always smoke-test (16 envs, 2 iters) first.
- **Logging:** wandb (reuse existing setup).

---

## 9. Reference-motion data for AMP (how to get "what a wakeboard start looks like")
Pick the cheapest viable path:
1. **Keyframe authoring (v1, no external data):** hand-author 3–5 keyframes (cannonball → mid-rise → tall stance) and interpolate → a short reference clip. Enough to seed AMP / shape the pose.
2. **Video → pose (v2):** take 2–3 YouTube wakeboard-start clips, run a **3D human pose estimator** (e.g. a monocular 3D pose model), **retarget** to the G1 skeleton (motion retargeting), use as AMP demos. Higher fidelity, more work.
- Store clips in `wakeboarding-experiment/data/reference_motions/`.

---

## 10. Checkpoints, milestones & what to compare

### 10.1 Checkpoint ladder (save every one to HF, per the repo workflow)
| ID | Milestone | Accept criteria |
|---|---|---|
| `ckpt_00_smoke` | env builds, pipeline reaches PPO | 16 envs, 2 iters, no crash |
| `ckpt_10_stage1_slow` | Stage I up at 10 km/h | ≥60% success @10 km/h |
| `ckpt_20_stage1_30` | Stage I up at 30 km/h | ≥50% success @30 km/h (any style) |
| `ckpt_30_stage2_deploy` | Stage II smooth+natural @30 km/h | ≥70% success, smoothness ✓, AMP on |
| `ckpt_40_robust` | + full domain randomization | ≥60% success under DR + perturbation tug |

### 10.2 Metrics (logged every eval)
- **Start success rate** (primary), **time-to-stand**, **fall rate**, **mean episode length**, **board-angle adherence** (% time in [10°,20°]), **arm-straightness** (mean elbow extension), **smoothness** (action accel), **energy** (Σ τ²).

### 10.3 Comparison tables (the "show recruiters" deliverables)
- **Table A — across pull speed:** success/fall/time-to-stand at 20 / 25 / 30 / 35 km/h.
- **Table B — ablations:** (i) no pull-speed curriculum, (ii) no biomechanics-shaping rewards, (iii) no AMP style, (iv) no domain randomization. Show each hurts a specific metric.
- **Table C — Stage I vs Stage II:** success similar but smoothness/energy/naturalness much better in II.
- Save all as JSON in `results/` **and** render into the live Obsidian results note (§12).

---

## 11. Phase 2 (later) — real-world CV coaching loop
The user's bigger vision: a humanoid that has learned the *optimal* start angles becomes a **coach** for a human learner.
- **Perception:** webcam → **monocular pose estimation** (e.g. MediaPipe Pose / a 3D lifter) → human **knee, hip, torso-lean, arm** angles in real time.
- **Reference:** the trained policy's average successful-start joint trajectory = the "ideal" angle profile per phase.
- **Feedback:** compare human angles vs robot-ideal at the matching phase → actionable cues ("bend knees more", "straighten arms", "don't pull the handle"). Optionally close the loop: use human-correction data to fine-tune the reward.
- **Deliverable:** a small `coach/` app; out of scope for v1 but the env/observation design (joint-angle-centric) is chosen to make this trivial later.

---

## 12. Documentation & Obsidian discipline (REQUIRED, ongoing)
This project must stay self-documenting:
- The Obsidian vault lives in `wakeboarding-experiment/vault/` with **`00_INDEX.md`** as the live hub.
- **Rule:** whenever the executor adds/changes a **script**, it updates `vault/02_Implementation/Scripts.md` (what the script does, inputs/outputs, how to run).
- **Rule:** whenever a **run produces results**, it appends to `vault/03_Results/Results_Live.md` (the comparison tables in §10.3) and updates the status badge in `00_INDEX.md`.
- `vault/04_Log/Experiment_Log.md` = dated entries of every training run (config, ckpt path, metrics, observations). See `DOC_PROTOCOL.md`.

---

## 13. Proposed file/folder structure (executor creates these)
```
wakeboarding-experiment/
├── PLAN.md                      ← this file
├── README.md
├── DOC_PROTOCOL.md              ← the "always document" rules
├── requirements.txt
├── configs/
│   ├── smoke.yaml  full.yaml  stage1.yaml  stage2.yaml
├── src/
│   ├── tasks/wakeboard_start_cfg.py     ← env (robot+board+rope), obs, terminations
│   ├── rewards/wakeboard_rewards.py     ← all §5 reward terms
│   ├── rope_model.py                    ← §3.3 pull models A/B
│   ├── board.py                         ← §3.2 board asset + foot binding
│   ├── curriculum.py                    ← §6 pull-speed + stage curricula
│   └── amp/ (discriminator, ref-motion loader)   ← §5.3/§9
├── scripts/
│   ├── 00_smoke.sh  10_train_stage1.sh  20_train_stage2.sh
│   ├── 30_eval.sh   31_eval_speed_sweep.sh   40_record_video.sh
│   └── 99_collect_results.sh
├── data/reference_motions/      ← §9 AMP clips
├── results/                     ← eval JSONs (+ .gitkeep)
├── checkpoints/                 ← (gitignored; pushed to HF)
└── vault/                       ← Obsidian (see §12)
```

---

## 14. Executor task list (numbered, do in order)
1. **Scaffold** the folder structure in §13; write `README.md`, `DOC_PROTOCOL.md`, `requirements.txt`.
2. **Board asset + foot binding** (`board.py`) — spawn board, weld to feet; verify in a no-policy sim.
3. **Rope model** (`rope_model.py`) — implement model B (velocity-target spring) + A; unit-test the force.
4. **Env config** (`wakeboard_start_cfg.py`) — G1 (arms actuated) + board + rope + initial crouch pose + observations (§4.1) + terminations (§4.3). CPU-importable guard like the existing tasks.
5. **Reward terms** (`wakeboard_rewards.py`) — implement every §5 term as a `RewTerm`; expose weights in config.
6. **Smoke test** (`00_smoke.sh`, 16 envs, 2 iters) → save `ckpt_00_smoke`. Document in vault.
7. **Curriculum** (`curriculum.py`) — pull-speed auto-curriculum (§6.1).
8. **Stage I training** (`10_train_stage1.sh`) — discovery rewards/weights; train to `ckpt_20_stage1_30`. Log to wandb + Experiment_Log.
9. **Reference motion** (§9 option 1 keyframes) + **AMP** discriminator (`amp/`).
10. **Stage II training** (`20_train_stage2.sh`) — slowed-trajectory tracking + strict reg + AMP → `ckpt_30_stage2_deploy`.
11. **Domain randomization** (§7) → `ckpt_40_robust`.
12. **Eval + sweeps** (`30_eval.sh`, `31_eval_speed_sweep.sh`) — produce Tables A/B/C as JSON in `results/`.
13. **Video** (`40_record_video.sh`) — headless rendering kit + `--enable_cameras` (reuse the vision blocker fix) → mp4 rollout.
14. **Update docs** — fill `Results_Live.md`, `Scripts.md`, `00_INDEX.md` badges. (Per `DOC_PROTOCOL.md`, this happens continuously, not just at the end.)

---

## 15. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Reward hacking (robot "cheats" up without correct form) | biomechanics-shaping terms (§5.2) + AMP style + visual review of rollouts |
| Board/foot constraint unstable in PhysX | start with welded fixed joints, low solver iteration tuning; test asset before RL |
| 30 km/h yank instantly throws robot | pull-speed curriculum (§6.1) is mandatory; never start at 30 |
| Sand-vs-water abstraction unrealistic | document it as a deliberate dry analog; friction sweep in DR; note as limitation |
| AMP data scarce | start with hand-authored keyframes (§9.1); video-retarget later |
| Sim-to-real (Phase 2) | heavy DR + the HumanUP Stage-II deployable recipe; CV-coach uses angles, robust to embodiment gap |

---

## 16. Requirements / dependencies
- Isaac Sim 5.1 + Isaac Lab (existing Docker image), RSL-RL, PyTorch (cu126), Unitree G1 URDF, wandb.
- AMP: a discriminator (small MLP) + reference-motion loader; a 3D pose estimator only if doing §9 option 2.
- Phase 2 coach: a webcam + monocular pose estimation lib (MediaPipe / equivalent).
- Compute: Lightning AI L40S (camera/video steps need the headless rendering kit).

**Sources:** [HumanUP getting-up (G1)](https://arxiv.org/html/2502.12152v1) · [AMP](https://arxiv.org/pdf/2104.02180) · [wakeboard start technique](https://monstertower.com/blog/post/getting-up-staying-up-on-a-wakeboard) · [arm/handle position](https://watervolleyball.com/wakeboarding-for-beginners/) · [body position](https://blog.axiswake.com/beginner-wakeboarding-tips) · [Isaac Lab WBC under disturbance](https://developer.nvidia.com/blog/streamline-robot-learning-with-whole-body-control-and-enhanced-teleoperation-in-nvidia-isaac-lab-2-3/)
