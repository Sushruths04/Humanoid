# 6-Month Physical-AI Program: Robot Training → VLA → World Models (with NVIDIA Cosmos)

**Owner:** Sushruth · **Created:** 2026-06-02 · **Status:** APPROVED-DRAFT (pending owner review)
**Anchor robot:** Unitree G1 humanoid (continues existing repo work)
**Organizing approach:** Capstone funnel — 5 escalating projects (P0–P5) feed one integrated flagship demo.

---

## 0. How to use this plan (read this first, executing agent)

This document is written so an autonomous agent — or a human — can pick up **any project** and start building without re-deriving context.

- Each **Project (Pn)** is self-contained: objective → prerequisites → what to build → **checkpoints** → success metrics → deliverables.
- Each **Checkpoint (CPn.m)** has a **Definition of Done (DoD)** that is *verifiable* (a command runs, a number is hit, a file exists). Do not mark a checkpoint complete until its DoD is literally observed. Evidence before assertion.
- **Smoke-test rule (non-negotiable, inherited from this repo's existing workflow):** before any long/expensive run, run the smallest version that proves the pipeline reaches the training loop. Never scale before the smoke test passes.
- **Efficiency notes** appear per checkpoint — they tell you how to do it cheaply on a single GPU.
- Work top-to-bottom. P0 fixes existing debt and must finish before P1+.
- When a checkpoint produces a large artifact (checkpoint weights, videos, datasets), push to Hugging Face, not git. Code and small docs go to GitHub. (Existing repo convention — see `START_HERE.md`.)

---

## 1. Constraints & assumptions (the envelope)

| Constraint | Value | Implication |
|---|---|---|
| Compute | **1 cloud GPU** (L40S 48GB or A100 80GB), pay-as-you-go | *Use/fine-tune* Cosmos (2B class) with **LoRA**; never pretrain 14B/64B. Prefer A100-80GB for P4. |
| Timeline | **~6 months** | ~3–4 wk per project, P5 gets 6–8 wk. |
| Primary outcome | **Job / portfolio in physical AI** | Every project ships a demo video + writeup + clean repo. One flagship (P5). |
| World-model depth | **Build a small one + use Cosmos** | P2 = implement a small WM from scratch (understand internals). P3/P4 = use Cosmos at scale. |
| Domain anchor | **Unitree G1 humanoid** | Reuse existing Isaac Lab env, Docker image, GR00T N1.7 fine-tune. |

**Stack as of June 2026 (verified current):**
- **Isaac Lab** + **Isaac Lab-Arena** (standardized policy eval).
- **Isaac GR00T N1.7** — open VLA, backbone = Cosmos-Reason 2 / Qwen3-VL. (You already fine-tuned this.)
- **Cosmos Predict 2.5** (2B / 14B) — video world model; has a **Robot/Policy** post-training recipe (RoboCasa, Libero) + action-conditioned distillation, in `cosmos-cookbook`.
- **Cosmos Transfer 2.5** — sim→photoreal structure-preserving translation (synthetic data / domain randomization).
- **Cosmos Reason 2** — VLM reasoning (spatiotemporal, chain-of-thought).
- **Cosmos 3** (released 2026-05-31) — open omnimodel, Nano 16B / Super 64B, backbone for World-Action-Models. *Inference/eval only on our budget; do not plan to train it.*

---

## 2. Compute & cost budget (the "how efficient" part)

**Golden efficiency rules (apply everywhere):**
1. **LoRA / PEFT** for every foundation-model fine-tune (GR00T, Cosmos Predict). Never full-fine-tune on one GPU.
2. **Mixed precision (bf16)** + **gradient checkpointing** + **8-bit optimizer** (bitsandbytes) for any model >1B params.
3. **Smoke test first**: tiny env count / 2 iters / 32×32 camera before scaling (already in this repo's scripts).
4. **Reuse the prebuilt Docker image** (`ghcr.io/sushruths04/humanoid-isaaclab:latest`); never rebuild per run.
5. **Checkpoint to Hugging Face**; treat the cloud GPU as disposable.
6. **Maximize parallel envs, not wall-clock**: Isaac Lab gives ~100k+ steps/s on G1 at 8192 envs — prefer more envs + fewer iterations.
7. **Cache Cosmos generations**: WFM video gen is expensive; generate a synthetic dataset once, store on HF, reuse across runs.

**Rough budget per project (single GPU, on-demand ≈ $1.5–2.5/hr for L40S, ~$2–3.5/hr A100-80GB):**

| Project | GPU-hours (est.) | Notes |
|---|---|---|
| P0 | 15–30 | Mostly RL; cheap. |
| P1 | 40–80 | RL + curriculum; teacher then student. |
| P2 | 30–60 | Small WM; toy domains keep it cheap. |
| P3 | 50–100 | Vision rendering is the cost driver; Cosmos-Transfer gen is bursty. |
| P4 | 80–150 | **Most expensive.** Cosmos Predict post-train (LoRA) + generation. Prefer A100-80GB. |
| P5 | 100–200 | Integration + lots of eval/video. |
| **Total** | **~315–620 GPU-hr** | ≈ **$600–1,500** across 6 months if on-demand; less with spot/credits. |

---

## 3. Global conventions

**Repo layout additions** (extends current structure; everything new lives under `programs/`):
```
programs/
  p0_foundation_eval/        # honest baseline + eval harness
  p1_language_vla/           # state-based language-conditioned VLA
  p2_world_model_mini/       # small world model from scratch (pure PyTorch)
  p3_vision_vla_cosmos/      # vision policy + Cosmos-Transfer synthetic data
  p4_cosmos_world_sim/       # Cosmos Predict post-train + planning/eval
  p5_capstone/               # integrated agent
  common/                    # shared: eval, logging, text encoders, video utils
docs/
  PHYSICAL_AI_6MONTH_PLAN.md # this file
  results/                   # result tables + writeups per project
```

**Experiment tracking:** Weights & Biases (or MLflow). Every run logs: task id, command set, reward/obs terms, iters, GPU type, Isaac Lab + warp versions, success rate, fall rate, best-checkpoint path, rollout video. (Mirror of the table already in `FUTURE_TASKS_HUMANOID_ROBOT.md`.)

**Definition of Done — universal gates** (every project must pass before "complete"):
- [ ] Smoke test passes (pipeline reaches training loop).
- [ ] Reproducible: a single documented command reruns the result from clean clone + Docker pull.
- [ ] Metrics logged to W&B and copied into `docs/results/<project>.md`.
- [ ] Demo video (mp4/gif) recorded and pushed to HF.
- [ ] Short writeup (what/why/how/results/limits) committed.
- [ ] No secrets, no `*.pt`, no `__pycache__` committed to git.

---

## 4. Project ladder (overview)

| # | Project | Builds on | Headline deliverable |
|---|---|---|---|
| P0 | Foundation fix + eval harness | existing repo | Honest, reproducible G1 nav baseline + eval |
| P1 | Language-conditioned VLA (state) | P0 | G1 obeys varied NL navigation commands |
| P2 | Small world model from scratch | P0 data | Dreamer-mini learning "in imagination" |
| P3 | Vision VLA + Cosmos synthetic data | P1 | Appearance-robust vision nav policy |
| P4 | Cosmos controllable world sim | P1/P3 data | Action-conditioned Cosmos rollout + planning/eval |
| P5 | **CAPSTONE: integrated agent** | P1–P4 | Flagship: language→reason→see→plan humanoid demo |

---

## P0 — Foundation Fix + Evaluation Harness  ⏱ 2 weeks · 💻 15–30 GPU-hr

### Objective
Turn the *existing* G1 work into an **honest, reproducible baseline** and build the **evaluation harness** every later project reuses. This fixes the critical defect found in review: the current "marker navigation" and "language conditioning" are decorative (fixed command `"walk forward"`, fixed marker positions, reward never references command or marker — it's plain velocity tracking).

### Why it matters / what you learn
Honest RL task design, command-conditioned reward shaping, evaluation rigor, experiment tracking. A reviewer reading your repo must see real navigation, not spheres next to a locomotion policy.

### Prerequisites
- Working Isaac Lab Docker image (exists).
- `my-humanoid-project` importable in container (exists).

### What to build
- A *real* command-conditioned navigation task (randomized target + reward coupling).
- A reusable eval module under `programs/common/eval/` (success rate, fall rate, distance-to-target, episode length, video recorder).
- W&B integration + a results table generator.
- Repo hygiene cleanup.

### Checkpoints

**CP0.1 — Repo hygiene.**
Add `__pycache__/`, `*.pyc`, `*.pt` to `.gitignore`; `git rm --cached` tracked checkpoints/pyc; move existing `.pt` files to a HF model repo.
*DoD:* `git ls-files | grep -E '\.(pt|pyc)$'` returns empty; checkpoints downloadable from HF.
*Efficiency:* one-time; prevents bloated clones on every machine switch.

**CP0.2 — Randomized command + target.**
In the task config, at each `reset`: sample a target id ∈ {red, blue, …}, randomize that marker's position within a ring; store `env.target_id` and `env.target_pos` per env. Put `target_id` (one-hot or embedding) into the policy observation.
*DoD:* logging shows target id/pos vary per env per episode; observation tensor contains the target signal (print shape + sample).
*Efficiency:* no extra GPU cost; pure config.

**CP0.3 — Command-conditioned reward.**
Add a reward term: `+` for decreasing distance to the **commanded** marker, `−` penalty for approaching the wrong marker, `+` bonus on reaching commanded target, keep existing stability/velocity terms.
*DoD:* unit check — with a scripted policy walking to the correct marker, episode return rises; walking to wrong marker, return falls.
*Efficiency:* cheap; validate with 64 envs before scaling.

**CP0.4 — Train real baseline.**
Smoke test (256 envs, 50 iters) → full run (4096–8192 envs). Target: **success rate ≥ 70%** reaching the commanded marker.
*DoD:* W&B run shows success-rate curve climbing >0.7; checkpoint saved to HF.
*Efficiency:* 8192 envs, adaptive LR; stop on plateau (early-stop on success-rate).

**CP0.5 — Eval harness + results table.**
`programs/common/eval/evaluate.py`: loads a checkpoint, runs N eval episodes, outputs success/fall/dist/length + records 1 rollout mp4. Wire toward Isaac Lab-Arena conventions.
*DoD:* `python evaluate.py --ckpt <hf-path>` prints a metrics dict and writes `docs/results/p0_baseline.md` + an mp4.
*Efficiency:* reuse across P1–P5 — build it once, well.

### Success metrics
Commanded-target success ≥70%, fall rate <10%, eval reproducible from one command.

### Deliverables
Honest baseline checkpoint (HF), eval harness, `docs/results/p0_baseline.md`, rollout video, repo-hygiene commit.

### Risks & mitigations
- *Reward hacking* (robot spins near both markers): add wrong-target penalty + require facing/stopping. Inspect videos every checkpoint.

---

## P1 — Language-Conditioned VLA (state-based)  ⏱ 3–4 weeks · 💻 40–80 GPU-hr

### Objective
A *genuine* vision-language-action policy on **state** (no pixels yet): the G1 follows **varied natural-language commands** for **multi-goal, sequential** navigation with obstacles.

### Why it matters / what you learn
Real language grounding (frozen text encoder, not a hash), curriculum learning, teacher→student distillation, sequential/long-horizon reward. This is the conceptual core of "VLA" minus vision.

### Prerequisites
P0 complete (command-conditioned reward + eval harness).

### What to build
- Replace the hash embedding with a **frozen text encoder** (CLIP text / SentenceTransformer); cache embeddings for a fixed command set.
- Command grammar: `go to {color}`, `go to {color} then {color}`, `avoid obstacles and go to {color}`, style modifiers (`slowly`, `carefully`).
- Sequential subgoal tracking + obstacle assets + collision penalty.
- Optional: privileged-state **teacher** → **student** distillation.

### Checkpoints

**CP1.1 — Frozen text encoder integration.**
Swap `embedding_for_text` for a frozen encoder; cache `{command: vector}` to disk/HF; keep env interface unchanged.
*DoD:* embeddings for semantically similar commands have higher cosine similarity than dissimilar ones (print a similarity matrix); training still launches.
*Efficiency:* encode once offline, load cached tensors — zero per-step text-encoder cost.

**CP1.2 — Multi-goal conditioned policy.**
Train on randomized single commands over 3–5 colored targets.
*DoD:* per-command success ≥75%; wrong-target rate <10%; W&B breakdown by command.
*Efficiency:* shared policy, command in obs; 8192 envs.

**CP1.3 — Obstacles + collision avoidance.**
Add static→randomized obstacles; collision penalty; curriculum (wide gaps → narrow).
*DoD:* goal success ≥65% with obstacles, collision rate <15%.
*Efficiency:* curriculum auto-advances on success threshold to avoid wasted iters.

**CP1.4 — Sequential instructions.**
Two-step (`red then blue`) → three-step; phase variable in obs; reward per completed subgoal.
*DoD:* full-sequence completion ≥50% on two-step; correct-ordering ≥70%.
*Efficiency:* start two-step; only extend to three-step after two-step passes.

**CP1.5 — (Optional) teacher→student distillation.**
Train privileged teacher, distill to student with restricted obs.
*DoD:* student within 10% of teacher success.

### Success metrics
Multi-goal ≥75%, obstacle nav ≥65%, two-step sequence ≥50%, semantically-meaningful command embeddings.

### Deliverables
Language-conditioned checkpoint (HF), command-breakdown table, demo video showing the *same robot* obey different spoken commands, `docs/results/p1_language_vla.md`.

### Risks & mitigations
- *Command ignored* (policy solves task without reading command): verify by swapping the command mid-episode → behavior must change. Add this as an explicit eval probe.

---

## P2 — Small World Model From Scratch  ⏱ 4 weeks · 💻 30–60 GPU-hr

### Objective
Implement a **small world model in pure PyTorch** and train an agent that learns **in imagination**. Goal is *understanding internals*, not scale.

### Why it matters / what you learn
Latent dynamics (RSSM), reconstruction vs. prediction, imagination rollouts, model-based RL. This is the conceptual foundation that makes Cosmos (P3/P4) legible instead of magic.

### Prerequisites
P0 (you'll reuse logged G1 rollouts as one test domain). Independent of P1 — can run in parallel if time allows.

### What to build (choose ONE track at CP2.0)
- **Track A — Dreamer-mini (recommended):** RSSM (deterministic GRU + stochastic latent) + reward/continue heads + actor-critic trained on imagined latent rollouts.
- **Track B — Small video predictor:** a compact transformer or latent-diffusion next-frame predictor (a "tiny Cosmos") to directly motivate P4.

### Checkpoints

**CP2.0 — Pick track + domain.**
Choose Track A or B; pick a cheap domain (DeepMind Control / CarRacing / gridworld, *plus* your low-dim G1 nav rollouts).
*DoD:* design note committed stating track, domain, and the equations/architecture you'll implement.

**CP2.1 — Encoder/decoder + dynamics.**
Implement observation encoder, latent dynamics, decoder; train on logged transitions.
*DoD:* reconstruction loss converges; decoded predictions visually track ground truth on a held-out clip (side-by-side image saved).
*Efficiency:* low-res obs (64×64), small latent (≤256), bf16; toy domain keeps it <1 GPU-hr/epoch.

**CP2.2 — Multi-step prediction.**
Roll the model forward K steps without ground truth.
*DoD:* K-step prediction error reported; degradation curve plotted (error vs horizon).

**CP2.3 — Imagination learning (Track A).**
Train actor-critic purely on imagined latent rollouts; deploy in the real env.
*DoD:* agent trained *only* in imagination beats a random policy by a clear margin in the real env; "dream vs reality" rollout video saved.
*Efficiency:* imagination is cheap (no simulator in the loop) — this is the payoff to demonstrate.

**CP2.4 — Writeup.**
Explain RSSM/predictor internals with diagrams.
*DoD:* `docs/results/p2_world_model.md` published with architecture diagram + curves.

### Success metrics
Coherent K-step predictions; imagination-trained agent > random baseline; clear written explanation of internals.

### Deliverables
Pure-PyTorch WM repo, "learning in a dream" video, internals writeup (strong portfolio/interview piece).

### Risks & mitigations
- *Posterior collapse / blurry predictions:* start with a tiny deterministic model, add stochasticity only once deterministic works; KL balancing.

---

## P3 — Vision VLA + Cosmos Synthetic Data  ⏱ 4 weeks · 💻 50–100 GPU-hr

### Objective
Go from "blind" to **pixels**: a vision-based policy using the G1 head camera, made **appearance-robust** with **Cosmos-Transfer 2.5** photoreal synthetic data. This also resolves the known **Vulkan/`libGLX_nvidia.so.0`** camera blocker.

### Why it matters / what you learn
Vision policies (CNN/ViT encoders), rendering cost management, sim-to-real via world-foundation-model synthetic data — the most "industry-relevant" rung.

### Prerequisites
P1 (language conditioning) — vision is added on top. A GPU host with `nvidia-container-toolkit` graphics support.

### What to build
- Fix the rendering blocker; enable `TiledCamera` (config already exists in `g1_vla_vision_cfg.py`).
- CNN/ViT vision encoder feeding the policy (CNN runner config already drafted).
- A **Cosmos-Transfer 2.5** pipeline: take sim renders → generate photoreal/lighting/texture variations → train on the augmented distribution.

### Checkpoints

**CP3.1 — Unblock graphics.**
On the new host, verify Vulkan inside the container: `vulkaninfo --summary` succeeds; run the camera smoke test.
*DoD:* camera RGB observation shape prints (`[VLA] Camera RGB observation shape: ...`); pipeline reaches PPO at 16 envs / 32×32.
*Efficiency:* keep envs/resolution tiny until this passes — never debug graphics at scale.

**CP3.2 — Vision smoke → small train.**
Scale to NUM_ENVS=128, 128×128 camera (the CNN runbook defaults).
*DoD:* training stable for ≥200 iters without OOM; reward trending up.
*Efficiency:* TiledCamera, low update period, gradient checkpointing; one camera per env is the VRAM driver — find max envs empirically.

**CP3.3 — Vision baseline policy.**
Train vision nav to the commanded marker (state target hidden; must use pixels).
*DoD:* vision-only success ≥50%; probe confirms policy degrades when camera is masked (proves it uses pixels).

**CP3.4 — Cosmos-Transfer synthetic dataset.**
Use Cosmos-Transfer 2.5 to convert a batch of sim renders into N photoreal variants (lighting/material/background). Store dataset on HF.
*DoD:* dataset of paired (sim → photoreal) frames exists on HF; visual samples in writeup.
*Efficiency:* generate **once**, cache on HF; reuse for all robustness runs. Batch generation; 2B Transfer model.

**CP3.5 — Robustness training + eval.**
Train/augment with Cosmos-Transfer data; evaluate on held-out appearances.
*DoD:* success on shifted appearances improves vs CP3.3 baseline by a measurable margin (report both).

### Success metrics
Vision-only success ≥50%, demonstrable pixel dependence, appearance-robustness gain from Cosmos data.

### Deliverables
Vision policy checkpoint (HF), Cosmos-Transfer synthetic dataset (HF), before/after robustness table, vision rollout video, `docs/results/p3_vision_vla.md`.

### Risks & mitigations
- *Graphics still blocked:* this is a host config issue, not code. DoD CP3.1 gates everything — do not proceed until `vulkaninfo` passes. Fallback: a cloud host with confirmed graphics capability.
- *OOM from cameras:* lower resolution/env count; the existing trainer already fails fast on oversized camera env counts.

---

## P4 — Cosmos as a Controllable World Simulator  ⏱ 4 weeks · 💻 80–150 GPU-hr (prefer A100-80GB)

### Objective
Use **Cosmos Predict 2.5** as a **controllable, action-conditioned world simulator** of your robot: post-train (LoRA) on your task data, distill it, and use it to **predict rollouts** and do **short-horizon planning / policy evaluation** — a real World-Action-Model loop.

### Why it matters / what you learn
WFM post-training, action conditioning, distillation, model-predictive control in learned-model space. This is the frontier-skill rung that most directly connects to NVIDIA's WAM direction.

### Prerequisites
P1/P3 (you need logged action-conditioned robot data). The `cosmos-cookbook` Robot/Policy recipe.

### What to build
- A dataset of (frames, actions) from your G1 task in the cookbook's expected format.
- A **LoRA post-train** of Cosmos Predict 2.5 (2B) using the Robot/Policy / action-conditioned recipe.
- A **rollout predictor** + a simple **planner** (sample action sequences, score predicted outcomes, pick best — CEM/MPC).
- A **policy-eval** mode: compare predicted vs. real rollouts.

### Checkpoints

**CP4.1 — Inference baseline.**
Run stock Cosmos Predict 2.5 (2B) inference; generate a video from a prompt/initial frame.
*DoD:* a generated mp4 exists; VRAM/time recorded.
*Efficiency:* 2B model, bf16, offload if needed; confirm it fits before any training.

**CP4.2 — Data prep.**
Export G1 (frames, actions) into the cookbook's action-conditioned format; store on HF.
*DoD:* a dataloader yields aligned (frame_t, action_t, frame_t+1) batches; shapes printed.

**CP4.3 — LoRA post-train (smoke → real).**
Smoke (tiny subset, few steps) → real post-train using the Robot/Policy recipe.
*DoD:* training loss decreases; an action-conditioned generation visibly responds to *different* input actions (two actions → two different predicted futures, saved side by side).
*Efficiency:* **LoRA only**, gradient checkpointing, 8-bit optimizer; this is the budget-critical run — smoke test hard before committing GPU-hours.

**CP4.4 — Action-conditioned rollout / distillation.**
Distill into a faster controllable simulator (per cookbook distillation guide); roll out K steps conditioned on an action plan.
*DoD:* K-step action-conditioned rollout produced; fidelity vs real env reported.

**CP4.5 — Planning + policy eval.**
Implement CEM/MPC over the learned model to reach a target; and a policy-eval that scores P1/P3 policies via predicted rollouts.
*DoD:* planner reaches a goal in the *real* env using only model-predicted lookahead on ≥1 task; policy-eval rankings correlate with real eval.
*Efficiency:* short horizons (8–16 steps); cache generations.

### Success metrics
Action-conditioned predictions that respond to actions; a working short-horizon planner; predicted-vs-real eval correlation.

### Deliverables
LoRA-post-trained Cosmos checkpoint (HF), "imagine the rollout" prediction videos, planning demo, `docs/results/p4_cosmos_world_sim.md`.

### Risks & mitigations
- *Compute overrun (biggest risk):* this project alone can blow the budget. Hard-gate with CP4.3 smoke; if 2B LoRA won't fit/finish on the single GPU, restrict scope to **inference + planning on stock Cosmos** and document the post-train as future work. Consider a one-off burst rental only for this run.

---

## P5 — CAPSTONE: Integrated Language→Reason→See→Plan Humanoid  ⏱ 6–8 weeks · 💻 100–200 GPU-hr

### Objective
Integrate P1–P4 into **one agent**: a natural-language command is **reasoned/decomposed** (Cosmos-Reason 2 style), executed by the **vision VLA policy** (P3), with the **world model in the loop** (P2/P4) for lookahead/safety, evaluated in **Isaac Lab-Arena** and stress-tested with **Cosmos-Transfer** photoreal scenes. This is the portfolio centerpiece.

### Why it matters / what you learn
Full physical-AI stack integration and systems thinking — the thing that signals "can build real robot AI," not just train one model.

### Prerequisites
P1, P3 mandatory; P2 and P4 in-the-loop components as available (degrade gracefully if P4 stayed inference-only).

### What to build
- **Reasoning layer:** Cosmos-Reason 2 (or GR00T N1.7's reasoning) decomposes a high-level instruction (`"tidy the area: visit red, avoid the box, then blue, slowly"`) into subgoals.
- **Execution layer:** the P3 vision VLA policy executes subgoals.
- **World-model layer:** P4/P2 model does short lookahead to veto unsafe actions / pick among options.
- **Eval layer:** Isaac Lab-Arena scenarios + Cosmos-Transfer photoreal eval.

### Checkpoints

**CP5.1 — Architecture spec.**
Define module interfaces (reason → subgoal queue → policy → world-model check → env) and data contracts.
*DoD:* an interface diagram + a stub pipeline runs end-to-end with mock modules.

**CP5.2 — Reasoning → subgoals.**
Wire Cosmos-Reason 2 / GR00T reasoning to emit an ordered subgoal list from a free-form instruction.
*DoD:* 10 sample instructions produce correct subgoal decompositions (report accuracy).

**CP5.3 — End-to-end execution.**
Reasoner output drives the vision policy through a multi-step task.
*DoD:* full instruction completed end-to-end in ≥1 scenario; video saved.

**CP5.4 — World-model in the loop.**
Insert lookahead (P4 rollout or P2 model) to choose/veto actions at decision points.
*DoD:* a documented case where lookahead changes the outcome (e.g., avoids a predicted collision); ablation with/without.

**CP5.5 — Arena + photoreal eval + final cut.**
Evaluate across Isaac Lab-Arena scenarios; render Cosmos-Transfer photoreal eval; produce the flagship demo video + writeup + landing page.
*DoD:* Arena scorecard table; ≥1 photoreal eval clip; polished demo video; project landing page published.

### Success metrics
End-to-end instruction completion on multiple scenarios; measurable benefit from the world-model-in-the-loop; a portfolio-grade demo.

### Deliverables
Integrated agent repo, Arena scorecard, **flagship demo video**, project landing page + blog, `docs/results/p5_capstone.md`.

### Risks & mitigations
- *Integration sprawl:* freeze each upstream component (no retraining during P5); integrate against fixed checkpoints. If P4 stayed inference-only, use P2's small WM for the lookahead layer.

---

## 5. Cross-cutting portfolio track (runs through all projects)

For "job in physical AI," the *artifacts* are half the value:
- **Per project:** a 1–2 min demo video, a clean tagged GitHub release, and a 600–1000 word blog post (problem → approach → result → what I learned).
- **Capstone:** a single landing page (project site) linking all five, leading with the P5 demo.
- **Narrative thread:** "I built the physical-AI stack bottom-up: real RL task design → language grounding → world-model internals → world-foundation-models at scale → an integrated reasoning agent." That sentence is your interview pitch.
- **Honesty:** label what is real vs. in-progress (the P0 lesson). Demonstrable + honest beats impressive + unverifiable.

---

## 6. Sequencing & parallelism (6-month calendar)

| Month | Primary | Parallel/portfolio |
|---|---|---|
| 1 | P0 (wk1–2) → P1 start (wk3–4) | repo hygiene, W&B, blog template |
| 2 | P1 finish | P2 can start in parallel (independent) |
| 3 | P2 finish + P3 start | P0/P1/P2 blog posts |
| 4 | P3 finish | P3 blog + dataset on HF |
| 5 | P4 | P4 is budget-critical — watch GPU-hours |
| 6 | P5 capstone | flagship video + landing page |

---

## 7. Glossary & references

- **VLA** — Vision-Language-Action model (maps instruction + perception → robot actions).
- **WFM / WM** — World Foundation Model / world model (predicts future world state; enables planning + synthetic data).
- **WAM** — World-Action-Model (Cosmos 3 direction: controllable physics-grounded simulator coupled to a policy).
- **RSSM** — Recurrent State-Space Model (Dreamer's latent dynamics core).
- **LoRA** — Low-Rank Adaptation (cheap fine-tuning; mandatory on our budget).

**Links (verified June 2026):**
- Cosmos Predict 2.5: https://github.com/nvidia-cosmos/cosmos-predict2.5
- Cosmos Predict/Transfer 2.5 overview: https://huggingface.co/blog/nvidia/cosmos-predict-and-transfer2-5
- Cosmos org + cookbook: https://github.com/nvidia-cosmos
- Cosmos platform: https://www.nvidia.com/en-us/ai/cosmos/
- Cosmos WFM paper: https://arxiv.org/abs/2511.00062
- Isaac GR00T N1.7: https://github.com/NVIDIA/Isaac-GR00T · https://huggingface.co/blog/nvidia/gr00t-n1-7
- Isaac GR00T: https://developer.nvidia.com/isaac/gr00t
- Isaac Lab-Arena: https://developer.nvidia.com/blog/simplify-generalist-robot-policy-evaluation-in-simulation-with-nvidia-isaac-lab-arena/
- WFM synthetic trajectory data: https://developer.nvidia.com/blog/enhance-robot-learning-with-synthetic-trajectory-data-generated-by-world-foundation-models/

---

*End of plan. Each project's checkpoints are independently executable; do not skip a checkpoint's Definition of Done before advancing.*
