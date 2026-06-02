# Tabletop Manipulation Program: Imitation → Language-Conditioned Manipulation → Cosmos World Models

**Owner:** Sushruth · **Created:** 2026-06-02 · **Status:** APPROVED-DRAFT (pending owner review)
**Anchor:** Robot arm tabletop manipulation (Franka / LIBERO / RoboCasa) — **runs in parallel to the humanoid program**, does not replace it.
**Companion document:** `docs/PHYSICAL_AI_6MONTH_PLAN.md` (humanoid track). Shared rules live there; this doc references them instead of duplicating.

---

## 0. How to use this plan

Same rules as the humanoid plan. In particular, reuse **everything in §0–§3 of `PHYSICAL_AI_6MONTH_PLAN.md`**: the agent-onboarding notes, the **golden efficiency rules** (LoRA, bf16, gradient checkpointing, smoke-test-first, cache Cosmos generations, checkpoint to HF), the **universal Definition-of-Done gates**, and the experiment-tracking conventions. This document only adds the manipulation-specific projects (**T0–T5**) and how they interlock with the humanoid projects (**P0–P5**).

Checkpoints (`CPTn.m`) each carry a **verifiable Definition of Done** and an **efficiency note**. Do not advance until the DoD is observed.

---

## 1. Why a second, manipulation track (and why it's cheap to add)

- **Cosmos is more mature here.** Cosmos Predict 2.5 ships **Robot/Policy** post-training recipes for **RoboCasa** and **LIBERO** — both are *manipulation* benchmarks. GR00T N1.7 (which you already fine-tuned) is primarily a *manipulation* VLA. So the Cosmos/GR00T rungs (T4/T5) produce crisper, more standard, more recognizable portfolio results than humanoid nav.
- **Different, complementary skills.** Humanoid track teaches locomotion RL + navigation. Manipulation track teaches **imitation learning** (diffusion policy / ACT), **dexterous control**, **grasping**, and **benchmark-driven evaluation** — the other half of what physical-AI employers look for.
- **Shared spine = less work.** T0 reuses the P0 eval harness; T2 reuses the P2 world-model code; T4 reuses the P4 Cosmos post-training infrastructure. Estimate: the manipulation track is **~60% of the cost** of the humanoid track because of this reuse.

### Domain & framework choices (decided defaults)
- **Simulator:** Isaac Lab manipulation envs (Franka) + **LIBERO** benchmark tasks (language-conditioned, standard). Optionally **RoboCasa** for kitchen-scale scenes (matches Cosmos recipe).
- **Imitation framework:** **LeRobot** (Hugging Face) for datasets + **Diffusion Policy / ACT** baselines — keeps everything in the HF ecosystem you already use.
- **VLA backbone:** **GR00T N1.7** (already fine-tuned in this repo) as the manipulation policy backbone in T1/T5.

---

## 2. Compute & cost (single GPU)

Reuse the budget model in the humanoid doc. Manipulation is generally **cheaper than humanoid vision** (no 8192-env locomotion sims; imitation learning runs on modest batch sizes), except T4 (Cosmos post-train) which mirrors P4's cost.

| Project | GPU-hr (est.) | Driver |
|---|---|---|
| T0 | 10–20 | Env + eval setup, scripted/BC baseline |
| T1 | 30–60 | GR00T N1.7 LoRA fine-tune on demos |
| T2 | 20–40 | Diffusion policy / ACT + small WM reuse |
| T3 | 40–80 | Vision manipulation + Cosmos-Transfer data |
| T4 | 80–150 | Cosmos Predict RoboCasa/Libero recipe (prefer A100-80GB) |
| T5 | 80–160 | Integration + eval |
| **Total** | **~260–510 GPU-hr** | ≈ **$500–1,250** on-demand |

---

## 3. Repo layout (extends the `programs/` tree)

```
programs/
  ...                    # p0..p5 from humanoid plan
  t0_manip_foundation/   # manipulation env + eval + BC baseline
  t1_groot_language/     # language-conditioned manipulation (GR00T N1.7 LoRA)
  t2_imitation_wm/       # diffusion policy / ACT + world-model-assisted
  t3_vision_manip_cosmos/# vision manipulation + Cosmos-Transfer data
  t4_cosmos_robocasa/    # Cosmos Predict RoboCasa/LIBERO recipe
  t5_manip_capstone/     # integrated language-vision manipulation agent
  common/                # SHARED with humanoid track (eval, text encoders, video)
```

---

## T0 — Manipulation Foundation + Eval  ⏱ 1.5 weeks · 💻 10–20 GPU-hr

### Objective
Stand up a tabletop manipulation env + benchmark and a manipulation eval harness (reusing `programs/common/eval`). Establish a grasp/place baseline.

### What to build
- Isaac Lab Franka pick/lift/stack env + LIBERO task suite wired in.
- Manipulation metrics in the shared eval module: **grasp success, place success, task success, steps-to-success, object-drop rate**.
- A scripted or behavior-cloning baseline.

### Checkpoints
**CPT0.1 — Env + benchmark up.** Install LIBERO + Franka env; render one task.
*DoD:* a single manipulation episode runs headless and records an mp4; task list prints.
*Efficiency:* CPU/low-env debugging first; GPU only for rollouts.

**CPT0.2 — Manipulation eval harness.** Extend shared eval with grasp/place/task-success.
*DoD:* `evaluate.py --suite libero --ckpt <x>` prints metric dict + writes `docs/results/t0_manip.md`.

**CPT0.3 — BC baseline.** Behavior-clone a small demo set (LeRobot dataset).
*DoD:* BC policy beats scripted random on ≥1 task; success-rate logged.
*Efficiency:* small dataset, short epochs; this is a smoke baseline, not SOTA.

### Deliverables
Manipulation env + eval harness, BC baseline, `docs/results/t0_manip.md`, rollout video.

---

## T1 — Language-Conditioned Manipulation (GR00T N1.7)  ⏱ 3–4 weeks · 💻 30–60 GPU-hr

### Objective
A real language-conditioned manipulation policy: `"pick up the red block"`, `"put the cup on the plate"` — multi-object, instruction-grounded. This is where your existing **GR00T N1.7** fine-tune becomes a first-class portfolio piece.

### What to build
- LeRobot-format demo dataset across multiple objects/instructions (teleop or scripted-expert generated).
- **LoRA fine-tune of GR00T N1.7** on the dataset (reuse your Phase-1 GR00T pipeline).
- Instruction-grounding evaluation (does the right object get manipulated?).

### Checkpoints
**CPT1.1 — Multi-instruction dataset.** Generate/collect demos with varied object+instruction pairs in LeRobot format; push to HF.
*DoD:* dataset on HF with N instructions over M objects; loader yields (obs, instruction, action) batches.

**CPT1.2 — GR00T N1.7 LoRA fine-tune (smoke → real).** Smoke (few steps) → full LoRA fine-tune.
*DoD:* train loss decreases; policy executes ≥1 instruction successfully on held-out scene; checkpoint on HF.
*Efficiency:* **LoRA only**, bf16, gradient checkpointing — reuse Phase-1 GR00T config.

**CPT1.3 — Instruction grounding eval.** Probe: swap the instruction, confirm the manipulated object changes.
*DoD:* per-instruction success ≥60%; wrong-object manipulation <20%; mid-episode instruction swap changes behavior.

### Deliverables
GR00T-N1.7 manipulation checkpoint (HF), instruction-grounding table, demo video (same arm, different spoken commands), `docs/results/t1_groot_language.md`.

### Risks & mitigations
- *Policy ignores language:* the instruction-swap probe (CPT1.3) is the gate — must pass.

---

## T2 — Imitation + World-Model-Assisted Manipulation  ⏱ 3 weeks · 💻 20–40 GPU-hr

### Objective
Learn the modern imitation-learning toolkit (**Diffusion Policy**, **ACT**) and connect manipulation to the **small world model from P2** for prediction/replanning.

### Why it matters
Diffusion Policy / ACT are the dominant manipulation-policy architectures in industry. Plus you reuse P2's world model on a manipulation domain — showing the WM generalizes.

### Checkpoints
**CPT2.1 — Diffusion Policy / ACT baseline.** Train both on the T1 dataset; compare to GR00T.
*DoD:* both train and evaluate; a comparison table (success, smoothness, inference latency) is produced.
*Efficiency:* action chunking reduces inference cost; small models.

**CPT2.2 — World-model on manipulation.** Retrain/adapt the P2 mini-WM on manipulation rollouts.
*DoD:* K-step prediction of object/gripper state plotted; error-vs-horizon curve.

**CPT2.3 — WM-assisted replanning.** Use the WM to pick among candidate action chunks.
*DoD:* a documented case where WM-based selection improves task success vs. raw policy (ablation).

### Deliverables
Diffusion Policy + ACT checkpoints, architecture comparison table, WM-assisted manipulation writeup, `docs/results/t2_imitation_wm.md`.

---

## T3 — Vision Manipulation + Cosmos-Transfer Data  ⏱ 3–4 weeks · 💻 40–80 GPU-hr

### Objective
Pixel-based manipulation made robust to appearance via **Cosmos-Transfer 2.5** photoreal synthetic data (same technique as humanoid P3, applied to a tabletop scene).

### Checkpoints
**CPT3.1 — Vision manipulation baseline.** Camera-based policy (wrist/overhead cam).
*DoD:* vision-only task success ≥40%; masked-camera probe confirms pixel dependence.
*Efficiency:* reuse P3's graphics fix; low-res wrist cam.

**CPT3.2 — Cosmos-Transfer manipulation dataset.** Generate photoreal variants (lighting/material/clutter) of tabletop renders; store on HF.
*DoD:* paired sim→photoreal manipulation dataset on HF; samples in writeup.
*Efficiency:* generate once, cache, reuse.

**CPT3.3 — Robustness training + eval.** Augment with Cosmos data; eval on held-out appearances/distractors.
*DoD:* success under appearance/distractor shift improves vs. CPT3.1 (report both).

### Deliverables
Vision manipulation checkpoint, Cosmos-Transfer manipulation dataset (HF), robustness table, video, `docs/results/t3_vision_manip.md`.

---

## T4 — Cosmos Predict for Manipulation (RoboCasa / LIBERO recipe)  ⏱ 4 weeks · 💻 80–150 GPU-hr (prefer A100-80GB)

### Objective
The flagship technical rung: run NVIDIA's **Cosmos Predict 2.5 Robot/Policy** post-training recipe for **RoboCasa/LIBERO** — action-conditioned world model for manipulation, used for **synthetic trajectory generation**, **policy evaluation**, and **planning**. This is the most "mature path" and the crispest portfolio result of either track.

### Why it matters
This is the exact workflow NVIDIA ships for manipulation; doing it end-to-end is a strong, recognizable hire signal. It directly reuses P4's Cosmos infrastructure.

### Checkpoints
**CPT4.1 — Recipe inference baseline.** Run the cosmos-cookbook RoboCasa/LIBERO policy model inference as published.
*DoD:* reproduce a published-style generation/rollout; record VRAM/time.
*Efficiency:* 2B class; offload if needed; confirm fit before training.

**CPT4.2 — Data prep in recipe format.** Convert your T1/T3 manipulation data to the cookbook's expected format.
*DoD:* dataloader yields recipe-format batches; shapes verified.

**CPT4.3 — LoRA post-train (smoke → real).** Post-train on your tasks.
*DoD:* loss decreases; action-conditioned generation responds to different action inputs (two actions → two futures).
*Efficiency:* **LoRA**, gradient checkpointing, 8-bit optimizer; smoke-gate hard (budget-critical).

**CPT4.4 — Synthetic trajectory augmentation.** Use the WFM to generate synthetic manipulation trajectories; augment T1/T2 training; measure policy gain.
*DoD:* a policy trained with WFM-augmented data beats the non-augmented baseline (report delta). This is a headline result.

**CPT4.5 — Policy eval + planning.** Score T1/T2/T3 policies via predicted rollouts; short-horizon CEM/MPC on ≥1 task.
*DoD:* predicted-vs-real eval correlation reported; planner reaches a goal via model lookahead.

### Deliverables
Cosmos manipulation checkpoint (HF), synthetic-trajectory augmentation result (the headline), planning/eval demo, `docs/results/t4_cosmos_robocasa.md`.

### Risks & mitigations
- *Compute overrun:* same as P4 — hard-gate at CPT4.3 smoke; fall back to inference + synthetic-data-gen on stock Cosmos if LoRA post-train won't fit. Consider one burst rental for this run.

---

## T5 — CAPSTONE: Language-Vision Manipulation Agent  ⏱ 5–7 weeks · 💻 80–160 GPU-hr

### Objective
Integrate T1–T4 into one manipulation agent: free-form instruction → (Cosmos-Reason 2 / GR00T reasoning) decomposes into manipulation subgoals → vision manipulation policy executes → world model verifies/replans → evaluated on LIBERO/RoboCasa + Cosmos-Transfer photoreal scenes.

### Checkpoints
**CPT5.1 — Architecture + interfaces.** Define reason → subgoal → policy → WM-check → env contracts; mock end-to-end run.
*DoD:* interface diagram + stub pipeline runs end-to-end.

**CPT5.2 — Reasoning → manipulation subgoals.** Wire reasoning model to emit ordered manipulation steps (e.g., `"set the table"` → pick/place sequence).
*DoD:* 10 instructions decomposed correctly (accuracy reported).

**CPT5.3 — End-to-end multi-step manipulation.** Reasoner drives the vision policy through a multi-step task.
*DoD:* full instruction completed end-to-end on ≥1 scenario; video saved.

**CPT5.4 — WM-in-the-loop.** Insert lookahead to veto failing grasps / pick among options.
*DoD:* a documented case where lookahead prevents a failure; ablation with/without.

**CPT5.5 — Benchmark eval + final cut.** Evaluate on LIBERO/RoboCasa suites + Cosmos-Transfer photoreal; produce flagship demo + writeup + landing page.
*DoD:* benchmark scorecard; photoreal eval clip; polished demo video; landing page.

### Deliverables
Integrated manipulation agent, LIBERO/RoboCasa scorecard, **flagship manipulation demo video**, landing page + blog, `docs/results/t5_manip_capstone.md`.

---

## 4. How the two programs combine (humanoid P0–P5 + manipulation T0–T5)

### Shared spine (build once, use twice)
| Shared asset | Built in | Reused in |
|---|---|---|
| Eval harness (`common/eval`) | P0 | T0 (add manip metrics) |
| Frozen text encoder + command cache | P1 | T1 |
| Small world model (Dreamer-mini) | P2 | T2 |
| Graphics/Vulkan fix + vision pipeline | P3 | T3 |
| Cosmos post-train infrastructure | P4 | T4 |
| Reasoning→subgoal→WM-check integration pattern | P5 | T5 |

### Realistic combined timeline (single GPU)
Two full ladders ≠ 6 months. Three honest options:

- **Option 1 — Sequential (~10–12 months):** finish humanoid P0–P5, then manipulation T0–T5 (faster due to reuse). Lowest risk, cleanest portfolio.
- **Option 2 — Interleaved (~8–9 months):** build the shared spine once (P0/T0 together, P2 once, P4/T4 Cosmos infra together), then fork the embodiment-specific projects. Most efficient; recommended if you want both.
- **Option 3 — Converged capstone (~7–8 months):** run both tracks through T4/P4, then **merge the two capstones into one loco-manipulation flagship** (a humanoid that navigates *and* manipulates — `"walk to the table and put the red block on the plate"`). Highest-impact single portfolio piece; highest integration risk.

### The convergence opportunity (optional, high-value)
The strongest possible portfolio centerpiece is a **single loco-manipulation agent**: humanoid navigation (P-track) + arm/dexterous manipulation (T-track) under one language interface, with Cosmos in the loop. This is exactly NVIDIA's GR00T direction. If you want this, the merged capstone replaces P5+T5 with one combined flagship — but keep both full ladders intact up to that point so each domain is independently demonstrable.

---

## 5. Recommended next decision
Pick a combined-timeline option (1 sequential / 2 interleaved / 3 converged). My recommendation: **Option 2 (interleaved)** for efficiency, with an eye toward the **Option 3 converged capstone** as a stretch goal — it gives you both independent tracks *and* the headline loco-manipulation demo.

*References: same as `PHYSICAL_AI_6MONTH_PLAN.md` §7, plus LIBERO and RoboCasa benchmark suites and the Cosmos Predict 2.5 Robot/Policy recipe in the cosmos-cookbook.*
