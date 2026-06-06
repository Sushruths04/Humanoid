# Master Roadmap — Option 3: Converged Loco-Manipulation Flagship

**Owner:** Sushruth · **Created:** 2026-06-02 · **Updated:** 2026-06-06 · **Status:** GOVERNING PLAN (chosen path)
**This document governs.** It sequences the two source plans and replaces their separate capstones (P5 + T5) with one merged flagship (**C5**).

- Humanoid projects **P0–P4**: see `docs/PHYSICAL_AI_6MONTH_PLAN.md`
- Manipulation projects **T0–T4**: see `docs/TABLETOP_MANIPULATION_PLAN.md`
- Converged capstone **C5**: defined in §3 below (supersedes P5 and T5)
- Per-task GPU/VRAM sizing (rent the right card): `docs/GPU_VRAM_REQUIREMENTS.md`

**Target duration:** ~7–8 months, single GPU. **Strategy:** build each shared component once, run both embodiments through it, then merge into a single language-driven loco-manipulation agent.

---

## 1. The strategy in one picture

```
            ┌── nav variant ──► P1 ─► P2 ─► P3 ─► P4 ──┐
SHARED SPINE┤                                          ├──► C5  (loco-manipulation flagship)
            └── manip variant ─► T1 ─► T2 ─► T3 ─► T4 ──┘
   (P0+T0)
 eval harness,
 text encoder,
 world model,
 vision/graphics,
 Cosmos infra
```

Every shared asset is **built once** and consumed by both the navigation (P) and manipulation (T) variant, then both feed the converged capstone.

---

## 2. Phase calendar (~7–8 months, 34 weeks)

| Phase | Weeks | Build-once shared work | P-variant | T-variant | Phase gate (must pass to advance) |
|---|---|---|---|---|---|
| **Ph0 — Foundations** | 1–3 | `common/eval` harness; repo hygiene; W&B | **P0** honest G1 nav baseline + real command reward | **T0** Franka/LIBERO env + manip metrics + BC baseline | Both eval harnesses reproducible from one command; P0 success ≥70%, T0 BC > random |
| **Ph1 — Language** | 4–8 | Frozen text encoder + command cache (built once) | **P1** multi-goal/sequential NL navigation | **T1** GR00T N1.7 LoRA language manipulation | Instruction-swap probe passes on **both** (behavior changes with command) |
| **Ph2 — World model** | 9–12 | Dreamer-mini WM implemented once (P2) | P2 applied to nav rollouts | **T2** WM adapted to manip + Diffusion Policy/ACT | Imagination-trained agent > random (nav) AND DP/ACT comparison table (manip) |
| **Ph3 — Vision + Cosmos data** | 13–18 | Vulkan/graphics fix; Cosmos-Transfer pipeline (built once) | **P3** vision nav, appearance-robust | **T3** vision manip, appearance-robust | Pixel-dependence probe passes on both; robustness gain reported on both |
| **Ph4 — Cosmos world sim** | 19–26 | Cosmos Predict post-train infra + cookbook setup (once) | **P4** action-conditioned nav rollouts + planning | **T4** RoboCasa/LIBERO recipe + synthetic-traj augmentation | Action-conditioned generation responds to actions on both; T4 augmentation beats baseline |
| **Ph5 — Converged capstone** | 27–34 | — | — merged — | — merged — | **C5** end-to-end loco-manipulation demo + Arena scorecard |

**Budget (single GPU, on-demand):** combined ≈ **480–900 GPU-hr ≈ $900–2,100** over 7–8 months. Ph4 is the cost spike (two Cosmos post-trains) — apply the LoRA/smoke-gate discipline hard, and consider a one-off A100-80GB burst rental for Ph4 only.

**Parallelism note (single GPU):** "build-once" work is sequential on one GPU, but the P- and T-variants within a phase share code and can be trained back-to-back in the same week. Do the shared component → smoke-test the P-variant → smoke-test the T-variant → launch the longer of the two overnight.

---

## 3. C5 — CONVERGED CAPSTONE: Language-Driven Loco-Manipulation  ⏱ 8 weeks · 💻 120–220 GPU-hr

### Objective
One humanoid agent that **navigates and manipulates** under a single natural-language interface, with Cosmos reasoning + world-model lookahead, evaluated in Isaac Lab-Arena and Cosmos-Transfer photoreal scenes.

**Canonical demo instruction:**
> *"Walk to the table and put the red block on the plate."*
decomposes to → `[navigate: table]` → `[manipulate: pick red block]` → `[manipulate: place on plate]`.

### Architecture (modules + contracts)
```
NL instruction
   ▼
Reasoning layer  (Cosmos-Reason 2 / GR00T N1.7 reasoning)
   ▼  ordered subgoal list, each tagged {NAV | MANIP}
Skill router
   ├─ NAV subgoal  ─► P-track navigation policy (P3/P4)
   └─ MANIP subgoal ─► T-track manipulation policy (T3/T4 / GR00T)
   ▼  each action candidate
World-model lookahead  (P2/P4 nav WM · T2/T4 manip WM)  → veto/select
   ▼
G1 whole-body env  (Isaac Lab; arms enabled)
   ▼
Eval: Isaac Lab-Arena scenarios + Cosmos-Transfer photoreal
```

### Checkpoints

**CPC5.1 — Unified embodiment.** Enable the G1's arm/hand joints alongside locomotion in one Isaac Lab env (currently arms are effectively unused in the nav task).
*DoD:* a single G1 instance can both locomote and actuate its arms in the same episode; both action spaces exposed; smoke test reaches the control loop.
*Efficiency:* start with the **upper body fixed-base at the table** (manipulation only) and a **separate walking** scenario; do not attempt simultaneous whole-body control until CPC5.3.

**CPC5.2 — Skill router from reasoning.** Reasoning layer emits an ordered, tagged subgoal list; router dispatches each to the correct policy.
*DoD:* 10 mixed instructions decompose and route to the correct policy sequence (accuracy reported).

**CPC5.3 — Staged hand-off (the core integration).** Execute `navigate → arrive at table → switch to manipulation → pick → place` as a staged sequence with a stable transition between controllers.
*DoD:* full canonical instruction completed end-to-end on ≥1 scenario; video saved. (Staged controller switch is acceptable — simultaneous whole-body control is a stretch, see fallback.)

**CPC5.4 — World-model in the loop.** Insert lookahead at decision points (e.g., predict a grasp will fail → re-approach; predict a collision en route → reroute).
*DoD:* a documented case where lookahead changes the outcome; ablation with/without.

**CPC5.5 — Benchmark + photoreal eval.** Evaluate across Isaac Lab-Arena loco-manipulation scenarios; render Cosmos-Transfer photoreal eval clips.
*DoD:* Arena scorecard table; ≥1 photoreal eval clip; success rate on the multi-step task reported.

**CPC5.6 — Flagship cut.** Produce the polished demo video, the project landing page, and the capstone writeup.
*DoD:* landing page published linking all P/T projects; flagship video leads with the canonical instruction executed end-to-end.

### Success metrics
End-to-end loco-manipulation completion on ≥2 scenarios; measurable world-model-in-loop benefit; an Arena scorecard; a portfolio-grade flagship demo.

### Risk & graceful degradation (read this before Ph5)
Whole-body humanoid loco-manipulation is genuinely hard and may exceed a single-GPU budget. Degrade in this order — each level is still a strong portfolio result:
1. **Full:** simultaneous whole-body walk-and-manipulate.
2. **Staged (target):** navigate, stop, then manipulate with a controller hand-off (CPC5.3). **This is the planned deliverable.**
3. **Stitched fallback:** if even the staged hand-off is unstable, present the **separate P5-style nav demo and T5-style manip demo unified by one shared language/reasoning front-end** — same instruction interface, two embodiment demos. Document whole-body control as future work.

Do not let Ph5 perfectionism sink the program — ship level 2, attempt level 1 only if time/compute remain.

---

## 4. Definition of "done" for the whole program
- All phase gates passed (table in §2).
- P0–P4 and T0–T4 each have a result doc + demo video + HF checkpoint.
- C5 flagship video + landing page published.
- One-paragraph interview pitch written: *"I built the physical-AI stack bottom-up across two embodiments — honest RL task design, language grounding, world-model internals, world-foundation-models at scale — and converged them into a single language-driven loco-manipulation humanoid with Cosmos in the loop."*

---

## 5. Progress Snapshot (2026-06-06)

| Task | Result | Status |
|---|---|---|
| P0 CommandNav | 94.5% success, 7.8% fall rate | ✅ Done |
| P1.2 LangNav | 98.8% per-command | ✅ Done |
| P1.3 ObstacleNav | 85.9% success | ✅ Done |
| P1.4 SeqNav | 80.9% full-seq / 94.5% ordering | ✅ Done |
| P2 World Model | loss 0.76→0.011 | ✅ Done |
| T0 BC baseline | 50% task_success (libero_spatial:0) | ✅ Done |
| T1 GR00T N1.7 eval | **97.0%** mean success (10 tasks) | ✅ Done |
| T2 WM Manipulation | loss 1.40→0.008 | ✅ Done |
| T3 Pixel BC | 0% success (ablation — intended) | ✅ Done |
| **P3 Vision Nav** | Not started | 🔜 **NEXT — L40S** |
| P4 Cosmos Predict | Not started | 🔜 Future |
| C5 Capstone | Not started | 🔜 Future |

## 6. Immediate next action

**P3 Vision Nav** — rent L40S (48GB). See `docs/NEW_MACHINE_SETUP.md` for full setup runbook. Scripts scaffolded at `programs/p3_vision_nav/`. DoD: pixel-conditioned nav policy ≥60% success on CommandNav.

*Source plans: `docs/PHYSICAL_AI_6MONTH_PLAN.md`, `docs/TABLETOP_MANIPULATION_PLAN.md`. References: see those docs' §7.*
