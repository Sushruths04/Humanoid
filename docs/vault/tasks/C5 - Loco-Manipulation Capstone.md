---
tags: [task, c5, capstone, loco-manipulation, cosmos-reason, skill-router, whole-body, future]
---

# C5 — Language-Driven Loco-Manipulation (Capstone)

**Status**: 🔜 Not Started — needs L40S 48GB  
**DoD**: One humanoid agent that navigates AND manipulates under a single natural-language interface; end-to-end staged hand-off demonstrated on ≥2 scenarios; flagship demo video published  
**GPU**: L40S 48 GB  
**Estimated cost**: 120–220 GPU-hr · 6–8 weeks coding + training

---

## Canonical Demo

> *"Walk to the table and put the red block on the plate."*

Decomposes to → `[navigate: table]` → `[manipulate: pick red block]` → `[manipulate: place on plate]`

---

## Architecture

```
NL instruction
      ▼
Reasoning layer  (Cosmos-Reason 2 / GR00T N1.7 reasoning)
      ▼  ordered subgoal list, each tagged {NAV | MANIP}
Skill router
      ├─ NAV subgoal  ─► P3/P4 nav policy (existing)
      └─ MANIP subgoal ─► T1 GR00T manipulation policy (existing)
      ▼  each action candidate
World-model lookahead  (P2/P4 WM) → veto / select
      ▼
G1 whole-body env  (Isaac Lab; arms enabled)
      ▼
Eval: Isaac Lab-Arena + Cosmos-Transfer photoreal
```

**~70% reuse** of existing components (P3 nav, T1 GR00T, P2/T2 WMs). New work = unified env + skill router + staged hand-off + world-model veto.

---

## Checkpoints

### CPC5.1 — Unified Embodiment
Enable G1's arm/hand joints alongside locomotion in one Isaac Lab env.  
**DoD:** G1 can both locomote and actuate arms in the same episode; both action spaces exposed; smoke test reaches the control loop.  
**Strategy:** Start with upper body fixed-base (manipulation only) + separate walking scenario. Do NOT attempt simultaneous whole-body until CPC5.3.  
**Status:** 🔜

---

### CPC5.2 — Skill Router from Reasoning
Reasoning layer emits ordered, tagged subgoal list; router dispatches each to correct policy.  
**DoD:** 10 mixed instructions decompose and route to correct policy sequence (accuracy reported).  
**Status:** 🔜

---

### CPC5.3 — Staged Hand-off *(core integration target)*
Execute `navigate → arrive at table → switch to manipulation → pick → place` as staged sequence.  
**DoD:** Full canonical instruction completed end-to-end on ≥1 scenario; video saved. Controller switch is acceptable — simultaneous whole-body is stretch goal.  
**Status:** 🔜

---

### CPC5.4 — World Model in the Loop
Insert lookahead at decision points (predict grasp failure → re-approach; predict collision → reroute).  
**DoD:** Documented case where lookahead changes the outcome; ablation with/without.  
**Status:** 🔜

---

### CPC5.5 — Benchmark + Photoreal Eval
Evaluate across Isaac Lab-Arena loco-manipulation scenarios; render Cosmos-Transfer photoreal eval clips.  
**DoD:** Arena scorecard table; ≥1 photoreal eval clip; success rate on multi-step task reported.  
**Status:** 🔜

---

### CPC5.6 — Flagship Cut
Polished demo video, project landing page, capstone writeup.  
**DoD:** Landing page published linking all P/T projects; flagship video leads with canonical instruction executed end-to-end.  
**Status:** 🔜

---

## VRAM Requirements

| Operation | Required | Notes |
|---|---|---|
| Unified env + whole-body RL | ≥24 GB | L4 possible for training |
| GR00T inference in loop | ≥16 GB | Already validated in T1 |
| Full C5 eval pipeline | **L40S 48 GB** | Vision env + GR00T + nav + WM simultaneously |

---

## Training Estimates

| Component | GPU-hr | Notes |
|---|---|---|
| CPC5.1 unified env smoke | ~2–5 | Config changes, no long training |
| CPC5.3 staged hand-off | ~20–50 | Fine-tuning nav policy for approach + stop |
| CPC5.4 WM integration | ~5–10 | Inference loop integration, no retraining |
| CPC5.5 eval + video render | ~10–20 | Many eval episodes + photoreal gen |
| **Total** | **~40–90** | Coding ~4–6 weeks; training ~20–50 GPU-hr |

---

## Graceful Degradation (read before starting)

Degrade in this order — each level is still a strong portfolio result:

1. **Full (stretch):** simultaneous whole-body walk-and-manipulate
2. **Staged (target):** navigate → stop → manipulate with controller hand-off (CPC5.3)
3. **Stitched fallback:** separate P3 nav demo + T1 manip demo unified by one shared language/reasoning front-end; document whole-body control as future work

**Ship level 2. Attempt level 1 only if time/compute remain.**

---

## Prerequisites (all must be done before starting C5)

- [x] P0 — CommandNav ✅
- [x] P3 — VisionNav ✅ (nav policy for C5)
- [x] T1 — GR00T N1.7 LoRA ✅ (manipulation policy for C5)
- [x] P2 — World Model ✅ (lookahead for C5)
- [ ] P4 — Cosmos Predict (action-conditioned WM for C5; degrade to P2 WM if P4 inference-only)

---

## Deliverables

- `programs/c5_capstone/` — all integration scripts
- `programs/c5_capstone/skill_router.py` — tags subgoals {NAV|MANIP}, dispatches to policy
- `programs/c5_capstone/unified_env.py` — Isaac Lab env with arms + locomotion
- `programs/c5_capstone/run_c5.py` — end-to-end demo runner
- Arena scorecard + photoreal eval clips
- **Flagship demo video** (canonical instruction end-to-end)
- Project landing page linking all P/T projects
- `docs/results/c5_capstone.md`
- Portfolio pitch (one paragraph for interviews)

---

## Interview Pitch (draft)

> "I built the physical-AI stack bottom-up across two embodiments — honest RL task design, language grounding, world-model internals, world-foundation-models at scale — and converged them into a single language-driven loco-manipulation humanoid with Cosmos in the loop."

---

## Related

- [[P3 - VisionNav]] — nav policy (the locomotion half of C5)
- [[T1 - GR00T LoRA]] — manipulation policy (the arm half of C5)
- [[P2 - World Model]] — WM lookahead (fallback for P4 WM)
- [[P4 - Cosmos Predict]] — action-conditioned WM (preferred lookahead if available)
- [Master Roadmap §3](../../MASTER_ROADMAP_CONVERGED.md)
