---
tags: [study, roadmap, next-steps]
---

# Open Questions & Next Steps (the deep dive)

This note reconciles **what you *intend* to do** (`FUTURE_TASKS_HUMANOID_ROBOT.md`, `FUTURE_WORK.md`) with **what the code actually is right now** ([[Results_Summary]]). Read this before picking the next task — because the future docs assume a few things are "done" that are really just **plumbing**.

---

## ⚠️ The reconciliation (intended vs. real)

| The roadmap assumes… | Reality in the code | Consequence |
|---|---|---|
| "Marker navigation working" | MarkerNav *trained* (reward 28.9), but **no saved checkpoint/log** in repo | Re-run + save before building on it |
| "Language conditioning working" | Command is a **fixed constant** 16-dim hash vector; **not randomized per episode** ([[Phase2_G1_Locomotion_and_Language]]) | The policy cannot *yet* condition on language — this is a **prerequisite**, not done |
| "Vision pipeline" | Runs to PPO at smoke scale; **no trained vision policy** ([[Phase3_Vision_VLA]]) | Vision is *unblocked*, not *trained* |
| Reward shaping for nav/sequence | Tasks reuse **stock locomotion rewards**; no goal-progress / wrong-target / sequence rewards exist yet | Every advanced task needs **new reward terms** first |

**Take-away:** the single highest-leverage next move is **not** Phase 9 vision — it's to make the **language signal actually informative**. Everything in the roadmap ("go to red", sequences, styles) depends on the policy being able to *read a varying command*. Right now it can't.

---

## 🎯 Your stated end-goal
From `FUTURE_TASKS_HUMANOID_ROBOT.md`, the chosen thesis direction is:

> **Language-conditioned G1 humanoid navigation with sequential goals, obstacle avoidance, and robustness testing.**
> e.g. *"Go to the red marker, avoid obstacles, then go to the blue marker and stop."*

That's a great target. It needs five capabilities stacked: **commanded locomotion → goal grounding → sequencing → obstacles → robustness**. You already have **robustness** ([[Phase2.5_Sim2Real_Robust]]); the rest is unbuilt.

---

## 🪜 Concrete next-steps ladder (re-ordered for *real* state)

### Step 0 — Lock the baseline (prereq, ~½ day)
- Re-run MarkerNav + baseline; **save checkpoints + logs** so results are reproducible from this repo (currently only `g1_robust` is saved). Fill the TBD table in `FUTURE_TASKS_HUMANOID_ROBOT.md`.

### Step 1 — **Make language real** (the unlock) 
This is the true "Phase 1" given the code:
1. In `language_command_embedding`, **randomize the command per episode** (sample from the 4 `COMMANDS`, or a velocity-command set) instead of the constant `"walk forward"`.
2. Add a **reward term that depends on the command** (e.g. velocity-tracking for "walk fast/slow/stop", or heading for "turn left/right"). Without command-dependent reward, the embedding stays decorative.
3. (Optional but strong) swap the SHA256 hash for a **frozen sentence encoder** — the code is explicitly designed for this drop-in (`embedding_for_text` → CLIP/SentenceTransformer). Keep it **frozen** (no grad) for stability.
4. **Metric:** per-command behavior separation — does "stop" actually stop, does "fast" raise speed? Plot velocity vs. commanded velocity.

### Step 2 — Multi-goal grounding ("go to red/blue/green")
- Spawn 3–5 colored markers at **random** positions; command = which color.
- **New rewards:** + progress toward correct marker, − progress toward wrong marker.
- **Metrics:** correct-target reach rate, wrong-target rate, time-to-target.

### Step 3 — Sequential goals ("red then blue")
- Add a **subgoal index** to the observation; advance it when a marker is reached.
- Start 2-step, then 3-step. **Metrics:** full-sequence completion, ordering correctness.

### Step 4 — Obstacles
- Static boxes/walls → curriculum to randomized layouts. **Rewards:** − collision, + path progress. **Metrics:** reach rate, collision rate, path efficiency.

### Step 5 — Push recovery (robustness++)
- Random torso forces, escalating magnitude. **Metrics:** recovery rate, fall rate under disturbance. (Pairs naturally with your existing DR work.)

### Step 6 — Vision (only now)
- Replace **privileged marker positions** with the **camera** ([[Phase3_Vision_VLA]]). Strongly consider **teacher–student / privileged distillation**: train a state-based teacher (Steps 2–4), then distill into a vision student. This is the standard way to make pixel policies trainable and is the most research-credible framing.

---

## 🔬 Alternative track (from `FUTURE_WORK.md`): Loco-Manipulation
A higher-ceiling but harder direction: **unfreeze the G1 arms** and do "pick up the object at the red marker" — coordinating walking + reaching in one policy. Research question worth stating: *can one language embedding gate both locomotion and manipulation sub-policies?* Defer until Steps 1–3 prove the language signal works.

---

## ❓ Open research questions to put in the thesis
1. Does a **frozen** text encoder beat a **learned-from-scratch** command embedding for this many discrete commands? (Probably overkill for 4 commands — worth measuring.)
2. How much does **vision** cost vs. privileged state (the sim-to-real gap)? Quantify the performance drop.
3. Does **domain randomization** trained for terrain also help **push recovery** zero-shot?
4. What's the **teacher→student** distillation gap for vision navigation?

---

## ✅ Definition of "thesis-done" (proposed)
- Steps 1–4 trained **with saved checkpoints + logs + rollout videos** in-repo (not just `FINAL_RESULTS.md` prose).
- A results table with **success rate / fall rate / episode length** per task (the table in `FUTURE_TASKS_HUMANOID_ROBOT.md`, filled in).
- One honest **ablation**: language-on vs language-off; vision vs state.

Related: [[Project_Overview]] · [[Phase2_G1_Locomotion_and_Language]] · [[Results_Summary]] · [[Domain_Randomization_and_Sim2Real]]
