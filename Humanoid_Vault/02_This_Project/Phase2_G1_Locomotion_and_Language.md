---
tags: [project, phase, locomotion, language]
---

# Phase 2 — G1 Locomotion + Language Conditioning

## 2a. Baseline locomotion
- **Task:** `Isaac-Velocity-Flat-G1-v0` (Isaac Lab stock).
- **Result:** trained ~300 iterations, **~14,000 steps/sec** on L40S. Confirms the G1 + [[RSL_RL_Runner|RSL-RL PPO]] stack works.

## 2b. Language conditioning — **read this carefully**
The policy observation is extended with a **language command embedding** term:

```python
self.observations.policy.language_command = ObsTerm(func=language_command_embedding)
```

But the embedding is **NOT** a learned text encoder. Source: `language_commands.py`:

- `LANGUAGE_EMBEDDING_DIM = 16`
- `embedding_for_text(text)` = **SHA256 hash** of the string → bytes mapped to `[-1, 1]` → **L2-normalized 16-dim vector**. Fully deterministic, no network, CPU-safe.
- 4 fixed commands: `pick up the red cube`, `pick up the blue cube`, `walk to the cube`, `stand still`.
- During the "smoke" run the command is **fixed** for all envs (`language_command_text`, default `"walk forward"`) — so every env gets the *same* constant vector.

### What this means
This is **observation plumbing**, deliberately a placeholder. The note in the code says it plainly: *"A frozen text encoder can replace `embedding_for_text` later without changing the env interface."* So:
- ✅ The interface (16-dim command → policy obs) is built and validated.
- ❌ The policy is **not** yet learning language-conditioned behavior — a constant vector carries no information for the policy to condition on.

**This is the #1 thing to be honest about.** It's good engineering (clean upgrade path) but it is *not* a working language model. To make it real: swap in a frozen CLIP/SentenceTransformer encoder **and** vary the command per episode so it actually correlates with reward. See [[Open_Questions_and_Next_Steps]].

## 2c. Markers (the "Nav" contribution)
`LanguageConditionedG1CustomTaskCfg` adds two visual sphere markers — **Red** at `(2, 1, 0)`, **Blue** at `(2, -1, 0)`, radius 0.2 — into every env. Task id `Humanoid-G1-Custom-MarkerNav-v0`. Trained at **8,192 envs**, **114,000 steps/sec**, mean reward **28.9**.

Related: [[Architecture_Task_Hierarchy]] · [[Vision_Language_Action_Models]] · [[PPO_for_Locomotion]]
