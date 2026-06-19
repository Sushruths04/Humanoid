# Plan — Turn Language Conditioning "ON" (G1)

> **Goal:** make the G1 actually **respond to different commands**, instead of receiving a constant dummy vector. Executor-ready for a Sonnet-class model. Small, high-leverage, the prerequisite for every advanced nav task.

## Why (the problem today)
In `my-humanoid-project/my_humanoid_project/language_commands.py` + `tasks/g1_language_pickplace_cfg.py`:
- The command embedding is a **deterministic SHA256 hash** of a string → 16-dim vector (fine).
- **But** `language_command_embedding()` uses a **fixed** `env.language_command_text` (default `"walk forward"`) — **the same constant vector for every env, every episode**.
- And **no reward depends on the command**. So the policy has zero incentive (and zero ability) to condition on language. It's wiring, switched off.

## The fix (three changes)
### Change 1 — Randomize the command per episode
- Maintain a per-env command id `cmd_id[num_envs]`, **resampled on reset** from a command set.
- Start with a **velocity-command** set (cleanest reward signal), e.g.:
  `{0: walk forward, 1: walk backward, 2: turn left, 3: turn right, 4: stand still, 5: walk slow, 6: walk fast}`.
- The observation embedding = `embedding_for_text(COMMANDS[cmd_id].text)` per env (vectorized). Keep the 16-dim hash encoder for v1.

### Change 2 — Make reward depend on the command (the actual "on" switch)
Add a **command-tracking reward term** that maps each command to a target base velocity / yaw rate, and rewards matching it:
- forward → target vx = +1.0 m/s; backward → −1.0; slow → +0.5; fast → +1.5; stand still → 0; turn left/right → target yaw rate ±0.5 rad/s.
- `reward_cmd_track = exp(-‖(vx, vy, yaw_rate)_actual − target(cmd)‖² / σ²)`, weight ~1.5.
- Keep the stock locomotion stability/effort terms. This is the standard Isaac Lab **velocity-command** formulation — we're just *driving the command from the language embedding* instead of a raw vector.

### Change 3 — (optional, v2) swap the hash for a frozen text encoder
- Replace `embedding_for_text` with a **frozen** sentence/CLIP encoder (no grad), projected to the policy's expected dim. The code was explicitly designed for this drop-in. Keep frozen for stability. Only worth it once >a handful of commands or natural-language phrasing is needed.

## Files to touch
- `my_humanoid_project/language_commands.py` — extend `COMMANDS` to the velocity set; add `target_velocity(cmd_id)` helper.
- `tasks/g1_language_pickplace_cfg.py` — per-env `cmd_id` buffer + resample on reset; vectorized embedding obs; register the `reward_cmd_track` term.
- `thesis/scripts/11_g1_language_cond.sh` — train this; **save checkpoint + log** (currently nothing is saved for the language task).
- new `eval` — per-command behavior eval (below).

## Training
- Reuse G1 flat locomotion base + RSL-RL PPO (same HPs as the locomotion runner). 4096 envs, ~1.5–3k iters on L40S. Smoke-test first (16 envs/2 iters).

## Acceptance criteria / metrics (this is how you prove "language is on")
- **Per-command velocity-tracking error** (actual vs target) low for each command.
- **Behavior separation plot:** "stand still" → ~0 speed; "fast" > "slow" > "stand"; "turn left/right" → opposite yaw — clearly distinct.
- **Command-switch test:** change the command mid-episode → behavior changes accordingly.
- **Saved:** checkpoint + log + an `eval_language.json` + a behavior-separation figure. (None of these exist today.)

## Ablation (for the portfolio)
Train **language-off** (constant vector) vs **language-on** (this plan); show the off version can't follow distinct commands while on can. This single ablation is the credible "I made the VLA's L real" story.

## Executor task list
1. Extend `language_commands.py` with the velocity-command set + `target_velocity`.
2. Add per-env `cmd_id` buffer + reset-time resampling in the task cfg; vectorize the embedding obs.
3. Add `reward_cmd_track` term + register it; expose weight in config.
4. Smoke test (16 envs, 2 iters); fix shapes/devices.
5. Full train (4096 envs) via `11_g1_language_cond.sh`; **save checkpoint + log**.
6. Write per-command eval → `eval_language.json` + behavior-separation figure.
7. Run the language-off vs language-on ablation.
8. Document: update the main `Humanoid_Vault/02_This_Project/Phase2_G1_Locomotion_and_Language.md` status from "placeholder" → "on", with the real numbers.

## Where this sits in the bigger roadmap
This is **Step 1** of `Humanoid_Vault/03_Study/Open_Questions_and_Next_Steps.md`. After it works: multi-goal marker grounding → sequential goals → obstacles → push recovery → vision (teacher→student). The wakeboarding project (`wakeboarding-experiment/`) is a **parallel, independent** track.
