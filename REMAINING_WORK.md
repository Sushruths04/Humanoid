# REMAINING WORK — Executor Backlog (Humanoid project)

> **For the downstream agent:** execute tasks top-to-bottom within each section. Every task lists
> exact **file paths**, **what to do**, and **acceptance criteria**. Do not skip acceptance criteria.
> **Legend:** 🟢 = can do now (no GPU) · 🔴 = needs a GPU (Isaac Lab) · ⭐ = high priority.
> Mark each `[ ]` → `[x]` as you finish. Repo root = `/home/laptop/AUtonomous/Humanoid/`.

---

## SECTION 1 — HTML docs site: make it DETAILED (🟢, ⭐)
Path: `Humanoid_Docs_Site/`. The chapters currently explain *what* but are thin on *how/why* depth.
Goal: each chapter goes from "overview" to "deep-dive" while staying readable. **Do not break** the
shared structure: keep `class="rv"`, the sidebar (`HSite.init`), and the pager script at the bottom of
each page. Add content as new `<section>`/`<h2>` blocks, SVG `.diagram`s, tables, and `<pre>` code peeks.

### Global additions
- [ ] 🟢 Add a new page `chapters/13_references.html` (papers + links: HumanUP, AMP, Isaac Lab, GR00T, π0, OpenVLA, DreamerV3) and register it in `assets/site.js` `CHAPTERS` array + add a sidebar entry.
- [ ] 🟢 Add a **"📂 Code peek"** block to chapters 4, 5, 8 showing the *real* snippet from the repo (copy the actual function, e.g. `wakeboard_rewards.board_positive_angle`, `rope_model.compute_force`). Wrap in `<pre>`.
- [ ] 🟢 Add **cross-links**: every "Interview answer" box should link to the relevant deeper chapter.

### Per-chapter deepening (what to ADD to each)
- [ ] 🟢 **01 Big Picture** — add a "What RL actually optimizes" sub-section (return = Σ discounted reward) with a tiny SVG of the trajectory→return idea.
- [ ] 🟢 **02 The Stack** — add a "data/■control flow" sequence diagram (obs tensor shapes, action dim ~23, PD gains) + a version table (Isaac Sim 5.1, warp-lang 1.4.2, torch cu126).
- [ ] 🟢 **03 Sit→Stand** — add the **two-stage reward tables** (Stage I discovery terms vs Stage II tracking terms) and a phase-decomposition diagram; explain "8× slowdown" concretely.
- [ ] 🟢⭐ **04 Rewards** — this is the centerpiece, go deepest: (a) show the **exact math** of 2–3 terms (gaussian tracking `exp(-err²/σ²)`, board-angle band, phase-gating); (b) a **full weights table** pulled from `wakeboarding-experiment/configs/stage1.yaml`; (c) a worked "what happens if weight X is too high" paragraph each; (d) a reward-hacking case study (the PRM +/- marker bug).
- [ ] 🟢 **05 Simulation** — add: observation vector breakdown (what each entry is + dim), action space (PD targets), episode/termination conditions, and a domain-randomization ranges table copied from `configs/stage2.yaml`.
- [ ] 🟢 **06 Results** — add the **eval JSON schema** (`eval.py` output fields), the comparison-table templates (speed sweep / ablations / Stage I vs II), and how `separation_score` is computed for language.
- [ ] 🟢 **07 Language & Vision** — add the CNN architecture table (channels 32/64/64, kernels 8/4/3, strides 4/2/1), the hash-embedding explanation with the actual 16-dim formula, and the frozen-encoder upgrade path.
- [ ] 🟢⭐ **08 Wakeboarding** — add: the rope **spring-force equation** `F = kp·(x_anchor−x_handle) + kd·(v_anchor−v_handle)` capped at `f_max`, the curriculum ladder table (10→30 km/h), the full reward table, and the checkpoint ladder (ckpt_00→ckpt_40) from `PLAN.md §10`.
- [ ] 🟢 **09 Model Choices** — add a comparison matrix (rows: GR00T/π0/OpenVLA/Dreamer/PPO; cols: embodiment, data, sample-eff, locomotion-fit) and a 2–3 line "steelman" of each rejected option (be fair, then say why not here).
- [ ] 🟢 **10 Compute** — add a concrete cost worked-example (e.g. Stage II ≈ 15 h × L40S rate) and the exact `modal run` / `docker_image_portability.sh` commands.
- [ ] 🟢 **11 Roadmap** — turn the ladder into a table with owner/effort/blocker columns.
- [ ] 🟢 **12 Glossary** — add 10–15 more terms (GAE, advantage, KL, on-policy, actor-critic, PD control, URDF/USD, contact sensor, articulation, manager-based env, TiledCamera, retargeting).

**Acceptance:** every chapter has ≥1 new diagram OR table OR code-peek; `node -c assets/site.js` passes; `<div>` open/close counts balance on every page; all sidebar links resolve.

---

## SECTION 2 — Wakeboarding GPU bring-up (🔴, ⭐)
Path: `wakeboarding-experiment/`. Code is written + CPU-verified; these are the runtime-only `# VERIFY`
markers. Do these on an **interactive Lightning box** (see `PRE_GPU_CHECKLIST.md`).

- [ ] 🔴 **Smoke first:** `bash scripts/00_smoke.sh`. Iterate until `runs/wakeboard_smoke/model_*.pt` is written.
- [ ] 🔴 `src/tasks/wakeboard_start_cfg.py::_apply_handle_force` — wire the real external-force API: confirm `robot.set_external_force_and_torque(forces, torques, body_ids=self._hand_body_ids)` signature for the installed Isaac Lab; print once to confirm force lands on the hands.
- [ ] 🔴 `src/board.py` + env — implement the **foot→board fixed-joint weld** (currently a wiring point). Tune PhysX solver iters if the board jitters.
- [ ] 🔴 Replace the reset event with a **cannonball crouch** init pose (deep hip/knee flex, torso reclined, board pitched up ~15°). Currently uses `reset_scene_to_default`.
- [ ] 🔴 Confirm the `loco_mdp.*` observation term names exist in the installed version (`joint_pos_rel`, `projected_gravity`, `last_action`…); adjust imports if renamed.
- [ ] 🔴 Confirm `G1_HAND_LINKS` gripper link name (`*_rubber_hand` vs `*_wrist_yaw_link`) and `G1_TORSO_LINK` (`torso_link`); verify `_quat_pitch` quaternion order (wxyz).
- [ ] 🟢 `src/amp/reference_motion.py::build_keyframe_reference` — author the 3 real G1 keyframes (crouch→mid→tall) using the known joint layout, instead of zeros.
- [ ] 🔴 Train: `bash scripts/10_train_stage1.sh` → reach `ckpt_20_stage1_30` (≥50% success @30 km/h).
- [ ] 🔴 Train Stage II: `STAGE1_CKPT=… bash scripts/20_train_stage2.sh` → `ckpt_30_stage2_deploy` (≥70%).
- [ ] 🔴 Eval + sweep: `CKPT=… bash scripts/31_eval_speed_sweep.sh`; then `bash scripts/99_collect_results.sh`.
- [ ] 🔴 Video (Lightning only): `CKPT=… bash scripts/40_record_video.sh` → `media/wakeboard_rollout.mp4` for the docs.
- [ ] 🟢 After each run: update `vault/03_Results/Results_Live.md` + `vault/04_Log/Experiment_Log.md` + `vault/00_INDEX.md` (per `DOC_PROTOCOL.md`).

**Acceptance:** a saved Stage-II checkpoint, an eval JSON with success ≥70% @30 km/h, a results table, and a rollout mp4.

---

## SECTION 3 — Language-conditioning "ON" completion (🔴 + 🟢)
Path: `my-humanoid-project/` (+ plan `PLANS/LANGUAGE_ON_PLAN.md`). Code + eval written & CPU-verified.

- [ ] 🔴 Smoke: `bash scripts/run_language_velocity.sh` (16 envs). Verify the `_reset_idx` per-episode resample hook works on the installed version; fix shapes/devices.
- [ ] 🔴 Full train: `FULL=1 bash scripts/run_language_velocity.sh` → save checkpoint + log.
- [ ] 🔴 Proof: produces `results/eval_language.json` + `results/behavior_separation.png` with `language_is_on: true`.
- [ ] 🔴 Ablation: train a **language-OFF** variant (constant embedding) and compare `separation_score` (off ≪ on).
- [ ] 🟢 Update `Humanoid_Vault/02_This_Project/Phase2_G1_Locomotion_and_Language.md` status "placeholder → ON" with the real numbers + the plot.
- [ ] 🟢 Add the behavior-separation plot to docs chapter `07_language_vision.html` (drop in `media/` and `<img>` it).

**Acceptance:** a behavior-separation plot showing distinct per-command behavior + the on/off ablation.

---

## SECTION 4 — Locomotion & vision gaps (🔴)
- [ ] 🔴 **Vision long run:** the camera pipeline only smoke-tested. Run a real Vision-VLA training and **save a checkpoint + log** (currently none in repo). Update `FINAL_RESULTS.md` Phase 3.
- [ ] 🔴 **Reproducibility:** re-run MarkerNav + baseline locomotion and **save checkpoints/logs** so their numbers are backed by files (today only `g1_robust` is saved).
- [ ] 🟢 Once vision/marker results exist, refresh docs chapters 06 & 07 and `media/`.

**Acceptance:** vision + marker results each have an on-disk checkpoint + log (not just prose).

---

## SECTION 5 — Compute / Docker verification (🟢 + 🔴)
- [ ] 🔴 Inside the container: `python -c "import rsl_rl"`. If missing, build `wakeboarding-experiment/docker/Dockerfile` (it layers rsl-rl) and `docker_image_portability.sh push` the result.
- [ ] 🟢 `wakeboarding-experiment/modal_app.py` — verify the `cwd="/workspace/wakeboarding-experiment"` path matches how the image mounts the repo; adjust if needed.
- [ ] 🔴 Smoke a Modal run end-to-end once (`modal run modal_app.py --action train --config configs/smoke.yaml`).

**Acceptance:** one green Modal smoke run + confirmed deps.

---

## SECTION 6 — Media / assets (🟢 after runs)
Path: `Humanoid_Docs_Site/media/`.
- [ ] export `robust_rollout.mp4` (existing G1 robust policy) → appears in docs ch.6 automatically.
- [ ] add `wakeboard_rollout.mp4` (from §2) → docs ch.8.
- [ ] add `behavior_separation.png` (from §3) → docs ch.7.
- [ ] optional `poster.jpg` still frame.

---

## Suggested order for the small agent
1. **Section 1** (docs deepening) — all 🟢, do now, biggest visible payoff. ⭐
2. **Section 5** 🟢 items + **Section 2** keyframes (🟢).
3. When GPU is up: **Section 2** smoke→train, then **Section 3**, then **Section 4**.
4. **Section 6** media as results land, refreshing docs per `DOC_PROTOCOL.md`.
