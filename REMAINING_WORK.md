# REMAINING WORK — Executor Backlog (Humanoid project)

> **For the downstream agent:** execute tasks top-to-bottom within each section. Every task lists
> exact **file paths**, **what to do**, and **acceptance criteria**. Do not skip acceptance criteria.
> **Legend:** 🟢 = can do now (no GPU) · 🔴 = needs a GPU (Isaac Lab) · ⭐ = high priority.
> Mark each `[ ]` → `[x]` as you finish. Repo root = `/home/laptop/AUtonomous/Humanoid/`.

---

## SECTION 0 — Wakeboarding Stage I: DONE ✅ (2026-06-21)

All items below are complete:

- [x] Modal training pipeline fully working on L40S
- [x] rope.reset() double-indexing bug fixed (root crash cause)
- [x] RewardManager timing bug fixed (apply_reward_weights BEFORE env construction)
- [x] CUDA_LAUNCH_BLOCKING=1 debug flag removed (was causing 7x slowdown)
- [x] Stage I trained to 5000 iters — `model_latest.pt` saved in Modal volume `wakeboard-ckpts`
- [x] Best checkpoint: `model_980.pt` (0% fell), `model_5700.pt` (5% fell, 81% timeout)
- [x] play.py fixed (omni.replicator API: get_annotator + orchestrator.step)

**Checkpoint location:** `modal volume get wakeboard-ckpts /wakeboard_stage1/model_latest.pt`

---

## SECTION 1 — Wakeboarding Video (🔴⭐ — needs Lightning AI)

Modal gVisor sandbox blocks Nvidia kernel → `omni.replicator` RTX fails → no video on Modal.
Must run on **Lightning AI** or local machine with A2000.

- [ ] 🔴⭐ SSH into Lightning AI Isaac container
- [ ] 🔴 Download checkpoint: `modal volume get wakeboard-ckpts /wakeboard_stage1/model_latest.pt ./model_latest.pt`
- [ ] 🔴 Run: `python play.py --checkpoint ./model_latest.pt --v_pull_kmh 10 --episodes 3 --steps 400 --out rollout.mp4`
- [ ] 🔴 Copy `rollout.mp4` back locally
- [ ] 🟢 Add video to `Humanoid_Docs_Site/media/wakeboard_rollout.mp4` + embed in docs ch.8

**Isaac container setup (inside Lightning):**
```bash
ln -sf /isaac-sim/kit/python/bin/python3 /usr/local/bin/python && \
export ISAAC_PATH=/isaac-sim EXP_PATH=/isaac-sim/apps \
CARB_APP_PATH=/isaac-sim/kit LD_PRELOAD=/isaac-sim/kit/libcarb.so \
RESOURCE_NAME=IsaacSim && source /isaac-sim/setup_python_env.sh
```

---

## SECTION 2 — Wakeboarding Stage II (🔴⭐)

Stage I trained at fixed 10 km/h. Stage II ramps speed via curriculum: 10 → 20 → 30 km/h.
G1 must learn to dynamically balance at real wakeboard speeds.

- [ ] 🟢 Write `configs/stage2.yaml` — enable curriculum, promote_success_rate=0.6, v_pull_levels_kmh=[10,20,30]
- [ ] 🔴 Launch: `modal run modal_app.py --action train --config configs/stage2.yaml --resume /ckpts/wakeboard_stage1/model_latest.pt`
- [ ] 🔴 Monitor until success ≥70% at 30 km/h
- [ ] 🔴 Eval: `modal run modal_app.py --action eval --checkpoint /ckpts/wakeboard_stage2/model_latest.pt`
- [ ] 🔴 Speed sweep: eval at 10/15/20/25/30 km/h, build success-vs-speed table
- [ ] 🟢 Update `FINAL_RESULTS.md` + docs ch.6/8 with Stage II numbers

**Acceptance:** checkpoint + eval JSON with success ≥70% at 30 km/h.

---

## SECTION 3 — HTML Docs Site (🟢, ⭐)

Path: `Humanoid_Docs_Site/`. Chapters need deeper content.

- [x] 🟢 Add chapter 13 references page
- [x] 🟢 Add code-peek blocks to chapters 4, 5, 8
- [x] 🟢 Add cross-links in interview answer boxes
- [x] 🟢 All chapters: add ≥1 diagram OR table OR code-peek
- [ ] 🟢 **Chapter 08 (Wakeboarding)** — update with Stage I results table, actual training curve numbers, rope force equation, curriculum ladder
- [ ] 🟢 **Chapter 06 (Results)** — update with Stage I results once video is available
- [ ] 🟢 Add `wakeboard_rollout.mp4` to `media/` and embed in ch.8 (after video is made)

---

## SECTION 4 — Language + Vision (🔴)

- [ ] 🔴 Language-ON full training run → `results/eval_language.json` + `behavior_separation.png`
- [ ] 🔴 Language-OFF ablation → compare `separation_score`
- [ ] 🔴 Vision-VLA long run → saved checkpoint + log
- [ ] 🟢 Update docs ch.6 & ch.7 with real numbers once runs complete

---

## SECTION 5 — Git hygiene (🟢)

- [ ] 🟢 Merge `gpu-l4-bringup` → `main` (user decision)
- [ ] 🟢 Delete stale `.pt` files from local repo (model_1450.pt, model_5000.pt, model_5700.pt — already in Modal volume, no need in git)

---

## Current priority order

1. **Section 1** — Get the wakeboard video (biggest visible demo payoff) ⭐
2. **Section 2** — Stage II training (the hard riding behavior)
3. **Section 3** — Docs update after video lands
4. **Section 4** — Language/Vision when GPU available
