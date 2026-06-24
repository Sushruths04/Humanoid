# Wakeboarding RL — Executor Handoff (GPU tasks)

> **For the executor agent.** The GPU bring-up is DONE: the smoke test is green on an L4 and
> the whole pipeline trains. Your job is the long-running training/eval grind below. Work
> top-to-bottom. **Stop and report back to the orchestrator (the user will relay to the lead
> agent) at every ⏸️ checkpoint, or on any error you can't fix in 2 attempts.** Don't burn GPU
> hours guessing — escalate.

---

## 0. Connection & environment

**SSH to the L4 Lightning Studio** (key already configured in `~/.ssh/lightning_rsa` + `~/.ssh/config` on this machine):
```bash
ssh s_01kvjpvyze89dkgnykht09z69w@ssh.lightning.ai
```
If the key is rejected (different machine / new studio), re-register by running the user's
`curl -s "https://lightning.ai/setup/ssh?t=...&s=<studioid>" | bash` link, then retry.

**Key facts about the box:**
- Repo: `~/Humanoid` (= `/teamspace/studios/this_studio/Humanoid`), on branch **`gpu-l4-bringup`**.
- GPU: **NVIDIA L4, 24 GB** (RT cores OK). NEVER use A100/H100 — Isaac Sim won't render.
- Docker image cached: `ghcr.io/sushruths04/humanoid-isaaclab:latest` (17.6 GB). **Isaac Sim 5.1, rsl-rl ≥ 5.0.**
- All runs go through `wakeboarding-experiment/docker/run.sh` (wraps `docker compose`). The compose
  file already fixes the entrypoint, EULA, Isaac env vars, and unbuffered logs — **do not change it.**
- Work dir for every command: `cd ~/Humanoid/wakeboarding-experiment`.

**How to launch a long run (detached, survives SSH drop) and watch it:**
```bash
cd ~/Humanoid/wakeboarding-experiment
rm -f ~/_setup_logs/stage1.log
setsid bash docker/run.sh train stage1 > ~/_setup_logs/stage1.log 2>&1 < /dev/null & disown
# monitor:
tail -f ~/_setup_logs/stage1.log
# checkpoints appear under: runs/wakeboard_stage1/model_*.pt
```

**Gotchas (learned the hard way — don't be fooled):**
- Container start takes ~1–2 min; Isaac Sim init ~40 s; **first run downloads G1 USD assets (~5 min, silent, needs network).**
- `train.py` stdout is unbuffered now, so PPO iterations stream live. Isaac logs are verbose — grep for `Learning iteration`, `Mean reward`, `Traceback`.
- **100% GPU with no new log ≠ progress.** Confirm real progress by the `.pt` files appearing, not GPU%.
- L4-safe `num_envs` defaults are 2048 (in the scripts). Override per-run with `NUM_ENVS=4096 ...` only if VRAM allows.

---

## 1. ⚠️ VERIFY the physics is real (do FIRST, ~15 min) ⏸️
The smoke proves the pipeline RUNS, but three physics pieces are still placeholder/unconfirmed.
In the smoke log, `forward_glide≈0`, `board_positive_angle≈0`, `pen_pull_against_rope=0` — suggests
the **rope may not actually be pulling and the board may not be welded.** Confirm before a 5 h run.

- [ ] Run a 30-iteration probe: `NUM_ENVS=64 MAX_ITERS=30 setsid bash docker/run.sh ... ` (or edit `configs/smoke.yaml` max_iterations=30) and watch whether `forward_glide` and `Episode_Reward/pen_pull_against_rope` become non-zero. If they stay zero, the rope force isn't landing.
- [ ] If rope force is dead: `src/tasks/wakeboard_start_cfg.py::_apply_handle_force` — the `set_external_force_and_torque` call is wrapped in `try/except pass`. Remove the bare except, print the exception, and fix the API signature for Isaac Lab 5.1 (`body_ids` arg name, force tensor shape `(N, num_bodies, 3)`). The hand body ids are `*_palm_link` (already resolved).
- [ ] `src/board.py` — confirm the foot→board weld actually attaches (board shouldn't fall away). If the board free-falls, the fixed-joint isn't being created.
- [ ] Reset pose still uses `reset_scene_to_default` (not the cannonball crouch). Acceptable for first training, but note it.

**⏸️ REPORT to orchestrator:** "rope force lands: yes/no; board welded: yes/no" + the probe reward table. The lead agent decides whether to fix physics first or train as-is.

---

## 2. Stage I training → `ckpt_20_stage1_30` (🔴 ~4–5 h L4) ⏸️
```bash
cd ~/Humanoid/wakeboarding-experiment
rm -f ~/_setup_logs/stage1.log
setsid bash docker/run.sh train stage1 > ~/_setup_logs/stage1.log 2>&1 < /dev/null & disown
tail -f ~/_setup_logs/stage1.log
```
- Watch: `Mean reward` should climb; `Episode_Termination/fell` should fall; the curriculum should
  advance (`[curriculum] advanced -> 15 km/h`, then 20, 25, 30) as success ≥ 60%.
- Checkpoints save to `runs/wakeboard_stage1/model_<iter>.pt` every `save_interval` (100).
- **Target:** `ckpt_20_stage1_30` ≡ a checkpoint that reaches ≥50% stand success at 30 km/h.

**⏸️ REPORT to orchestrator** with: final mean reward, fell%, whether curriculum reached 30 km/h, and the latest model path. The lead agent judges if Stage I is good enough to proceed.

---

## 3. Stage II training → `ckpt_30_stage2_deploy` (🔴 ~5 h) ⏸️
First author the 3 AMP keyframes (CPU, no GPU): `src/amp/reference_motion.py::build_keyframe_reference`
— replace the zero placeholders with crouch→mid→tall G1 poses (joint layout is known).
```bash
STAGE1_CKPT=runs/wakeboard_stage1/model_<best>.pt \
  setsid bash docker/run.sh ... train stage2 > ~/_setup_logs/stage2.log 2>&1 < /dev/null & disown
```
(Note: `docker/run.sh train stage2` doesn't pass `STAGE1_CKPT`/`--resume` — use `scripts/20_train_stage2.sh`
inside the container instead, or extend run.sh. **Flag this to the orchestrator if unsure.**)
- **Target:** `ckpt_30_stage2_deploy` ≡ ≥70% success @30 km/h, smooth/natural motion.

---

## 4. Eval + speed sweep + video (🔴 ~30 min) ⏸️
```bash
# inside container shell:  bash docker/run.sh shell
CKPT=runs/wakeboard_stage2/model_<best>.pt bash scripts/31_eval_speed_sweep.sh
bash scripts/99_collect_results.sh           # -> results table
CKPT=... bash scripts/40_record_video.sh     # -> media/wakeboard_rollout.mp4 (L4 only, needs Vulkan)
```
Copy the mp4/plots into `Humanoid_Docs_Site/media/` (they auto-appear in the docs).

---

## 5. SEPARATE TASK — Language conditioning "ON" (🔴, different project)
**This is NOT wakeboarding.** Project: `~/Humanoid/my-humanoid-project/`. Plan: `PLANS/LANGUAGE_ON_PLAN.md`.
The G1 language is currently a placeholder embedding; this turns it real.
```bash
cd ~/Humanoid/my-humanoid-project
bash scripts/run_language_velocity.sh           # 16-env smoke first
FULL=1 bash scripts/run_language_velocity.sh    # full train
```
- **Targets:** `results/eval_language.json` + `results/behavior_separation.png` with `language_is_on: true`;
  then an on/off ablation (constant-embedding variant) showing `separation_score` off ≪ on.
- ⚠️ Same Isaac/rsl-rl version gotchas as wakeboarding may apply — if the runner cfg or env wrapper
  errors, mirror the fixes already in `wakeboarding-experiment/train.py` (RslRlVecEnvWrapper, the
  rsl-rl 5.0 actor/critic config). **Escalate to orchestrator if it diverges.**

---

## 6. SEPARATE TASK — Vision / MarkerNav reproducibility (🔴)
Run the Vision-VLA and MarkerNav trainings and **save checkpoints + logs** (currently only `g1_robust`
is saved). See `VISION_VLA_CNN_RUNBOOK.md`. Update `FINAL_RESULTS.md` with real numbers.

---

## Escalation protocol (cost control)
**Stop and report to the orchestrator when:**
- you hit any ⏸️ checkpoint above,
- an error you can't resolve in 2 attempts,
- a run would exceed its rough time budget by >1.5×,
- physics looks wrong (rewards stuck at 0, robot behaves nonsensically).

Give the orchestrator: the exact command run, the last 30 log lines, checkpoint paths, and the
reward/termination table. The orchestrator (lead agent) makes the judgment call so we don't waste L4 hours.

## Branch / git
All bring-up fixes are on `gpu-l4-bringup` (pushed). Commit new work there with explicit messages;
do NOT merge to `main` without the user's say-so.
