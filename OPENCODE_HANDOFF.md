# OPENCODE HANDOFF — Humanoid Wakeboarding RL (continue here)

**One-line status:** Pipeline + physics are working on an L4 GPU (rope pulls, board welded, PPO
trains, checkpoints save). The remaining problem is that the **deep-water start isn't learnable
yet** — Stage I shows a **100% fall rate** because the robot starts standing and the rope yanks it
over. Your job: make it learnable, then run the training ladder.

---

## 1. CONNECT (quick)

```bash
ssh s_01kvjpvyze89dkgnykht09z69w@ssh.lightning.ai
```
- SSH key is already at `~/.ssh/lightning_rsa` (+ `~/.ssh/config` entry) on the operator machine.
- If the key is rejected (new machine/studio), re-register by running the user's Lightning link:
  `curl -s "https://lightning.ai/setup/ssh?t=<token>&s=<studioid>" | bash`, then retry. If you
  don't have that link, STOP and ask the user.
- The box is a Lightning Studio. GPU: **NVIDIA L4, 24 GB**. **Never use A100/H100** (Isaac Sim needs RT cores).

## 2. SETUP / ORIENT (everything is already installed — just verify)

Everything is pre-installed and cached on the box. You do NOT need to build anything.
```bash
cd ~/Humanoid                       # = /teamspace/studios/this_studio/Humanoid
git status -sb                      # should be on branch: gpu-l4-bringup
git pull --rebase origin gpu-l4-bringup
docker images | grep humanoid-isaaclab     # 17.6GB image must be present (cached)
```
If on a FRESH box with nothing:
```bash
git clone https://github.com/Sushruths04/Humanoid.git ~/Humanoid
cd ~/Humanoid && git checkout gpu-l4-bringup
docker pull ghcr.io/sushruths04/humanoid-isaaclab:latest   # ~15 min, one time
```

**All work happens in:** `cd ~/Humanoid/wakeboarding-experiment`
**Everything runs through:** `bash docker/run.sh <cmd>` (compose handles GPU + Isaac env, do not edit it).

```bash
bash docker/run.sh shell           # interactive container shell
bash docker/run.sh train smoke     # run configs/smoke.yaml (respects its max_iterations)
bash docker/run.sh train stage1    # Stage I
```
**Launch a long run detached (survives SSH drop) + watch:**
```bash
cd ~/Humanoid/wakeboarding-experiment
rm -f ~/_setup_logs/run.log
setsid bash docker/run.sh train stage1 > ~/_setup_logs/run.log 2>&1 < /dev/null & disown
tail -f ~/_setup_logs/run.log
```

## 3. WHERE TO REFER (file map)

| File | What |
|---|---|
| `wakeboarding-experiment/PHYSICS_BRINGUP_REPORT.md` | **READ FIRST.** Previous agent's full root-cause writeup of the rope + board fixes. |
| `WAKEBOARD_EXECUTOR_HANDOFF.md` | Overall task plan + escalation protocol. |
| `wakeboarding-experiment/PLAN.md` | The master design doc (rewards, curriculum, two-stage, checkpoint ladder). |
| `wakeboarding-experiment/src/tasks/wakeboard_start_cfg.py` | The env: scene, observations, **ActionsCfg**, rewards, terminations, **EventsCfg.reset_pose**, rope/board hooks. |
| `wakeboarding-experiment/src/rope_model.py` | Rope spring model + `reset()` (anchor lead) + `compute_force`. |
| `wakeboarding-experiment/src/board.py` | Board asset + `_bind_feet_to_board` weld. |
| `wakeboarding-experiment/src/rewards/wakeboard_rewards.py` | All 16 reward term functions. |
| `wakeboarding-experiment/configs/{smoke,stage1,stage2}.yaml` | Run configs (num_envs, iters, rope, DR, reward weights). |
| `wakeboarding-experiment/train.py` | Training entry (env build, RslRlVecEnvWrapper, rsl-rl 5.0 cfg). |
| `wakeboarding-experiment/scripts/` | `00_smoke … 99_collect_results`. |

## 4. WHAT'S DONE (don't redo)

- GPU bring-up: image entrypoint, Isaac env vars, EULA, rsl-rl 5.0 API, env actions, buffer ordering, G1 names. (commits up to `3e8e8ba`)
- Rope force lands correctly (`step()` override, 600N on both `*_palm_link`). (`9e1d1c8`)
- Board welded to both feet via USD fixed joints. (`11efc26`)
- Smoke is green; Stage I runs end-to-end and saves checkpoints.

## 5. THE PROBLEM TO SOLVE FIRST — make the start learnable

Stage I plateaus at **fell = 1.0000 (100%)** through 193+ iters. Two root causes, both UNFIXED:

1. **Reset pose** is still `reset_scene_to_default` (default standing). `EventsCfg.reset_pose` in
   `wakeboard_start_cfg.py`. The robot must START in a crouched **cannonball wakeboard-start**
   pose (deep hip/knee flex, torso reclined, arms forward toward the handle). *This is the single
   highest-leverage fix* (per the previous agent's report).
2. **Rope yanks at max 600N from step 1**: `rope_model.reset()` places the anchor `lead=5.0 m`
   ahead → spring wants `k_p*5 = 4000N` → capped to `f_max=600N` instantly. Reduce the initial
   lead (~0.4 m → ~320N at start, builds as the robot lags) and confirm `self.rope.reset(...)` is
   actually called in the env's `_reset_idx` so the anchor re-places each episode.

**Acceptance for "learnable":** in a 100-iter probe, the fall rate at iter 100 is clearly BELOW
iter 1 (policy is staying up longer), even if reward is still negative. Run probes with
`docker/run.sh train smoke` after setting `configs/smoke.yaml max_iterations: 100`.

⚠️ GOTCHA: `docker/run.sh smoke` (without `train`) forces 2 iters via `scripts/00_smoke.sh`. Use
`docker/run.sh train smoke` to respect the YAML.

## 6. THE PLAN / CHECKPOINT LADDER (after it's learnable)

| Checkpoint | Accept criteria | Notes |
|---|---|---|
| ✅ `ckpt_00_smoke` | pipeline + PPO run | DONE |
| ✅ physics real | rope pulls + board welded | DONE |
| ⏳ **learnable start** | fall rate drops over 100 iters | **YOU ARE HERE** (reset pose + rope lead) |
| `ckpt_10_stage1_slow` | ≥60% success @10 km/h | Stage I, curriculum start |
| `ckpt_20_stage1_30` | ≥50% success @30 km/h | Stage I target (`bash docker/run.sh train stage1`, ~4–5h L4) |
| `ckpt_30_stage2_deploy` | ≥70% @30 km/h, smooth | Stage II (needs AMP keyframes in `src/amp/reference_motion.py`; use `scripts/20_train_stage2.sh` with `STAGE1_CKPT=...`) |
| `ckpt_40_robust` | ≥60% under full DR | Stage II + domain randomization |

After Stage II: eval + speed sweep (`scripts/31_eval_speed_sweep.sh`), record video
(`scripts/40_record_video.sh` → `media/`), then collect results (`scripts/99_collect_results.sh`).

**Separate later tasks** (different projects, same box): language-ON at `~/Humanoid/my-humanoid-project/`
(see `PLANS/LANGUAGE_ON_PLAN.md`); vision/MarkerNav reproducibility (see `VISION_VLA_CNN_RUNBOOK.md`).

## 7. GOTCHAS (learned the hard way)

- Container start ~1–2 min; Isaac Sim init ~40s; **first run downloads G1 USD assets ~5 min, silently** (needs network).
- **100% GPU ≠ progress.** Confirm via `.pt` files appearing and the reward table, not GPU%.
- `train.py` output is unbuffered (good); Isaac logs are noisy — grep `Learning iteration|Mean reward|fell|Traceback`.
- L4-safe `num_envs` = 2048 (script defaults). Override with `NUM_ENVS=...` only if VRAM allows.
- rsl-rl is **≥5.0** (actor/critic `RslRlMLPModelCfg` + `obs_groups`, no `policy=`/`stochastic`). If you touch the runner cfg, mirror `train.py`'s `build_rsl_rl_cfg`.
- After the board weld, `board_range` being non-zero under an untrained policy is EXPECTED (board follows the feet during falls), not a bug.

## 8. ESCALATION / ORCHESTRATOR MODEL (cost control)

The user runs a lead "orchestrator" agent for judgment calls to save cost. **Stop and report back**
(the user relays to the orchestrator) when:
- the start is/ isn't learnable (the fall-rate trend) — before any full Stage I,
- each checkpoint in the ladder,
- an error you can't fix in 2 attempts,
- a run exceeds its rough time budget by >1.5×.

Report format: exact command run, last ~30 log lines, checkpoint paths, the reward+termination table.

## 9. CURRENT LIVE STATE (as of handoff)

- Branch `gpu-l4-bringup`, latest commit `2e14810`. All physics fixes committed; working tree clean.
- A Stage I run was left running at iter ~193/200 with **fell=1.0 (not learning)** — it is futile.
  **Recommended:** stop it (`docker kill $(docker ps -q)`), apply the Section-5 fixes, then re-probe.
- Git rule: commit to `gpu-l4-bringup` only. **Never merge to `main`** without the user's say-so.
