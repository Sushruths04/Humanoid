# Obsidian Learning Vault — Build Plan & Knowledge Dump

> **Audience:** an AI agent (e.g. Claude Sonnet) running in a fresh chat, plus Sushruth reading the result in Obsidian.
> **Goal:** turn everything done in this physical-AI / humanoid-RL program into a connected, teachable Obsidian vault — env setup, training recipes, every parameter tuned, what was achieved, and (most valuable) **how things failed and how each failure was solved.**
> **This file is self-contained:** all the facts you need are written below, because the chat that produced them has been compacted. Do not assume access to prior conversation. Verify against the repo where noted.

---

## 0. Your mission (executing agent, read first)

Create an Obsidian vault under **`docs/vault/`** in this repo. It is a set of interlinked markdown notes using `[[wikilink]]` syntax. Build it from the knowledge in §3–§9 below and by **linking to the existing repo markdown files** listed in §2. Then commit on branch `feat/planned-scripts`.

Rules:
- **Obsidian wikilinks**: `[[Note Title]]` or `[[Note Title|display text]]`. Use them liberally to cross-connect notes.
- **Tags**: put YAML frontmatter on every note with `tags:` (e.g. `tags: [setup, isaac-lab, failure, rl]`).
- **One concept per note**, small and focused. Link generously.
- **Every note ends with a "Related" section** of `[[wikilinks]]`.
- Start from the **MOC (Map of Content)** note `vault/00 - START HERE.md` that links to everything.
- For links to existing repo docs that live OUTSIDE the vault folder, use a relative markdown link AND mention the path, e.g. `[GPU VRAM table](../GPU_VRAM_REQUIREMENTS.md)`, because Obsidian resolves repo-relative paths if the whole repo is opened as the vault. (Recommended: tell Sushruth to open the **repo root** as the Obsidian vault so every `.md` is reachable.)
- Keep code/parameters in fenced code blocks so they're copy-pasteable.
- Do NOT invent numbers. Every metric/parameter below is real and measured. If you add anything new, mark it clearly as inferred.

---

## 1. Vault structure to create

```
docs/vault/
  00 - START HERE.md                  ← MOC, links to all notes + repo docs
  setup/
    Lightning Studio Environment.md
    Isaac Sim Docker Container.md
    GHCR Image & Auth.md
    PYTHONPATH & Python Interpreters.md
    SSH Key Recovery.md
  concepts/
    Isaac Lab Manager-Based RL.md
    Command-Conditioned Navigation.md
    Velocity-Command Steering Law.md
    Reward Shaping & Progress Rewards.md
    PPO with RSL-RL.md
    World Models (Dreamer-mini).md
    Frozen Text Encoder for Language Tasks.md
    Sequential Subgoal Navigation.md
  tasks/
    P0 - CommandNav.md
    P1.2 - LangNav.md
    P1.3 - ObstacleNav.md
    P1.4 - SeqNav.md
    P2 - World Model.md
  workflow/
    Training Recipe.md
    Evaluation Harness.md
    Rendering Demo Videos.md
    Reproduce From Scratch.md
  failures/
    00 - Failure Index.md            ← table of every failure + link
    Decorative Navigation Defect.md
    SeqNav Stand-Still Local Optimum.md   ← the headline debugging story
    Results Lost to Ephemeral Container Storage.md
    Eval Crash - Missing Buffer.md
    GHCR Auth Denied.md
    container.py Forces Rebuild.md
    SSH Heredoc Apostrophe Corruption.md
    Video Render Never Exits.md
    Stuck Wrapper Waiting on Lingering Process.md
  reference/
    All Parameters Cheat-Sheet.md
    Common Failure Patterns.md
    Glossary.md
```

---

## 2. Existing repo markdown to LINK INTO the vault (do not duplicate, link)

Governing plans & specs:
- `docs/MASTER_ROADMAP_CONVERGED.md` — the governing program plan (Option 3 converged)
- `docs/PHYSICAL_AI_6MONTH_PLAN.md` — humanoid track P0–P4 checkpoints
- `docs/TABLETOP_MANIPULATION_PLAN.md` — manipulation track T0–T4
- `docs/GPU_VRAM_REQUIREMENTS.md` — per-task VRAM sizing (rent the right card)
- `docs/PLANNED_SCRIPTS.md` — batch GPU test runbook + script inventory

Results:
- `docs/results/p0_baseline.md` — CommandNav 94.5%
- `docs/results/p1_langnav.md` — LangNav 98.8%
- `docs/results/humanoid-g1-obstaclenav-v0.md` — ObstacleNav 85.9%
- `docs/results/humanoid-g1-seqnav-v0.md` — SeqNav 80.9% / 94.5%

Infra / ops runbooks (older, still useful — cross-link but note some predate the public-image change):
- `docs/setup/LIGHTNING_SSH_SETUP.md`, `MACHINE_CHANGE_RUNBOOK.md`, `MACHINE_SWITCH_QUICK_REF.md`, `CPU_TO_GPU_MACHINE_SWITCH.md`, `DOCKER_IMAGE_REUSE.md`, `RECOVERY_GUIDE.md`, `REMOTE_WORKFLOW.md`, `LIGHTNING_BACKUP_WORKFLOW.md`, `AGENT_HANDOFF.md`, `START_HERE.md`

Code to reference (link to the file path, summarize what it does in the relevant note):
- `programs/common/commands.py` — sampling + steering laws
- `programs/common/rewards.py` — `commanded_target_reward`, `collision_penalty`
- `programs/common/sequence.py` — `advance_subgoal`, `sample_subgoal_sequence`
- `programs/common/text_encoder.py` — frozen MiniLM command cache
- `programs/common/eval/metrics.py` — `compute_episode_metrics`, `success_rate_by_command`, `sequence_eval_metrics`
- `programs/common/eval/evaluate.py` — single-target rollout evaluator
- `programs/common/eval/evaluate_seq.py` — sequence-aware rollout evaluator
- `programs/world_model/{rssm.py,agent.py,train_wm.py}` — Dreamer-mini
- `my-humanoid-project/my_humanoid_project/tasks/g1_*_cfg.py` — the Isaac Lab task configs
- `my-humanoid-project/custom_train.py`, `custom_play.py` — train/record entry points
- `programs/scripts/train_eval_nav.sh`, `batch_test_nav.sh` — launchers

---

## 3. Environment setup (write into setup/ notes)

**Platform:** Lightning AI Studio. Persistent repo path: `/teamspace/studios/this_studio/Humanoid`. SSH: `s_01kt558jf0ra2chne251dtsg8k@ssh.lightning.ai`.

**What persists vs. what is ephemeral across machine restarts (CRITICAL mental model):**
- PERSISTS: the `/teamspace/studios/this_studio/...` filesystem (code, conda env, git, ssh config, HF token cache).
- EPHEMERAL: **Docker** — images and containers are wiped on every machine swap/restart. You must re-pull + re-create the container every time you get a fresh GPU machine.

**Container image:** `ghcr.io/sushruths04/humanoid-isaaclab:latest` (≈17.6 GB, based on `nvcr.io/nvidia/isaac-sim:5.1.0`). It is now **PUBLIC** on GHCR, so `docker pull` works anonymously (no token). Check public: `curl -s "https://ghcr.io/token?scope=repository:sushruths04/humanoid-isaaclab:pull" -w "%{http_code}\n" -o /dev/null` → `200` = public.

**Bring the container up (the exact, working sequence):**
```bash
cd /teamspace/studios/this_studio/Humanoid
git checkout feat/planned-scripts && git pull
# 1) pull + tag (Docker is ephemeral)
docker pull ghcr.io/sushruths04/humanoid-isaaclab:latest
docker tag  ghcr.io/sushruths04/humanoid-isaaclab:latest isaac-lab-base
# 2) start with compose (NOT container.py — it forces a rebuild needing NGC base)
cd IsaacLab/docker
touch .isaac-lab-docker-history
DOCKER_NAME_SUFFIX= docker compose --env-file .env.base --profile base up isaac-lab-base -d --no-build
cd ../..
docker ps   # confirm isaac-lab-base is Up
```

**Bind mounts** (patched into `IsaacLab/docker/docker-compose.yaml`): host `programs/` → container `/workspace/programs`, host `my-humanoid-project/` → `/workspace/my-humanoid-project`. NOTE: the repo's `docs/` is **NOT** mounted — see failure note "Results Lost to Ephemeral Container Storage".

**Two Python interpreters — do not mix them up:**
- In-container Isaac Sim python: run via `/workspace/isaaclab/isaaclab.sh -p <script.py>` (this is the only one that can import `isaaclab`).
- CPU-only logic + unit tests (no Isaac Sim): `/home/zeus/miniconda3/envs/cloudspace/bin/python`. **A bare `python3` hits the wrong system Python — never use it.**

**PYTHONPATH for in-container runs:** `/workspace:/workspace/my-humanoid-project:/workspace/isaaclab/source`

**GPU measured this session:** NVIDIA L4, 23 GB. Training peak only ~**4.6 GB** at 4096 envs → **16 GB (T4) is plenty for all state-based nav RL.** Only the camera-based vision phases (P3/T3) need 24 GB+ and RT cores (L4/L40S, NOT A100). See `docs/GPU_VRAM_REQUIREMENTS.md`.

**GitHub push:** ssh:// remote + key `~/.ssh/github_humanoid` (gitconfig `insteadOf` rewrites git@→https, so use the `ssh://` remote form).

**Hugging Face:** `hf auth login --token <token>` (new CLI is `hf`, not deprecated `huggingface-cli`). HF user `mitvho09`, model repo `mitvho09/humanoid-g1-nav`. Upload: `hf upload mitvho09/humanoid-g1-nav <local> <repo-path>`.

---

## 4. The tasks, with EXACT parameters (write into tasks/ notes)

All nav tasks subclass `G1FlatEnvCfg` (Unitree G1 humanoid flat-terrain locomotion) and reuse its PPO config `G1FlatPPORunnerCfg` (experiment_name `g1_flat`). Pattern for every nav task:
1. Disable the base random velocity command: `self.commands.base_velocity.resampling_time_range = (1e9, 1e9)` and `heading_command = False`.
2. A **reset event** samples per-episode targets/markers.
3. A **per-step interval event** (`mode="interval", interval_range_s=(0.0, 0.0)` = fire every step) overwrites `command_manager.get_term("base_velocity").vel_command_b[:]` to steer toward the target. So the base locomotion velocity-tracking reward and navigation become aligned — the robot navigates by tracking a command that points at the goal.
4. An **observation term** exposes the command (one-hot id + relative target vector).
5. A **reward term** rewards progress toward the target + a reach bonus.

**Common training recipe:** RSL-RL PPO, `4096` envs, `500` iterations, ~2.4 s/iter → ~20 min on L4. Checkpoints land in `/workspace/isaaclab/logs/rsl_rl/g1_flat/<timestamp>/model_*.pt`. Eval uses `256` envs.

| Task | id | markers / subgoals | target radius | steer (speed/yaw_gain/max_yaw) | reward params | RESULT |
|---|---|---|---|---|---|---|
| **P0 CommandNav** | `Humanoid-G1-CommandNav-v0` | 2 / – | 2–5 m | 1.0 / 0.5 / 1.0 | weight 1.0, progress_scale 1.0, wrong_penalty 1.0, reach_bonus 10 | **94.5%** success (per-cmd [95.8, 93.4]) |
| **P1.2 LangNav** | `Humanoid-G1-LangNav-v0` | 3 / – | 2–5 m | same | same + frozen-text-embedding obs | **98.8%** per-command |
| **P1.3 ObstacleNav** | `Humanoid-G1-ObstacleNav-v0` | 3 markers + obstacles | 2–5 m | uses `velocity_command_to_target_avoiding` (potential-field) | nav weight 1.0 + collision penalty weight 1.0 | **85.9%** goal (per-cmd [83.7, 88.0]), collisions negligible |
| **P1.4 SeqNav** | `Humanoid-G1-SeqNav-v0` | 3 / 2 subgoals | **1–2.5 m** (after fix) | 1.0 / 0.5 / 1.0 | weight 1.0, progress_scale **2.0**, wrong_penalty 1.0, reach_bonus 10 | **80.9%** full-sequence, **94.5%** ordering, 97.7% first-subgoal |

**P2 World Model (Dreamer-mini)** — pure PyTorch, no Isaac Sim. `programs/world_model/`. `RSSM` = GRU deterministic state (deter=64) + stochastic latent (stoch=16, Normal), hidden=64. `WorldModel.observe/.loss/.imagine`; `Actor`/`Critic`/`imagine_returns`. Verified on a toy point-mass: loss 2.7 → 0.11. CPU-tested. Not yet trained on real Isaac rollouts (a remaining task).

**Key SeqNav design (after the rebuild — see failure note):** SeqNav reuses the proven CommandNav functions verbatim (`steer_velocity_to_target`, `nav_command_obs`, `nav_command_reward`, `_ensure_buffers`, `_commanded_target_xy`). Its only extra piece is a `seq_advance` interval event registered **before** the steer event: when the robot is within `reach_radius` of the current subgoal, it advances the phase and re-points `_nav_target_ids` to the next subgoal (and resets `_nav_prev_xy` so the target hop doesn't score a false progress penalty). To the policy it's just CommandNav with a target that hops forward through an ordered sequence.

---

## 5. Training recipe (workflow/Training Recipe.md)

```bash
# inside the repo, container up:
bash programs/scripts/train_eval_nav.sh <TASK_ID> 4096 500 256
# this does, via two docker exec calls into isaac-lab-base:
#  TRAIN: isaaclab.sh -p my-humanoid-project/custom_train.py --task <T> --headless --num_envs 4096 --max_iterations 500
#  EVAL : isaaclab.sh -p programs/common/eval/evaluate.py --task <T> --headless --num-envs 256 --checkpoint <latest> --out programs/results/<t>.md
```
`custom_train.py` registers the custom tasks (`import my_humanoid_project.tasks`) then hands off to Isaac Lab's RSL-RL `train.py`. Run detached with `nohup ... &` and tail the log; sim boot takes ~1–2 min before iteration 0.

**Reading training health:** the per-iteration log prints `Mean reward` (total) and `Episode_Reward/<term>` breakdown. THE key diagnostic: watch the **task-specific reward term** (e.g. `Episode_Reward/nav_command`), not just total reward. A rising *total* with a flat *nav* term means the policy is farming base locomotion reward without navigating (see the stand-still failure). Healthy nav bootstrap: total climbs from ≈ −5 to positive around iter ~150–250 AND the nav term grows toward ~1–3.

---

## 6. Evaluation harness (workflow/Evaluation Harness.md)

- `programs/common/eval/metrics.py` (pure, CPU-tested):
  - `compute_episode_metrics(reached, fell, final_distance, episode_length)` → success_rate, fall_rate, etc.
  - `success_rate_by_command(reached, command_ids, num_commands)` → per-command rates.
  - `sequence_eval_metrics(reach_steps, num_subgoals)` → `full_sequence_success`, `ordering_accuracy`, `first_subgoal_rate`. `reach_steps[i,k]` = first step subgoal k was reached (−1 if never). Full sequence = all reached AND in non-decreasing step order. Ordering accuracy = of episodes that reached all subgoals, fraction in correct order.
- `evaluate.py` — single commanded target; reads `_nav_markers_xy`, `_nav_target_ids`. Uses `handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))` before `OnPolicyRunner` (else `KeyError: class_name`).
- `evaluate_seq.py` — sequential; snapshots the first-episode `_seq_targets` + `_nav_markers_xy`, tracks per-subgoal first-reach step, then calls `sequence_eval_metrics`. Also logs diagnostics (mean_min_dist per subgoal, displacement, mean command magnitude) — these are what cracked the SeqNav bug.

---

## 7. Rendering demo videos (workflow/Rendering Demo Videos.md)

```bash
# custom_play.py = registers tasks, then hands off to stock RSL-RL play.py
isaaclab.sh -p my-humanoid-project/custom_play.py \
  --task <TASK_ID> --num_envs 16 --checkpoint <model_499.pt> \
  --video --video_length 600 --headless
# mp4 -> /workspace/isaaclab/logs/rsl_rl/g1_flat/<run>/videos/play/rl-video-step-0.mp4
```
**Gotchas (write as a note):** (1) `play.py` keeps the sim loop running *forever* after the video is saved — wait for the mp4 to appear, then `pkill -f custom_play.py`. (2) `simulation_app.close()` hard-exits before Python flushes stdout, so any diagnostic prints are lost — write them to a file with explicit flush instead. (3) camera video needs RT cores (L4 fine, A100 not). The first frame is slow (RTX pipeline warm-up).

---

## 8. FAILURES — the most valuable section. One note each in failures/.

For each: **Symptom → Root cause → Fix → Lesson.** Build `failures/00 - Failure Index.md` as a table linking all of them.

### 8.1 SeqNav Stand-Still Local Optimum  ★ headline story
- **Symptom:** SeqNav trained "successfully" (total reward +8) but eval showed 0.4% full-sequence success; robot barely moved (0.5 m displacement over a whole episode); reached first subgoal only ~6%.
- **Investigation (the method — teach this):**
  1. Built a sequence-aware evaluator because the old `evaluate.py` crashed on SeqNav (`AttributeError: _nav_target_ids` — different buffers).
  2. Instrumented it: `mean_robot_displacement=0.5`, `mean_cmd_vel_norm=0.25`, target distance sane (3.5 m). → robot stands still while "tracking" a low command.
  3. Checked training breakdown: `Episode_Reward/nav_command ≈ 0.001` while `track_lin_vel_xy_exp ≈ 0.85`. → policy farms base locomotion reward, ignores navigation.
  4. Ruled out code: rebuilt SeqNav on the *proven* CommandNav reward/steer/obs — still failed. A direct env diagnostic confirmed reset target = 3.4 m away and steer command vx_max ≈ 0.99 toward target (correct). So target, command, reward were all CORRECT.
- **Root cause:** a **training bootstrap / exploration failure**, NOT a bug. With two sequential subgoals 2–5 m apart, the policy could not reach subgoal-0 early enough (before episode timeout) to ever trigger the reach-bonus, so it never started the reach→bonus→progress learning loop, and settled into the stand-still local optimum (standing perfectly tracks a small command = easy reward).
- **Fix:** move targets closer (`RADIUS_RANGE` 2–5 m → **1.0–2.5 m**) so the first reach is achievable early, plus restore the wrong-marker shaping (`wrong_penalty_scale` 0 → 1.0) and `progress_scale` 1 → 2. Result: `nav_command` reward jumped ~100×, navigation bootstrapped by iter ~150, final **80.9% / 94.5%**.
- **Lesson (generalize):** *If a task-specific reward term stays flat near zero while total reward rises, you have a bootstrap/exploration problem, not a reward-magnitude problem. Make the FIRST success easier (closer goals / curriculum), don't just crank reward weights — scaling ~0 progress by 10× is still ~0.* Also: never trust total reward as "it's working" — always inspect the per-term breakdown and run a behavioral eval.

### 8.2 Decorative Navigation Defect (the original meta-lesson)
- **Symptom:** original repo "navigation" wasn't real — fixed command hash ("walk forward"), fixed markers, reward = plain velocity tracking. The language/markers were decorative; behavior didn't depend on them.
- **Fix:** built genuine command-conditioned tasks (randomized target+markers per episode, command-conditioned reward, steering that points the velocity command at the commanded target). Verified with an instruction-swap probe (behavior changes with the command).
- **Lesson:** a model that "looks like it works" can be ignoring its inputs. Always probe that behavior is *causally* conditioned on the command.

### 8.3 Results Lost to Ephemeral Container Storage
- **Symptom:** `train_eval_nav.sh` exited 0 but no result file on the host.
- **Root cause:** `evaluate.py --out docs/results/...` wrote a *relative* path inside the container; `docs/` is not bind-mounted, so it landed in throwaway container storage (`/workspace/docs/...`).
- **Fix:** write results under the bind-mounted `programs/results/` (→ persists on host), then mirror into `docs/results/`. Added `programs/results/` to `.gitignore`. Also: container writes are root-owned; `chown` them to host uid (1000) or write logs to a host-owned dir.
- **Lesson:** know exactly which paths are bind-mounted; relative paths in a container are a trap.

### 8.4 Eval Crash — Missing Buffer
- **Symptom:** `AttributeError: 'ManagerBasedRLEnv' object has no attribute '_nav_target_ids'` when evaluating SeqNav.
- **Root cause:** the original SeqNav used its own `_seq_*` buffers; `evaluate.py` assumed CommandNav's `_nav_*` buffers and single-command success metric.
- **Fix:** wrote `evaluate_seq.py` + `sequence_eval_metrics` (TDD: test first → red → implement → green) for full-sequence + ordering metrics.
- **Lesson:** sequential tasks need sequence metrics, not single-target ones. Match the evaluator to the task.

### 8.5 GHCR Auth Denied
- **Symptom:** `docker pull` → `denied` / `unauthorized` on a fresh machine.
- **Root cause:** the stored GHCR token (a `ghp_` PAT that had been pasted into chat and therefore had to be revoked) no longer worked.
- **Fix:** made the GHCR package **public** (GitHub → package → settings → change visibility → Public, type the package name to confirm). Then anonymous pull works. Alternative: `docker login ghcr.io -u <user>` with a fresh `read:packages` PAT.
- **Lesson:** never paste long-lived tokens into chat; rotate immediately if you do. Public image = no auth headache for pulls.

### 8.6 container.py Forces Rebuild
- **Symptom:** the repo's `IsaacLab/docker/container.py start` tries to BUILD the image (needs the NGC `nvcr.io/nvidia/isaac-sim` base you may not be logged into).
- **Fix:** bypass it — use `docker compose ... up isaac-lab-base -d --no-build`, with `touch .isaac-lab-docker-history` and `DOCKER_NAME_SUFFIX=` to satisfy compose. (Full command in §3.)
- **Lesson:** when a helper script insists on building, drop to raw compose with `--no-build` against a pre-pulled image.

### 8.7 SSH Heredoc Apostrophe Corruption
- **Symptom:** writing a Python file over SSH with a `cat <<'EOF'` heredoc corrupted the file mid-write; later `py_compile` failed with an unterminated string.
- **Root cause:** the file content contained an apostrophe (`subgoal k's`) which closed the outer single-quoted SSH argument.
- **Fix:** stop heredoc-ing code over SSH. Write the file locally and `scp` it. This became the standard method for all code edits this session.
- **Lesson:** transfer code as files (scp), never inline through nested shell quoting.

### 8.8 Video Render Never Exits / 8.9 Stuck Wrapper
- `play.py` loops forever after saving the mp4 → grab the file then kill it.
- A `while pgrep <proc>; do sleep; done` wrapper hung because the watched Isaac Sim process lingered in shutdown → kill the lingering process (inside the container) to release the wrapper, or launch the next job directly.
- **Lesson:** Isaac Sim processes linger during teardown; don't gate critical follow-up work on `pgrep`-polling a sim process.

### 8.10 SSH Key Drops on Restart
- **Symptom:** `Permission denied (publickey)` after the Studio sleeps/restarts.
- **Fix:** re-download the key via the Lightning ssh-gen / ssh-public PowerShell URLs (token `t=7a558fd4-...`). `ssh-gen` returns HTTP 500 if the key already exists, but the `ssh-public` URL still works.

---

## 9. Reference notes

**reference/All Parameters Cheat-Sheet.md** — collect the table in §4 + training recipe (§5) + RSSM dims (§4 P2) into one quick-reference.

**reference/Common Failure Patterns.md** — a table:

| Pattern | Tell | Fix |
|---|---|---|
| Reward farming / decorative | total reward up, task-term flat | per-term breakdown + behavioral probe; make first success easier |
| Ephemeral container loss | exit 0 but no host file | use bind-mounted paths |
| Stale auth on fresh machine | pull/push denied | re-pull image, re-login, make public, re-download ssh key |
| Forced rebuild | helper wants NGC base | raw `docker compose --no-build` |
| Quoting corruption | file broken after SSH heredoc | scp files instead |
| Lingering sim process | wrapper hangs / GPU "busy" | kill in-container python; don't poll pgrep |

**reference/Glossary.md** — Isaac Lab, ManagerBasedRLEnv, EventTerm (reset vs interval), RSL-RL, PPO, G1, vel_command_b, RSSM, potential field, bootstrap/local optimum, DoD.

---

## 10. Remaining program tasks (put in `00 - START HERE.md` as a TODO board, and the handoff prompt below)

Done ✅: P0 CommandNav (94.5%), P1.2 LangNav (98.8%), P1.3 ObstacleNav (85.9%), P1.4 SeqNav (80.9%/94.5%), P2 world model (CPU-verified). All on `feat/planned-scripts`, HF `mitvho09/humanoid-g1-nav`.

Next, roughly in order:
1. **Side-by-side demo reel** of all nav policies (CommandNav/ObstacleNav/SeqNav) — render each, stitch with ffmpeg. (User explicitly wants this.)
2. **P0 follow-ups:** reduce G1 fall rate (~24–28%); current reward has no strong upright/termination shaping.
3. **P2 on real rollouts:** train Dreamer-mini on Isaac nav rollouts (currently only toy point-mass), show imagination-trained agent > random.
4. **P3 Vision nav:** camera obs, appearance robustness. Needs 24 GB + RT cores (L4/L40S). Pixel-dependence probe.
5. **P4 Cosmos:** Predict/Transfer post-train infra (the one genuine 80 GB / A100 burst). See `programs/cosmos/README.md`.
6. **T-track (manipulation)** and the **C5 converged capstone** per `docs/MASTER_ROADMAP_CONVERGED.md`.

---

## 11. HANDOFF PROMPT (copy this into the new chat)

> I'm continuing a physical-AI / humanoid-RL portfolio program. The repo is at `/teamspace/studios/this_studio/Humanoid` on a Lightning AI Studio (SSH `s_01kt558jf0ra2chne251dtsg8k@ssh.lightning.ai`), GitHub `Sushruths04/Humanoid`, working branch `feat/planned-scripts`. Read `docs/obsidian_plan.md` first — it contains the full env setup, training recipes, all tuned parameters, and every failure + fix from the previous session, and is self-contained.
>
> **Task A — Build the Obsidian vault** exactly as specified in `docs/obsidian_plan.md` §0–§9: create the interlinked notes under `docs/vault/` (wikilinks, frontmatter tags, a START-HERE MOC, link the existing repo markdown listed in §2). Commit to `feat/planned-scripts`. Open the repo root as the Obsidian vault.
>
> **Task B — Then continue the program** from `docs/obsidian_plan.md` §10, starting with the side-by-side demo reel of the three nav policies (render each with `custom_play.py`, stitch with ffmpeg), then P0 fall-rate follow-up. Environment, container bring-up, and all gotchas are in §3/§7/§8 of that file. GPU note: T4/16 GB is enough for nav RL (measured ~4.6 GB); only vision phases need 24 GB+. Rotate any tokens that were exposed in chat.
>
> Work the way the previous agent did: scp code files instead of SSH heredocs; verify the per-reward-term breakdown not just total reward; run a behavioral eval before claiming a checkpoint passes.

---

*Generated at the end of the CP1.3/CP1.4 GPU session (2026-06-05). All metrics and parameters herein are real and measured. Verify file paths against the current repo before relying on them.*
