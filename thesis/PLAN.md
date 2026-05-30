# Mini-Thesis Execution Plan — Language-Commanded Humanoid Loco-Manipulation (VLA)

> **This document is the single source of truth.** It is written for an *executing AI agent* (and for the human, sushruths04@gmail.com).
> Everything runs **on this Lightning AI Studio**. Paths are `~/isaaclab-workspace/...` (persistent home).
> **GOAL:** Train a humanoid (NVIDIA GR00T N1 first, then Unitree G1 in Isaac Lab) to **walk, pick up an object, and act on a natural-language command** ("pick up the red cube"). Two-phase: replicate GR00T -> then custom G1 RL with language conditioning.

---

## 0. HOW THE EXECUTING AGENT MUST WORK (READ FIRST — NON-NEGOTIABLE RULES)

These rules exist because **Lightning AI Studios can sleep, interrupt, or shut down at any time**. We must never lose more than ~30 minutes of work.

### Rule 1 — Resumable, idempotent steps
- Every step writes a marker file `~/isaaclab-workspace/thesis/state/<STEP_ID>.done` when complete.
- Before running a step, check for its `.done` marker. If present, **skip** and log "already done".
- The master `STATE.json` always records `current_step` and `status`. A fresh agent reads it and resumes.

### Rule 2 — Checkpoint every 30 minutes (and after every step)
- A background auto-commit loop (`scripts/autosave.sh`, see section 4) commits `PROGRESS/`, `state/`, `scripts/`, configs, and small logs to git and pushes **every 30 minutes**.
- Model **weights** are too big for git -> they go to `checkpoints/` (persistent) AND are pushed to **Hugging Face Hub** every checkpoint (see section 4.3).
- After finishing any step, the agent runs `bash scripts/checkpoint.sh "<message>"` immediately — do not wait for the timer.

### Rule 3 — Markdown log for EVERY step (Obsidian-friendly)
- After each step, the agent **must** create/update a markdown file in `PROGRESS/step-logs/STEP-XX-<name>.md` containing:
  - What the step was, **exact commands run**, key output (trimmed), result (success/fail), time taken, and any errors + fixes.
  - Use Obsidian conventions: `# Heading`, `- [ ]`/`- [x]` checklists, code fences, and `[[links]]` to related step logs.
- Also append a one-line entry to `PROGRESS/TIMELINE.md` (date, time, step, status).
- These files are the thesis lab-notebook. They must be human-readable.

### Rule 4 — Hands-free
- Running `bash ~/isaaclab-workspace/thesis/run_thesis.sh` must drive the whole pipeline from wherever it left off, with **zero manual prompts**. No interactive flags. Use `--headless`, `-y`, non-interactive Docker config.

### Rule 5 — Simple first, results fast
- Always pick the **smallest configuration that produces a real result** first (fewer envs, fewer steps, smaller dataset). Get an end-to-end success, log it, THEN scale up. Do not start with the biggest run.

### Rule 6 — Verify before claiming done
- A step is "done" only when its **done-criteria command** prints the expected output. Paste that output into the step log. No assertions without evidence.

---

## 1. CURRENT STATE OF THIS STUDIO (verified facts)

| Item | Status |
|------|--------|
| Machine | Lightning AI Studio, currently **CPU-only** (4 vCPU, 15 GB RAM, 387 GB disk). Switch to a **GPU** Studio before Phase 1 Week-3 training. |
| Persistent home | `~/isaaclab-workspace/` **survives** restarts. Docker images do **NOT** survive — they must be rebuilt on the GPU machine. |
| Isaac Lab | Cloned at `~/isaaclab-workspace/IsaacLab/` (v0.54.3, shallow). Python pkg installed in conda env (CPU mode). |
| Reference repos | `humanoid-gym/`, `IsaacLab-Tutorial/`, `awesome-humanoid-learning/` present. |
| Your project | `~/isaaclab-workspace/my-humanoid-project/` — git repo, only README + .gitignore, **no remote yet** (you will push to your empty GitHub via another agent). |
| GR00T N1 | **Not yet cloned.** |
| Built-in loco-manip task | `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/` (G1 humanoid pick-and-place). |

### GPU note
GR00T N1 fine-tuning and Isaac Lab GPU sim **require a GPU Studio**. On Lightning AI, switch this same Studio's compute to a GPU. Because the home dir is persistent, the workspace, plan, progress logs, and checkpoints all carry over. Only Docker images need rebuilding.

---

## 1.5 GPU SELECTION PER STAGE (COST CONTROL — IMPORTANT)

**Golden rule for saving money on Lightning AI:** you are billed per GPU-hour *only while the Studio runs on a GPU machine*. So:
1. Do **all CPU work** (setup, installs, code authoring, data prep with existing datasets, logging, results write-up) on the **free/cheap CPU machine**.
2. Switch to a GPU **only** for the actual train/infer step, run it, then **switch back to CPU or STOP the Studio**.
3. The checkpoint/autosave system means you can stop the GPU at any time without losing work — resume later.
4. **Start on the smallest viable GPU**, confirm the run works end-to-end at tiny scale, and only move up to a bigger GPU if you need speed/scale.
5. Isaac Sim **requires an RTX-class GPU** (it uses RTX ray tracing). Do **not** pick a non-RTX card for Isaac Lab sim. L4 / A10G / L40S / A100 all work.

### Per-stage GPU map

| Step | Compute | Cheapest viable GPU | Why / notes |
|------|---------|---------------------|-------------|
| 00 setup | **CPU** | none | Scaffolding, git, scripts. No GPU. |
| 01 gr00t_install | **CPU** | none | pip install + import check only. |
| 02 gr00t_demo (inference of 2B) | GPU | **L4 (24 GB)** | 2B model inference fits in 24 GB. L4 is the cheapest 24 GB card. T4 (16 GB) *may* work with offload but L4 is safer/cheap. |
| 03 gr00t_gendata | **CPU** (Option A) / GPU (Option B) | none / **L4 or L40S** | Using an existing LeRobot dataset = CPU only. Only synthetic Isaac-Sim generation needs an RTX GPU (L40S best for rendering). **Start with Option A = CPU.** |
| 04 gr00t_finetune | GPU **(biggest cost)** | **A10G/L4 24 GB with LoRA** for the tiny first run; **A100 40 GB** only when you scale up | Do the first short finetune (2000 steps, small batch, LoRA/PEFT) on a single 24 GB card to keep cost low. Move to **one A100 40 GB** only for the full-scale run. Avoid H100/H200 unless you truly need speed — they cost the most. |
| 05 gr00t_eval | GPU | **L4 (24 GB)** | Same as demo — inference only, cheapest 24 GB card. |
| 10 g1_baseline (Isaac Lab RL) | GPU (RTX) | **L4 (24 GB)** @ `num_envs=512` | Headless RL (no camera) runs fine on L4. RTX-class required for Isaac Sim. Scale `num_envs` up on A100 only if throughput is the bottleneck. |
| 11 g1_language (author code) | **CPU** | none | Writing the language-conditioning env code. No GPU. |
| 12 g1_train_eval | GPU (RTX) | **L4** small -> **A100 40 GB** to scale | Train small on L4 first (proves it learns), then one A100 run for final numbers/curves. |
| 20 custom_task | GPU (RTX) | **L4 -> A100** | Same pattern: prove on L4, final run on A100 if needed. |
| 99 collect_results | **CPU** | none | Plots, tables, RESULTS.md. No GPU. |

### Cost tiers (Lightning AI, cheap -> expensive)
`CPU  <  T4  <  L4  <  A10G  <  L40S  <  A100-40GB  <  A100-80GB  <  H100  <  H200/B200`

**Recommended budget strategy for this thesis:**
- **90% of the work on CPU or L4.** L4 is the workhorse: cheap, 24 GB, RTX-class (works for both GR00T inference and Isaac Sim).
- **Rent an A100 40 GB only twice:** once for the full-scale GR00T finetune (STEP 04 scale-up) and once for the final G1 training run (STEP 12/20 scale-up). A few hours each.
- **Never leave a GPU Studio running idle.** Stop it the moment a run finishes — the checkpoint system has already saved everything.
- LoRA/PEFT fine-tuning (STEP 04) dramatically lowers VRAM + cost vs full fine-tuning — prefer it for the thesis.

> Set the active GPU in `config.env` knobs (`G1_NUM_ENVS`, `GR00T_FT_STEPS`) to match the card: small values on L4, larger only on A100.

---

## 2. DIRECTORY STRUCTURE TO CREATE (Phase 0)

```
~/isaaclab-workspace/thesis/
- PLAN.md                  <- this file (read-only reference)
- run_thesis.sh            <- MASTER hands-free orchestrator (resumes from STATE.json)
- STATE.json               <- {current_step, status, started_at, last_checkpoint}
- config.env               <- editable vars: GPU on/off, HF repo, GitHub repo, scale knobs
- scripts/
  - lib.sh                 <- shared funcs: log_step(), mark_done(), is_done(), md_log()
  - autosave.sh            <- background 30-min git auto-commit + push loop
  - checkpoint.sh          <- one-shot: commit+push code/logs, push weights to HF
  - 00_setup.sh
  - 01_gr00t_install.sh
  - 02_gr00t_demo.sh
  - 03_gr00t_gendata.sh
  - 04_gr00t_finetune.sh
  - 05_gr00t_eval.sh
  - 10_g1_baseline_train.sh
  - 11_g1_language_cond.sh
  - 12_g1_train_eval.sh
  - 20_custom_task.sh
  - 99_collect_results.sh
- state/                   <- <STEP_ID>.done markers (committed to git)
- PROGRESS/                <- OBSIDIAN VAULT CONTENT
  - 00-overview.md
  - TIMELINE.md            <- append-only running log
  - step-logs/
    - STEP-XX-*.md         <- one md per step (the lab notebook)
- checkpoints/             <- model weights (gitignored; mirrored to HF Hub)
- data/                    <- generated/teleop datasets (gitignored; large)
- logs/                    <- training logs / tensorboard (small parts committed)
```

`.gitignore` for `thesis/`: `checkpoints/  data/  *.pt  *.pth  *.ckpt  logs/**/events.*  __pycache__/`

---

## 3. STATE & RESUME MODEL

`STATE.json` schema:
```json
{
  "current_step": "01_gr00t_install",
  "status": "in_progress",
  "phase": 1,
  "started_at": "2026-05-30T18:00:00Z",
  "last_checkpoint": "2026-05-30T18:25:00Z",
  "notes": "free text"
}
```
(`status` is one of: not_started | in_progress | done | failed)

**Resume procedure for a fresh agent after an interruption:**
1. `cat ~/isaaclab-workspace/thesis/STATE.json` -> read `current_step`.
2. `ls ~/isaaclab-workspace/thesis/state/` -> see which steps have `.done`.
3. Read the latest `PROGRESS/step-logs/STEP-XX-*.md` to understand last context.
4. Run `bash run_thesis.sh` — the orchestrator skips done steps and continues.

---

## 4. CHECKPOINT / AUTOSAVE SYSTEM (build these in Phase 0)

### 4.1 `scripts/lib.sh` (shared helpers — skeleton)
```bash
#!/usr/bin/env bash
THESIS=~/isaaclab-workspace/thesis
is_done()    { [ -f "$THESIS/state/$1.done" ]; }
mark_done()  { touch "$THESIS/state/$1.done"; }
set_state()  { # $1=step $2=status
  python3 - "$1" "$2" <<'PY'
import json,sys,datetime,os
p=os.path.expanduser("~/isaaclab-workspace/thesis/STATE.json")
d=json.load(open(p)) if os.path.exists(p) else {}
d["current_step"]=sys.argv[1]; d["status"]=sys.argv[2]
d["last_checkpoint"]=datetime.datetime.utcnow().isoformat()+"Z"
json.dump(d,open(p,"w"),indent=2)
PY
}
md_log()     { # $1=step-id $2=title ; reads body from stdin
  f="$THESIS/PROGRESS/step-logs/$1.md"
  { echo "# $2"; echo; echo "_$(date -u +%FT%TZ)_"; echo; cat; } >> "$f"
  echo "- $(date -u +%FT%TZ) | $1 | $2" >> "$THESIS/PROGRESS/TIMELINE.md"
}
log_step()   { echo "[$(date -u +%T)] $*"; }
```

### 4.2 `scripts/autosave.sh` (run in background; commits every 30 min)
```bash
#!/usr/bin/env bash
cd ~/isaaclab-workspace/thesis
while true; do
  git add PROGRESS state scripts STATE.json config.env *.md 2>/dev/null
  git commit -m "autosave $(date -u +%FT%TZ)" 2>/dev/null && git push 2>/dev/null
  sleep 1800   # 30 minutes
done
```
Launch once per session: `nohup bash scripts/autosave.sh > logs/autosave.log 2>&1 &`

### 4.3 `scripts/checkpoint.sh` (explicit checkpoint — code+logs to git, weights to HF)
```bash
#!/usr/bin/env bash
cd ~/isaaclab-workspace/thesis
source config.env
git add PROGRESS state scripts STATE.json config.env *.md
git commit -m "checkpoint: ${1:-manual} $(date -u +%FT%TZ)" && git push
# Weights -> Hugging Face Hub (only if HF_REPO set and huggingface-cli logged in)
if [ -n "$HF_REPO" ] && [ -d checkpoints ] && [ "$(ls -A checkpoints)" ]; then
  huggingface-cli upload "$HF_REPO" checkpoints checkpoints --repo-type model || true
fi
```
> **Why two destinations:** GitHub for code + markdown logs + small logs (fast, versioned, free). Hugging Face Hub for large model weights (git-lfs-backed, free, resumable). This guarantees that if the Studio dies, both your *notebook* and your *weights* survive.

> **Alternative weight backup** if HF not desired: copy `checkpoints/` to a Lightning Drive / S3 bucket. The home dir itself persists across normal restarts, so HF/S3 is the *extra* safety net against total Studio loss.

### 4.4 `config.env` (editable knobs — keeps runs simple-first)
```bash
USE_GPU=0                       # set 1 after switching to GPU Studio
GITHUB_REPO=git@github.com:USER/REPO.git   # you fill in
HF_REPO=USER/humanoid-vla-thesis           # you fill in (optional)
# SCALE KNOBS — start tiny, scale later (Rule 5)
GR00T_DATA_TRAJ=200             # tiny dataset first; raise to thousands later
GR00T_FT_STEPS=2000             # short finetune first
G1_NUM_ENVS=512                 # raise to 2048 on big GPU
G1_MAX_ITERS=300                # short first
```

---

## 5. MASTER ORCHESTRATOR `run_thesis.sh` (skeleton)
```bash
#!/usr/bin/env bash
set -euo pipefail
cd ~/isaaclab-workspace/thesis
source scripts/lib.sh
source config.env
# start autosave once
pgrep -f autosave.sh >/dev/null || nohup bash scripts/autosave.sh > logs/autosave.log 2>&1 &

run() {  # $1 = step id, $2 = script
  if is_done "$1"; then log_step "SKIP $1 (done)"; return; fi
  set_state "$1" in_progress
  log_step "START $1"
  if bash "scripts/$2"; then
    mark_done "$1"; set_state "$1" done
    bash scripts/checkpoint.sh "$1"
    log_step "DONE $1"
  else
    set_state "$1" failed
    log_step "FAILED $1 — see PROGRESS/step-logs/$1.md"; exit 1
  fi
}

# ---- PHASE 0 ----
run 00_setup            00_setup.sh
# ---- PHASE 1: GR00T N1 ----
run 01_gr00t_install    01_gr00t_install.sh
run 02_gr00t_demo       02_gr00t_demo.sh
run 03_gr00t_gendata    03_gr00t_gendata.sh
run 04_gr00t_finetune   04_gr00t_finetune.sh
run 05_gr00t_eval       05_gr00t_eval.sh
# ---- PHASE 2: Isaac Lab G1 ----
run 10_g1_baseline      10_g1_baseline_train.sh
run 11_g1_language      11_g1_language_cond.sh
run 12_g1_train_eval    12_g1_train_eval.sh
# ---- PHASE 3: Custom + results ----
run 20_custom_task      20_custom_task.sh
run 99_collect_results  99_collect_results.sh
log_step "ALL STEPS COMPLETE"
```
> Each `scripts/NN_*.sh` MUST: do its work, write its `PROGRESS/step-logs/STEP-NN-*.md` via `md_log`, and exit non-zero on failure. Skeletons of intent are in sections 6-9 below; the executing agent fleshes them out using the *current* upstream READMEs (script/flag names evolve — always verify against the live repo).

---

## 6. PHASE 0 — SETUP (CPU ok) - target <= 1 session

**STEP 00_setup**
- Create the directory tree in section 2; init git in `thesis/`; write `.gitignore`; write `lib.sh`, `autosave.sh`, `checkpoint.sh`, `config.env`, `run_thesis.sh`.
- Fill `config.env` GITHUB_REPO / HF_REPO (human provides; if blank, autosave still commits locally).
- `git init && git add -A && git commit -m "thesis scaffold"`. (Pushing is done by you/another agent once remote is set.)
- **Done-criteria:** `bash run_thesis.sh` runs and immediately SKIPs nothing-yet, autosave process is alive (`pgrep -f autosave.sh`).
- **md log:** STEP-00-setup.md — record tree created, files written.

---

## 7. PHASE 1 — GR00T N1 REPLICATION (needs GPU from STEP 04) - Weeks 1-4

Repo: `https://github.com/NVIDIA/Isaac-GR00T`  · Paper: `https://arxiv.org/pdf/2503.14734`
> GR00T N1 = open humanoid VLA foundation model. **System 2** = Eagle-2 vision-language backbone (understands the image + the text command). **System 1** = diffusion action head (outputs motor actions). We fine-tune the released `GR00T-N1-2B` checkpoint on a small language-labelled pick task.

**STEP 01_gr00t_install** (CPU ok for install)
- `cd ~/isaaclab-workspace && git clone https://github.com/NVIDIA/Isaac-GR00T`
- Create/activate env per repo README; `pip install -e .` (and extras). Verify import.
- **Done-criteria:** `python -c "import gr00t; print('ok')"` prints ok. Log exact install commands + versions.

**STEP 02_gr00t_demo** (GPU recommended)
- Download `GR00T-N1-2B` from Hugging Face; run the repo's getting-started inference notebook/script on a sample LeRobot dataset to confirm the model loads and produces actions.
- **Done-criteria:** inference script outputs an action tensor for a sample observation. Save a screenshot/log.
- **md log:** include the exact checkpoint name + command + output shape.

**STEP 03_gr00t_gendata** (GPU)
- Build the **smallest** language-labelled dataset first (`GR00T_DATA_TRAJ` from config.env, start = 200).
- Option A (fastest results): use an **existing GR00T-compatible LeRobot demo dataset** (e.g. the repo's sample) and relabel/confirm language instructions like "pick up the object".
- Option B (later, scale-up): generate synthetic trajectories in Isaac Sim via GR00T's data-gen workflow.
- Convert/verify data is in **LeRobot format** the finetune script expects.
- **Done-criteria:** dataset dir has N episodes with `language_instruction` fields; a loader prints one batch.

**STEP 04_gr00t_finetune** (GPU REQUIRED)
- Run the repo's finetune script (verify name in README, typically `scripts/gr00t_finetune.py`) on the dataset.
- Use small `GR00T_FT_STEPS` first (2000). **Configure checkpoint saving every ~500 steps into `thesis/checkpoints/gr00t/`.** After each saved ckpt, the next `checkpoint.sh` (auto every 30 min) pushes weights to HF.
- **Done-criteria:** training loss decreases; >=1 checkpoint saved under `checkpoints/gr00t/`.
- **md log:** loss curve numbers, steps, time/step, GPU used.

**STEP 05_gr00t_eval** (GPU)
- Run inference with the fine-tuned checkpoint on held-out prompts. Record success: does the humanoid execute the commanded pick?
- Produce a small results table (prompt -> success/fail) + a rollout video/gif if the harness supports it.
- **Done-criteria:** `PROGRESS/step-logs/STEP-05-eval.md` contains a results table with >=5 prompts.
- **Phase-1 deliverable:** "GR00T N1 fine-tuned, responds to language pick commands at X% on N prompts."

---

## 8. PHASE 2 — ISAAC LAB G1 + LANGUAGE CONDITIONING - Weeks 5-9

Base env (already present):
`IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/locomanipulation_g1_env_cfg.py`
Configs: `.../locomanipulation/configs/{action_cfg.py, pink_controller_cfg.py, agile_locomotion_observation_cfg.py}` · MDP: `.../locomanipulation/mdp/`

**STEP 10_g1_baseline** (GPU) — run stock task, no language, smallest scale
- Build Isaac Lab Docker on GPU: `cd IsaacLab && python docker/container.py build && python docker/container.py start && python docker/container.py enter`.
- List tasks: `isaaclab -p scripts/environments/list_envs.py | grep -i "G1\|loco"` to get the exact task id.
- Train tiny: `isaaclab -p scripts/reinforcement_learning/rsl_rl/train.py --task <G1-LocoManip-id> --headless --num_envs $G1_NUM_ENVS --max_iterations $G1_MAX_ITERS`.
- **Configure RSL-RL `save_interval`** so checkpoints land in a host-mounted `logs/` (NOT inside the container image) -> mirror to `thesis/checkpoints/g1_baseline/`.
- **Done-criteria:** a checkpoint `model_*.pt` exists; reward trends up. Log reward curve.

**STEP 11_g1_language** (CPU ok to author code) — add language conditioning (the thesis contribution)
- In `my-humanoid-project/envs/`, subclass the G1 loco-manip env. Add:
  1. A small set of language commands (e.g. "pick up the red cube", "carry the cube to the goal").
  2. Encode each command once with a frozen text encoder (CLIP text or SentenceTransformers `all-MiniLM`) -> fixed embedding vector.
  3. Sample one command per episode at reset; append its embedding to the policy **observation** (extend `agile_locomotion_observation_cfg.py`).
  4. Shape reward for the commanded subgoal in `mdp/rewards.py` (reach object -> grasp -> carry -> place), gated by which command is active.
- Keep it **simple first**: start with just 2 commands ("pick up cube" vs "stand still"/"walk forward") to prove the policy *conditions* on language before adding more.
- **Done-criteria:** env builds; `obs` dim includes the language embedding; a random policy steps without error.

**STEP 12_g1_train_eval** (GPU)
- Train the language-conditioned policy (RSL-RL or SKRL). Start small, then scale `num_envs`/iters.
- Evaluate: success rate **per command**; and the key test — give command A vs B in the same scene and confirm behavior changes.
- **Done-criteria:** results table: command -> success%. Target >=60% on the primary pick command at modest scale.
- **md log:** learning curves, per-command success, comparison vs baseline (STEP 10) and vs GR00T (STEP 05).

---

## 9. PHASE 3 — CUSTOM TASK + RESULTS - Weeks 10-12

**STEP 20_custom_task** (GPU)
- Add ONE novel, not-too-complex scenario in `my-humanoid-project/envs/`:
  - e.g. distractor objects: "pick up the **RED** cube" (ignore the blue one), OR a 2-step "pick up cube, place in box".
- Reuse the language-conditioning machinery from STEP 11. Train + eval.
- **Done-criteria:** >=50% success on the custom instruction; results logged.

**STEP 99_collect_results** (CPU ok)
- Auto-generate `PROGRESS/RESULTS.md`: pulls the per-step tables into one comparison:
  - GR00T N1 finetune (Phase 1) vs G1 language-conditioned RL (Phase 2) vs custom task (Phase 3).
  - Metrics: task success %, sample efficiency, generalization to unseen prompts.
- Collect learning-curve plots, sample rollout media, and the final checkpoint locations.
- **Done-criteria:** `RESULTS.md` exists with a populated comparison table + links to checkpoints on HF.

---

## 10. THESIS DELIVERABLES (what you walk away with)
1. `PROGRESS/` = full Obsidian lab-notebook (every step, every command, every result).
2. Two trained approaches with numbers (GR00T finetune + custom G1 language-conditioned RL).
3. `RESULTS.md` comparison table + plots + rollout videos.
4. All code in your GitHub repo; all weights on Hugging Face Hub.
5. A clean narrative for the thesis: problem -> method (language conditioning for humanoid loco-manip) -> results -> ablation (with vs without language) -> future work.

---

## 11. READING LIST (do alongside, log notes in `PROGRESS/reading/`)
1. GR00T N1 — https://arxiv.org/pdf/2503.14734 (dual-system VLA)
2. WholeBodyVLA — https://arxiv.org/pdf/2512.11047 (unified latent VLA loco-manip — closest to thesis)
3. Isaac Lab — https://arxiv.org/html/2511.04831v1 (the sim framework)
4. FALCON / SkillBlender / ULTRA — listed in `~/isaaclab-workspace/awesome-humanoid-learning/README.md`

---

## 12. RISK / GOTCHA NOTES FOR THE EXECUTING AGENT
- **Studio interruptible:** never run >30 min without an autosave/checkpoint having fired. Confirm `autosave.sh` is alive at session start.
- **Docker images don't persist** across Studio compute switches -> rebuild on GPU (Phase 1 STEP 04 / Phase 2 STEP 10). Re-pull may need `docker login nvcr.io` ($oauthtoken + NGC key).
- **Weights never go to git** (size). HF Hub or Lightning Drive only. Code/markdown/small logs go to git.
- **Upstream script/flag names change** — always confirm against the live README of Isaac-GR00T and IsaacLab before running; record the exact verified command in the step log.
- **Verify GPU** at the start of any training step: `nvidia-smi` must show a GPU, else stop and tell the human to switch the Studio to GPU compute.
- **venv blocked on Lightning** — install into the conda env, not `python -m venv`.
- **Scale up only after a small run fully succeeds** (Rule 5).

---
_End of plan. The executing agent: start by reading STATE.json, then run `bash ~/isaaclab-workspace/thesis/run_thesis.sh`._
