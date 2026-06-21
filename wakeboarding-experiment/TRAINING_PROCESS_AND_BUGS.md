# Wakeboarding RL — Full Training Process & Every Bug Fixed

> **What this document is:** A complete record of every bug encountered, how it was diagnosed,
> and how it was fixed during the G1 wakeboarding Stage I training campaign (2026-06-20 to 2026-06-21).
> Useful for interviews, onboarding new agents, and debugging Stage II.

---

## Overview

Training a Unitree G1 humanoid to wakeboard using PPO (RSL-RL) on Isaac Lab is harder than
standard locomotion because:
- External rope force (not internal locomotion)
- Foot-board weld constraint (rigid binding, no contact adjustment)
- Two-stage crouch→stand biomechanics under active pull
- Serverless GPU (Modal) adds infra constraints on top of physics

**Total bugs fixed: 13** across physics, training code, infra, and API layers.

---

## The Full Timeline

### Phase 0: Environment Scaffolding (local CPU)

Code was written and CPU-verified before any GPU run. The scaffolding established:
- `WakeboardStartEnv` (manager-based Isaac Lab env)
- `RopeModel` (spring anchor model)
- `train.py` (resilient PPO loop with chunk-based checkpointing)
- `play.py` (rollout + video capture)

---

### Phase 1: Lightning AI GPU Probes (probes v1–v8)

First real GPU runs on Lightning AI (NVIDIA L4/L40S inside Docker container).

**Probe v1–v4: 256 envs, default weights**
- Result: Fell rate ~90% flat, reward -4.5, complete NaN explosion at ~100 iters
- Found: physics NaN explosion at 256 envs, cannonball pose broken, rope lead too long

**Probe v5–v6: 16 envs, fixes applied**
- Result: Reward still deeply negative (-2.5), no learning
- Found: YAML weights never actually applied (the root cause)

**Probe v7: ROOT CAUSE FIX applied**
- Result: Positive reward for first time (+1.1), policy starts learning

**Probe v8: 400 iters**
- Result: Fell rate 0.64 → 0.25, board_range dominating terminations

---

### Phase 2: Modal Serverless GPU (Stage I training)

Moved from Lightning AI → Modal L40S for cost and persistent volume storage.
New infra bugs emerged on top of the physics bugs already fixed.

**Run 1:** Crashed immediately (entrypoint bug)
**Run 2:** Crashed at iter ~200 (double-indexing crash)
**Run 3:** Ran but 7x too slow (CUDA_LAUNCH_BLOCKING debug flag left in)
**Run 4:** Clean run, model_980.pt = 0% fell rate ✅
**Run 5 (disconnected):** Stopped at iter 1450 (gRPC timeout)
**Run 6 (resumed):** Ran to iter 5000+ — Stage I complete ✅

---

## Every Bug — Detailed

---

### Bug 1: Rope Lead Too Long

**File:** `src/rope_model.py`
**When found:** Probe v1
**Symptom:** Robot experienced no rope pull at episode start; fell immediately without any force opposing the fall.
**Root cause:**
```python
# BEFORE (rope lead = 5.0m)
def reset(self, env_ids, handle_pos, lead=5.0):
    self.anchor_pos[env_ids] = handle_pos[env_ids] + lead * self.pull_dir
```
The virtual anchor spawned 5 metres ahead of the handle. The spring force is proportional to stretch distance — at episode start the anchor is barely moving forward so the robot feels almost no force. The rope had to stretch 5m before any meaningful tension built.

**Fix:**
```python
# AFTER (lead = 0.4m)
def reset(self, env_ids, handle_pos, lead=0.4):
    self.anchor_pos[env_ids] = handle_pos[env_ids] + lead * self.pull_dir
```
Lead 0.4m gives ~320N initial pull force vs 600N saturation — robot feels the rope immediately.

---

### Bug 2: Cannonball Pose Not Used on Auto-Reset

**File:** `src/tasks/wakeboard_start_cfg.py`
**When found:** Probe v2
**Symptom:** Robot started in cannonball at first reset, but after falling, auto-reset returned to default standing pose (0.74m pelvis height). Training was inconsistent — each episode started from a different pose.

**Root cause:**
Isaac Lab's `ManagerBasedRLEnv._reset_idx()` is called during `step()` when episodes terminate. Our `reset()` override called `_reset_to_cannonball()` and rope reset, but `_reset_idx()` bypassed all of that and called `event_manager.apply(mode="reset")` which uses the default pose.

**Fix:**
```python
def _reset_idx(self, env_ids):
    super()._reset_idx(env_ids)          # Isaac Lab internal reset
    self._reset_to_cannonball(env_ids)   # override pose after default reset
    self._refresh_biomech_buffers()      # update board_pitch, rope buffers
    self.rope.reset(env_ids, self._handle_pos)  # reset rope anchor
```

---

### Bug 3: YAML Reward Weights Never Applied (THE ROOT CAUSE)

**File:** `train.py`
**When found:** Probe v6 — spent multiple sessions rebalancing YAML weights with zero effect
**Symptom:** Every YAML weight change was completely ignored. Positive rewards stayed weak, penalties stayed dominant. Policy learned to freeze.

**Root cause:**
```python
# BUGGY ORDER in train.py:
env_cfg = WakeboardStartEnvCfg()
env = WakeboardStartEnv(env_cfg)              # RewardManager.__init__() runs HERE
                                              # copies term_cfg.weight into _term_weights dict
apply_reward_weights(env.cfg, yaml_weights)   # TOO LATE — dict already frozen
```

Isaac Lab's `RewardManager.__init__()`:
```python
self._term_weights[name] = term_cfg.weight   # copies at construction, not by reference
```
Modifying `term_cfg.weight` after construction has zero effect. Every single probe run used hardcoded defaults from `RewardsCfg` class.

**Fix:**
```python
# FIXED ORDER:
env_cfg = WakeboardStartEnvCfg()
apply_reward_weights(env_cfg, yaml_weights)   # BEFORE construction — modifies the cfg object
env = WakeboardStartEnv(env_cfg)              # RewardManager reads the updated cfg weights
```
Also changed function signature: `apply_reward_weights(env_cfg, ...)` not `apply_reward_weights(env, ...)`.

**Impact:** This was the root cause behind ALL apparent "reward tuning doesn't work" issues. Every prior probe ran with hardcoded defaults.

---

### Bug 4: NaN Physics Explosion at 256+ Envs

**File:** `src/tasks/wakeboard_start_cfg.py`, `configs/smoke.yaml`
**When found:** Probe v1–v4
**Symptom:** All observations become NaN at ~100 iters. Complete training crash. PPO throws `RuntimeError: std >= 0`.

**Root cause:**
PhysX GPU physics becomes unstable with many parallel rigid-body environments when each env has a foot→board weld constraint AND external force applied. GPU memory pressure on L4 causes the physics solver to produce degenerate states.

**Fix:**
1. Capped envs at 16 (16 × 24 steps = 384 steps/iter — minimum for stable PPO gradients)
2. Increased PhysX solver iterations from default to 16 position / 4 velocity iterations
3. Added NaN guard in `step()`:
```python
# Sanitize observations
for k in obs:
    if torch.isnan(obs[k]).any() or torch.isinf(obs[k]).any():
        nan_ids = torch.isnan(obs[k]).any(dim=-1).nonzero()
        obs[k] = torch.nan_to_num(obs[k], nan=0.0, posinf=1.0, neginf=-1.0)
        obs[k] = obs[k].clamp(-10.0, 10.0)
        self.rope.reset(nan_ids, self._handle_pos)  # reset rope for NaN envs
```

---

### Bug 5: pen_dof_pos_limits Returns NaN

**File:** `src/rewards/wakeboard_rewards.py`
**When found:** Probe v8 at iter ~399
**Symptom:** `pen_dof_pos_limits` reward term returns NaN, corrupts PPO gradients, crashes with `RuntimeError: normal expects all elements of std >= 0.0`

**Root cause:**
`soft_joint_pos_limits` can be None or contain NaN if the physics hasn't fully initialized or a joint enters a degenerate state. Division by zero or NaN propagation.

**Fix:**
```python
def pen_dof_pos_limits(env):
    limits = env.robot.data.soft_joint_pos_limits
    if limits is None or torch.isnan(limits).all():
        return torch.zeros(env.num_envs, device=env.device)
    lo = torch.nan_to_num(limits[..., 0], nan=0.0)
    hi = torch.nan_to_num(limits[..., 1], nan=0.0)
    j  = torch.nan_to_num(env.robot.data.joint_pos, nan=0.0)
    return -((j - hi).clamp(min=0) + (lo - j).clamp(min=0)).sum(dim=-1)
```

---

### Bug 6: Board Range Termination Too Tight

**File:** `src/tasks/wakeboard_start_cfg.py`
**When found:** Probe v8
**Symptom:** After fell rate dropped from 64% to 25%, board_range termination jumped to 79% — training stopped making progress because episodes ended too fast.

**Root cause:**
Board range bounds were `[-20°, 45°]`. Early in training, the policy acts randomly and the board swings freely. 20° in the negative direction is easily exceeded.

**Fix:**
Widened bounds to `[-40°, 60°]` so random early exploration doesn't terminate episodes. The agent needs time to experience the board range of motion.

---

### Bug 7: Isaac Container ENTRYPOINT Swallows Commands (Modal)

**File:** `wakeboarding-experiment/modal_app.py`
**When found:** First Modal run
**Symptom:** Modal function exits immediately with no error. No training output at all.

**Root cause:**
The Isaac Sim Docker image sets `ENTRYPOINT ["runheadless.sh"]`. When Modal tries to run `python train.py ...`, it gets passed as an argument to `runheadless.sh`, which silently swallows it and starts Isaac in interactive mode (then immediately exits because there's no display).

**Fix:**
```python
image = (
    modal.Image.from_registry("ghcr.io/sushruths04/humanoid-isaaclab:latest")
    .entrypoint([])    # CRITICAL: clears runheadless.sh entrypoint
    ...
)
```
Then run the full Isaac env setup inline in a bash shell command:
```python
shell_cmd = (
    "ln -sf /isaac-sim/kit/python/bin/python3 /usr/local/bin/python && "
    "export ISAAC_PATH=/isaac-sim EXP_PATH=/isaac-sim/apps "
    "CARB_APP_PATH=/isaac-sim/kit LD_PRELOAD=/isaac-sim/kit/libcarb.so "
    "RESOURCE_NAME=IsaacSim && "
    "source /isaac-sim/setup_python_env.sh && "
    + " ".join(cmd)
)
subprocess.run(["bash", "-c", shell_cmd], check=True, cwd=_REMOTE_DIR)
```

---

### Bug 8: rope.reset() Double-Indexing — ROOT CAUSE OF ALL MODAL CUDA CRASHES

**File:** `src/tasks/wakeboard_start_cfg.py`, `src/rope_model.py`
**When found:** Modal runs 1–3, multiple CUDA crashes
**Symptom:** CUDA crash with out-of-bounds index error during NaN recovery. Intermittent — only triggered when any env went NaN (i.e., first ~100 iters).

**Root cause:**
```python
# BUGGY — in wakeboard_start_cfg.py NaN recovery:
nan_ids = ...  # shape: [k] where k < num_envs
self.rope.reset(nan_ids, self._handle_pos[nan_ids])  # pre-sliced to shape [k, 3]

# In rope_model.py reset():
def reset(self, env_ids, handle_pos, lead=0.4):
    self.anchor_pos[env_ids] = handle_pos[env_ids] + ...  # indexes [k,3] with env_ids!
    # handle_pos is already [k,3] but env_ids indexes into it using original env indices
    # e.g. env_ids=[5,12,31] but handle_pos only has indices [0,1,2] → CRASH
```

`rope.reset()` was designed to receive the **full** `(num_envs, 3)` tensor and index it internally. Passing a pre-sliced tensor and then indexing again caused out-of-bounds.

**Fix:**
```python
# FIXED — pass FULL tensor, let rope.reset() index internally:
self.rope.reset(nan_ids, self._handle_pos)   # full (num_envs, 3) tensor
```
Docstring updated:
```python
def reset(self, env_ids, handle_pos, lead=0.4):
    """handle_pos must be full (num_envs, 3) — NOT pre-sliced"""
    self.anchor_pos[env_ids] = handle_pos[env_ids] + lead * self.pull_dir
```

**Impact:** This was the root cause of all Modal CUDA crashes. Once fixed, training ran stably to iter 980 without a single crash.

---

### Bug 9: .spawn() Exits Immediately (Modal)

**File:** `modal_app.py`
**When found:** First Modal training attempt
**Symptom:** `modal run modal_app.py` returned instantly. No training logs. App showed 0 tasks.

**Root cause:**
```python
# BUGGY:
train.spawn()   # fires the remote function but returns immediately
# local entrypoint exits → Modal cancels the spawned job
```
Modal cancels detached jobs when the local entrypoint exits.

**Fix:**
```python
# FIXED:
train.remote()  # blocks until the remote function completes
```

---

### Bug 10: CUDA_LAUNCH_BLOCKING=1 — 7x Training Slowdown

**File:** `modal_app.py` (environment variable)
**When found:** After rope.reset() fix — training was 7.9s/iter instead of ~1.1s/iter
**Symptom:** Training at iter 980 after 2h09min. Expected: ~18 minutes for 980 iters. Actual: 2+ hours.

**Root cause:**
`CUDA_LAUNCH_BLOCKING=1` was added to the Modal image environment to get synchronous CUDA errors during debugging (makes CUDA ops blocking/synchronous). This was left in after the rope.reset() bug was found and fixed. It serializes all CUDA operations — 7x slowdown.

**Fix:**
Removed `CUDA_LAUNCH_BLOCKING=1` from the Modal image `.env()` and relaunched. Speed returned to ~1.1s/iter.

**Lesson:** Always remove debug flags before long training runs. This wasted ~1.5 hours of L40S time.

---

### Bug 11: ffmpeg Not Found at Runtime (Modal)

**File:** `modal_app.py`
**When found:** During render_video function testing
**Symptom:** `FileNotFoundError: ffmpeg not found` when play.py tried to encode frames into mp4.

**Root cause:**
`apt-get install ffmpeg` was being run inside the Modal function at runtime (inside `subprocess.run()`). This ran inside the already-running container, not during image build.

**Fix:**
Moved ffmpeg installation to image build time:
```python
image = (
    modal.Image.from_registry(...)
    .entrypoint([])
    .run_commands("apt-get update -qq && apt-get install -y -q ffmpeg || true")  # baked in
    ...
)
```

---

### Bug 12: gRPC Deadline Exceeded — Training Disconnects

**When found:** During Stage I run at iter 1450
**Symptom:** `ConnectionError: Deadline exceeded` — local process disconnected from Modal servers. Training stopped saving new logs.

**Root cause:**
`modal run` holds a long-lived gRPC connection to stream logs. After ~2 hours, the connection times out. The remote function itself keeps running on Modal's side, but the local process can't receive logs or confirm completion.

**Fix:**
Added `--resume` support to `modal_app.py` local entrypoint:
```python
@app.local_entrypoint()
def main(action="train", config="configs/stage1.yaml", resume=""):
    if action == "train":
        train.remote(config=config, resume=resume or None)
```

And to `train.py`:
```python
if args.resume:
    runner.load(args.resume)
```

Relaunch: `modal run modal_app.py --action train --config configs/stage1.yaml --resume /ckpts/wakeboard_stage1/model_1450.pt`

---

### Bug 13: omni.replicator RTX Blocked on Modal (Video)

**File:** `play.py`
**When found:** When trying to capture video frames on Modal
**Symptom:** Camera setup fails silently, no frames captured. Only trace JSON saved, no mp4.

**Root cause:**
Modal runs containers in **gVisor sandbox** which intercepts system calls. The Nvidia kernel module (`/dev/nvidia*`) is not exposed through gVisor. `omni.replicator` needs RTX/Vulkan rendering which requires the real kernel driver → fails.

**Fix (workaround):**
Video rendering must run on **Lightning AI** or local machine with a proper GPU (A2000). play.py gracefully falls back to trace-only when camera setup fails:
```python
try:
    import omni.replicator.core as rep
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")  # correct API
    rep.orchestrator.step(delta_time=0.0)  # correct API (not rep.step.render())
    frames_ok = True
except Exception as e:
    print(f"[play] Camera setup failed: {e}")
    frames_ok = False
```

**Also fixed:** The omni.replicator API had two bugs from an older version:
- `rep.AnnotatorRegistry.get.annotators["rgb"]` → `rep.AnnotatorRegistry.get_annotator("rgb")`
- `rep.step.render()` → `rep.orchestrator.step(delta_time=0.0)`

---

## Training Progression Summary

| Phase | Run | Iters | Fell rate | Key outcome |
|---|---|---|---|---|
| Lightning v1–v4 | 256 envs | 100 | 90% flat | NaN explosion |
| Lightning v5 | 16 envs, cannonball | 100 | 47%→73% | Cannonball works |
| Lightning v6 | YAML rebalanced | 100 | 65%→85% | Weights still ignored |
| Lightning v7 | **Weights fix** | 100 | 79%→81% | **Positive reward!** |
| Lightning v8 | 400 iters | 400 | **64%→25%** | **Policy learning!** |
| Modal run 1 | Entrypoint fix | crash | — | Isaac ENTRYPOINT bug |
| Modal run 2 | rope.reset fix | crash | — | Double-indexing bug |
| Modal run 3 | CUDA_LB removed | 980 | **0%** | **Stable!** 7x speedup |
| Modal run 4 | Disconnected | 1450 | 45% | gRPC timeout |
| Modal run 5 | **Resume 1450** | 5000 | **~5%** | **Stage I complete** |

---

## Final Stage I Results

| Iter | Fell | Timeout | Reward | Notes |
|---|---|---|---|---|
| 0 | 97% | 3% | negative | Random policy |
| 500 | 6% | 94% | ~0.8 | Balance learned |
| 980 | **0%** | **100%** | ~1.5 | Stable riding |
| 5000 | ~5% | 81% | ~1.5 | Full pose + riding |

**Best checkpoint**: `wakeboard-ckpts:/wakeboard_stage1/model_latest.pt`
Also useful: `model_980.pt` (cleanest 0% fell), `model_5700.pt` (5% fell, 81% timeout)

---

## Architecture of the Final Working System

```
train.py
  ├── apply_reward_weights(env_cfg, yaml)   # MUST be before env construction
  ├── WakeboardStartEnv(env_cfg)
  │     ├── _reset_to_cannonball()          # hip/knee/torso angles, CANNONBALL_ROOT_Z=0.55
  │     ├── _reset_idx() override           # rope + buffer reset on auto-reset
  │     ├── step() → rope.step_anchor()    # advance virtual anchor at v_pull m/s
  │     ├── step() → rope.compute_force()  # F = kp*(anchor-handle) + kd*(v_anchor-v_handle)
  │     ├── step() → apply_handle_force()  # external force on palm links
  │     └── step() → NaN guard             # sanitize obs, reset NaN envs
  └── OnPolicyRunner.learn() chunks of 50  # resilient loop, saves model_{done}.pt each chunk

RopeModel
  ├── reset(env_ids, handle_pos)           # FULL tensor — do NOT pre-slice
  ├── step_anchor(dt)                      # anchor advances at v_pull m/s
  └── compute_force(pos, vel)             # spring force capped at 600N

modal_app.py
  ├── .entrypoint([])                      # clears runheadless.sh
  ├── bash shell_cmd with Isaac env setup  # ln -sf python + export + source setup_python_env.sh
  └── train.remote() blocking             # NOT .spawn()
```

---

## Interview-Ready Bug Summary

| # | Bug | Root cause | Fix |
|---|---|---|---|
| 1 | Rope lead 5m | Anchor too far, no tension at start | `lead=0.4` |
| 2 | Auto-reset wrong pose | `_reset_idx()` bypassed cannonball | Override `_reset_idx()` |
| 3 | YAML weights ignored | `apply_reward_weights` called after env construction | Move before `WakeboardStartEnv()` |
| 4 | NaN explosion 256 envs | PhysX GPU instability, memory pressure | Cap at 16 envs + NaN guard + solver iters |
| 5 | pen_dof_pos_limits NaN | `soft_joint_pos_limits` uninitialized | `nan_to_num` guard in reward function |
| 6 | Board range terminates 79% | Bounds too tight for early exploration | Widen to [-40°, 60°] |
| 7 | Isaac ENTRYPOINT swallows cmds | `runheadless.sh` as ENTRYPOINT | `.entrypoint([])` + bash shell_cmd |
| 8 | rope.reset() double-index | Pre-sliced tensor indexed again inside reset() | Pass full `(num_envs,3)` tensor |
| 9 | .spawn() exits immediately | Local process exits → Modal cancels job | `.remote()` blocking call |
| 10 | 7x slowdown | `CUDA_LAUNCH_BLOCKING=1` left in | Remove debug flag |
| 11 | ffmpeg not found | Installed at runtime not build time | `.run_commands()` in image def |
| 12 | gRPC disconnect at 1450 iter | Modal connection timeout ~2h | `--resume` flag + relaunch |
| 13 | No video on Modal | gVisor blocks Nvidia kernel | Use Lightning AI for video |
