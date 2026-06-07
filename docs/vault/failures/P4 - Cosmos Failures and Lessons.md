---
tags: [failures, p4, cosmos, lessons, post-mortem]
---

# P4 — Cosmos Failures and Lessons

9 documented failures from the P4 Cosmos Predict world simulation task. Each entry has root cause, exact fix, and the rule to remember.

---

## F1: ffmpeg Not Found After Machine Restart

**Symptom**: `RuntimeError: Program 'ffmpeg' is not found` at the start of CP4.4 run.

**Root cause**: ffmpeg was installed via `apt` in the previous Lightning AI session. Lightning AI does NOT persist apt-installed packages across session restarts — only `/teamspace/` storage is persistent.

**Fix**:
```bash
sudo apt-get install -y ffmpeg
```

**Rule**: Run `sudo apt-get install -y ffmpeg` at the start of every Lightning AI session before any script that uses `mediapy`. Add it to the machine setup checklist.

---

## F2: Wrong Pipeline Class — `Video2WorldPipeline` vs `Video2WorldActionConditionedPipeline`

**Symptom**: `TypeError: __init__() got an unexpected keyword argument 'action'` during inference.

**Root cause**: CP4.4 was written using the base `Video2WorldPipeline` class. The action-conditioned variant is `Video2WorldActionConditionedPipeline` — completely different API.

**Fix**:
```python
# WRONG
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
pipe = Video2WorldPipeline.from_pretrained(...)

# CORRECT
from cosmos_predict2.pipelines.video2world import Video2WorldActionConditionedPipeline
pipe = Video2WorldActionConditionedPipeline.from_config(
    config_job_name="Cosmos-Predict2-2B-Video2World-Sample-AV",
    checkpoint_dir="...", checkpoint_name="model.pt", ...
)
```

**Rule**: Cosmos has multiple pipeline classes. Always check exact class for action conditioning. Never use `from_pretrained` — use `from_config` with the config job name.

---

## F3: `padding_mask` Cannot Be None — Must Be bfloat16 Zeros

**Symptom**: `AttributeError: 'NoneType' object has no attribute 'to'` deep inside conditioner forward pass.

**Root cause**: `ActionCondition.padding_mask` is documented as optional but the underlying conditioner unconditionally calls `.to()` on it. Passing `None` crashes.

**Fix**:
```python
# WRONG
condition = ActionCondition(..., padding_mask=None, ...)

# CORRECT
condition = ActionCondition(
    ...,
    padding_mask=torch.zeros(B, 1, H_vid, W_vid, device=device, dtype=pipe.precision),
    ...
)
```

**Rule**: Always pass `padding_mask` as explicit `torch.zeros(...)` with the correct dtype (must match `pipe.precision` = `bfloat16`). Never pass `None`.

---

## F4: Video Temporal Dimension Must Satisfy `(T-1) % 4 == 0`

**Symptom**: `AssertionError` in the causal 3D convolution tokenizer with cryptic shape mismatch.

**Root cause**: Cosmos VAE tokenizer uses causal 3D convolutions with `temporal_window=4`. The input temporal dimension `T` must satisfy `(T-1) % 4 == 0` (valid values: 1, 5, 9, 13, ...). Passing T=12 fails (12-1=11, not divisible by 4).

**Fix**: Pad to T=13 or use T=9 (8 content frames + 1 conditioning frame):
```python
# Pad from T=12 to T=13
if vid.shape[2] % 4 != 1:
    pad = 4 - ((vid.shape[2] - 1) % 4)
    vid = torch.cat([vid, vid[:, :, -1:].expand(-1, -1, pad, -1, -1)], dim=2)
```

**Rule**: Always check `(T-1) % 4 == 0` before passing video tensors to the Cosmos tokenizer. The formula for valid T: `T = 4k + 1` for any non-negative integer k.

---

## F5: `pipe.denoise()` Returns `DenoisePrediction`, Not a Tensor

**Symptom**: `TypeError: unsupported operand type(s) for -: 'DenoisePrediction' and 'Tensor'` in Euler ODE loop.

**Root cause**: `pipe.denoise()` returns a `DenoisePrediction` dataclass, not a raw tensor. The predicted denoised latent is in `.x0` attribute.

**Fix**:
```python
pred = pipe.denoise(x_t, sigma_t, condition)
# WRONG: velocity = (x_t - pred) / t
# CORRECT:
velocity = (x_t - pred.x0) / (t_curr + 1e-8)
```

**Rule**: Always access `.x0` on the return value of `pipe.denoise()`. Check the `DenoisePrediction` dataclass fields before using them.

---

## F6: HDF5 Dataset Format Mismatch — Bridge Data Used Instead

**Symptom**: CP4.4 script was written expecting `g1_nav_cosmos.h5` HDF5 file. File does not exist; Bridge dataset is the actual source.

**Root cause**: The original plan called for exporting G1 nav rollouts to HDF5. In practice, the training (CP4.3) used the LeRobot `bridge_orig` dataset directly. CP4.4 was written to a spec that was never executed.

**Fix**: Rewrote CP4.4 to use Bridge data directly:
```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset("lerobot/bridge_orig", root="/teamspace/datasets/bridge_orig")
```

**Rule**: Before writing evaluation/rollout scripts, confirm which dataset format training actually used. Spec and reality often diverge.

---

## F7: CP4.5 Used Isaac Lab Docker — Not Available

**Symptom**: CP4.5 script called `docker run nvcr.io/nvidia/isaac-lab...` which is not installed/available on the A100 Lightning AI studio.

**Root cause**: CP4.5 was written assuming the same Isaac Lab Docker setup from P3. The P4 studio is a fresh A100 environment with only Cosmos installed, not Isaac Lab.

**Fix**: Rewrote CP4.5 as standalone CEM planner using only Cosmos + Bridge data — no Isaac Lab needed:
```python
# CEM scoring uses single-step x0 prediction at sigma=1.0 for speed
def score_actions(pipe, latent, actions, goal_latent):
    sigma_t = torch.full((B,), 1.0, device=device, dtype=pipe.precision)
    pred = pipe.denoise(noisy_latent, sigma_t, condition)
    return -(torch.mean((pred.x0 - goal_latent)**2, dim=(1,2,3,4)))
```

**Rule**: Never assume Docker is available on Lightning AI studios. Write scripts to be self-contained with only the packages installed in the conda environment.

---

## F8: SSH Key Drop Breaks All Remote Operations

**Symptom**: `Permission denied (publickey)` — all SSH commands to Lightning AI fail after session change.

**Root cause**: Lightning AI assigns a new studio ID when a session changes. The old SSH key config pointed to the old studio ID.

**Fix**: User provided new connection string; updated `~/.ssh/config`:
```
Host lightning-p4
    HostName ssh.lightning.ai
    User s_01kth1fwzr7xwxfnvv436s1pjz
    IdentityFile ~/.ssh/id_ed25519_lightning
```

**Rule**: Always get the current studio ID from the Lightning AI UI before SSH operations. Studio IDs change between sessions. Keep `~/.ssh/config` updated with the current alias.

---

## F9: CEM Planning Negative Result — Goal Representation Failure

**Symptom**: CEM-planned rollouts reach goal -6.9% less often than random action sampling.

**Root cause (multi-factor)**:
1. **Goal frame** was set to the first frame of the next episode — a completely unrelated trajectory. This is not a meaningful planning target.
2. **Latent-space MSE scoring at sigma=1.0** does not correlate well with pixel-space goal proximity at evaluation time. Single-step noisy x0 prediction is a weak planning signal.
3. **CEM horizon** was only 4 steps — too short for meaningful differentiation between action sequences on Bridge data.

**Fix** (what should be done differently):
```python
# Better goal: last frame of same episode (reachable goal)
goal_frame = episode_frames[-1]  # NOT next_episode_frames[0]

# Better scoring: multi-step rollout score, not single-step noisy x0
# Or: use FID/SSIM in decoded pixel space instead of latent MSE

# Better horizon: 16-32 steps for meaningful CEM differentiation
```

**Rule**: World model quality (SSIM 0.963) does not guarantee planning works. Planning requires: (1) reachable goals, (2) scoring that correlates with actual success, (3) sufficient horizon. Treat the planning component as a separate research problem from world model fidelity.

---

## Lessons Summary

| # | Category | Rule |
|---|---|---|
| F1 | Environment | Always reinstall apt packages (ffmpeg) at Lightning AI session start |
| F2 | API | Use `Video2WorldActionConditionedPipeline.from_config()`, not base class or `from_pretrained` |
| F3 | API | `padding_mask` must be explicit bfloat16 zeros, never None |
| F4 | API | Video T must satisfy (T-1) % 4 == 0 for Cosmos tokenizer |
| F5 | API | `pipe.denoise()` returns `DenoisePrediction` — access `.x0` |
| F6 | Data | Confirm actual training data format before writing eval scripts |
| F7 | Environment | Never assume Docker on Lightning AI studios; write self-contained scripts |
| F8 | SSH | Lightning AI studio IDs change between sessions; update ~/.ssh/config |
| F9 | Research | World model SSIM != planning success; goal representation and scoring are separate problems |

---

## Related

- [[P4 - Cosmos Predict Results]] — what succeeded
- [[P4 - Cosmos Predict]] — original task spec
- [[SSH Key Recovery]] — general SSH fix procedure
- [[Git LFS Mismatch on New Studio]] — related git issue
