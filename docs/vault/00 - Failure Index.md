---
tags: [failures, debugging, index, lessons]
---

# 00 — Failure Index

Quick-reference table of every significant failure across the program, what the symptom was, and how it was fixed. Read this before starting any new task to avoid repeating mistakes.

---

## Navigation Track (P0–P1.4)

| ID | Task | Symptom | Root Cause | Fix |
|---|---|---|---|---|
| F-01 | P0 | Agent moves but ignores command direction | Reward not connected to command | Added `command_alignment` reward term |
| F-02 | P0 | 28.1% fall rate | No upright reward — agent learned to lean | Added `upright_reward` (weight=0.5) |
| F-03 | P1.4 SeqNav | Robot stands still for entire episode | Training bootstrap failure — `seq_progress` always 0 | Fixed subgoal ordering seed; shaped reward with `progress_scale=2.0` |
| F-04 | P0 eval | Eval crash on first run | Wrong evaluator class for new task type | Matched evaluator to task (see [[Eval Crash - Missing Buffer]]) |
| F-05 | All nav | `exit 0` but no output file | Isaac Sim writes to ephemeral `/tmp` not persistent storage | Moved all outputs to `/teamspace/studios/this_studio/Humanoid/` |
| F-06 | All nav | SSH `Permission denied (publickey)` after restart | Lightning Studio wipes authorized_keys on machine restart | Re-add key from Studio UI each session |
| F-07 | Video render | Video render loop never exits | Sim keeps running after save | Added explicit `env.close()` + process kill |

---

## T1 — GR00T N1.7 (Manipulation)

| ID | Symptom | Root Cause | Fix | Commit |
|---|---|---|---|---|
| F-08 | `AttributeError: 'Gr00tPolicy' object has no attribute 'eval'` | `Gr00tPolicy` wraps `nn.Module` as `self.model` — it is not itself an `nn.Module` | `policy.eval()` → `policy.model.eval()` | `c867f1a` |
| F-09 | `KeyError: 'annotation.human.action.task_description'` | Language key was `"human.action.task_description"` — missing `annotation.` prefix | Read `processor_config.json` libero_sim section; corrected key | `2c58fdd` |
| F-10 | `KeyError: "Joint group 'x' not found in state dict"` | We passed `{"state": tensor(1,1,8)}` — GR00T expects per-joint-group dict `{"x":(1,1,1), "y":(1,1,1), ..., "gripper":(1,1,2)}` | Read `modality.json`; rewrote obs builder to pass per-key tensors | `7501010` |
| F-11 | 0% task success despite no errors | Three obs/action convention bugs vs official `libero_env.py` | Read official source; fixed all three (see detail below) | `388dc6b` |
| F-12 | `flash-attn` pip install fails with `wheel_stub.buildapi` | No source wheel available for flash-attn on this Python/CUDA combo | Install from pre-built wheel URL directly (see [[T1 - GR00T LoRA]]) | — |
| F-13 | `numpy` version conflict | `gym` dependency pulls numpy 2.x; `gr00t` needs 1.26.4 | `pip install 'numpy==1.26.4'` after all other installs | — |
| F-14 | T2 git push rejected from Studio | Studio had old remote state after local pushes | `git pull --rebase origin feat/planned-scripts` then push | — |

---

## F-11 Detail — The Three 0%-Success Bugs

This was the most subtle failure: the eval ran without errors but GR00T never succeeded. Root cause was three simultaneous obs/action convention mismatches vs the official `gr00t/eval/sim/LIBERO/libero_env.py`:

### Bug 1 — Wrong Rotation Representation

```python
# WRONG — we used Euler angles (scipy)
from scipy.spatial.transform import Rotation
xyzw = [quat[1], quat[2], quat[3], quat[0]]
euler = Rotation.from_quat(xyzw).as_euler("xyz")  # ❌

# CORRECT — axis-angle (matches training data)
den = np.sqrt(1.0 - quat[3] ** 2)
if math.isclose(den, 0.0):
    rpy = np.zeros(3)
else:
    rpy = (quat[:3] * 2.0 * math.acos(quat[3])) / den  # ✅
```

**Why it mattered:** GR00T was trained with axis-angle rotation. Passing Euler angles meant the orientation input was in completely the wrong space — the model received garbage state and produced garbage actions.

### Bug 2 — Images Not Flipped

```python
# WRONG — raw image from env
img = obs_dict["agentview_image"]                    # ❌

# CORRECT — flip both spatial axes
img = obs_dict["agentview_image"][::-1, ::-1]        # ✅
```

**Why it mattered:** LIBERO's `OffScreenRenderEnv` renders images in a coordinate system that is vertically and horizontally flipped relative to the convention used during GR00T's training data collection. The official wrapper always flips before passing to the model.

### Bug 3 — Gripper Action Not Transformed

```python
# WRONG — raw GR00T output
action[-1] = action_dict["gripper"][0, 0, 0]         # ❌ value in [0,1]

# CORRECT — normalize → binarize → invert sign
action[-1] = -np.sign(2.0 * action[-1] - 1.0)        # ✅
```

**Why it mattered:** GR00T outputs gripper in `[0, 1]` (LeRobot convention: 0=close, 1=open). LIBERO expects `[-1, +1]` with inverted sign (-1=open, +1=close). Without the transform, the gripper was always doing the opposite of what the model intended.

---

## Pattern: Check the Official Sim Wrapper First

The lesson from F-11: whenever integrating a new model into a new environment, **read the model's official sim wrapper before writing your own obs/action adapter**. For GR00T, this was `gr00t/eval/sim/LIBERO/libero_env.py`. It contained all three fixes.

---

## Related

- [[T1 - GR00T LoRA]] — full T1 task doc with implementation details
- [[Common Failure Patterns]] — pattern-level debugging guide
- [[SeqNav Stand-Still Local Optimum]] — F-03 in full detail
