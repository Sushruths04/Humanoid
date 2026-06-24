# Wakeboarding Start — Humanoid RL

Train a **Unitree G1** in **Isaac Lab** to perform a wakeboard **deep-water start**: begin crouched & seated, feet welded to a board, arms holding a rope handle; a virtual rope ramps pull speed to **30 km/h**; learn to rise from cannonball crouch to stable riding stance.

- **Full plan (executor-ready):** [`PLAN.md`](./PLAN.md)
- **Documentation rules:** [`DOC_PROTOCOL.md`](./DOC_PROTOCOL.md)
- **Compute guide:** [`COMPUTE_GUIDE.md`](./COMPUTE_GUIDE.md)

## Status

🟡 **Stage I IN PROGRESS** — gate-based training running on Modal L40S. Pipeline confirmed working; formal eval not yet run.
🔵 **Stage II BLOCKED** — do not start until Stage I formal eval passes and `traj_tracking`/`amp_style` reward terms are implemented.
🎥 **Video** — `rollout_980_sim.mp4`, `rollout_980_hq.mp4`, `rollout_980_vis.gif` in repo (gpu-l4-bringup branch).

## Stage I Status (as of 2026-06-22)

**Pipeline**: confirmed working (smoke test exit 0, no NaN, checkpoints saving to Modal volume).

**Checkpoints saved**: `model_980.pt` (local), plus iterations up to ~550 in Modal volume.

**Formal eval**: NOT YET RUN. `rollout_980_trace.json` shows board pitch swinging ±87° and negative mean reward — this is "physics running correctly" not "policy solved". Do not claim stable riding until `eval.py` is run and produces real metrics.

**Best checkpoint for eval**: `model_980.pt` (0% fell in training window, needs formal `eval.py` run to confirm).

## Architecture

- **Method:** PPO (RSL-RL ≥5.0) — two-stage curriculum (Stage I discovery → Stage II speed ramp)
- **Env:** `WakeboardStartEnv` (manager-based), 16 parallel envs, 10 km/h fixed pull (Stage I)
- **Rope model:** virtual spring anchor (`rope_model.py`) — `F = kp·(x_anchor − x_handle) + kd·(v_anchor − v_handle)`, capped at 600N
- **Board:** G1 feet welded to board via `UsdPhysics.FixedJoint`, PhysX solver 16/4 iters
- **Init pose:** cannonball crouch — deep hip/knee flex, torso reclined, `CANNONBALL_ROOT_Z=0.50`, 90° yaw (robot faces +Y)
- **Obs:** joint pos/vel, projected gravity, root vel, board pitch, rope force, last action (~70 dims)
- **Act:** 23 DOF joint position targets (PD control, scale=0.5)

## Key Bugs Fixed

1. **rope.reset() double-indexing** — root cause of all Modal CUDA crashes. `rope.reset(nan_ids, self._handle_pos)` — pass full tensor, not pre-sliced `[nan_ids]`
2. **RewardManager timing** — `apply_reward_weights()` MUST be called BEFORE `WakeboardStartEnv()`
3. **Isaac image entrypoint** — Modal needs `.entrypoint([])` to clear `runheadless.sh`, then full Isaac env setup in bash shell_cmd
4. **rsl-rl 5.0 deprecated kwargs** — strip `stochastic`, `init_noise_std` etc from `runner.to_dict()`
5. **Cannonball pose alignment** — spawn=weld=reset must all use `CANNONBALL_ROOT_Z=0.50`

## Quick Commands

```bash
# Train Stage I (Modal)
cd wakeboarding-experiment
modal run modal_app.py --action train --config configs/stage1.yaml

# Resume training
modal run modal_app.py --action train --config configs/stage1.yaml --resume /ckpts/wakeboard_stage1/model_latest.pt

# Check training status
modal app list
modal app logs <app-id>

# Download checkpoint
modal volume get wakeboard-ckpts /wakeboard_stage1/model_latest.pt ./model_latest.pt

# Run video (Lightning AI only — Modal gVisor blocks RTX)
python play.py --checkpoint ./model_latest.pt --v_pull_kmh 10 --episodes 3 --out rollout.mp4
```

## Relation to the rest of the repo

A **separate sub-project** inside the Humanoid repo. Reuses the same Isaac Lab + RSL-RL + Lightning AI + Docker stack. Distinct from the G1 locomotion/vision tasks under `my-humanoid-project/`.
