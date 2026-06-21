# Wakeboarding Start — Humanoid RL

Train a **Unitree G1** in **Isaac Lab** to perform a wakeboard **deep-water start**: begin crouched & seated, feet welded to a board, arms holding a rope handle; a virtual rope ramps pull speed to **30 km/h**; learn to rise from cannonball crouch to stable riding stance.

- **Full plan (executor-ready):** [`PLAN.md`](./PLAN.md)
- **Documentation rules:** [`DOC_PROTOCOL.md`](./DOC_PROTOCOL.md)
- **Compute guide:** [`COMPUTE_GUIDE.md`](./COMPUTE_GUIDE.md)

## Status

🟢 **Stage I COMPLETE** — 5000 iters trained on Modal L40S, `model_latest.pt` saved.
🔵 **Stage II PENDING** — curriculum 10→20→30 km/h, not yet started.
🎥 **Video PENDING** — needs Lightning AI (Modal gVisor blocks RTX renderer).

## Stage I Results (as of 2026-06-21)

| Iteration | Fell rate | Timeout rate | Mean Reward |
|---|---|---|---|
| 0 | 97% | 3% | negative |
| 500 | 6% | 94% | ~0.8 |
| 980 | **0%** | **100%** | ~1.5 |
| 5000 (final) | ~5% | **81%** | ~1.5 |

G1 learned to:
- Stay balanced on the moving board at 10 km/h
- Maintain cannonball/crouch stance (knee bend reward)
- Hold torso upright against rope tension
- Keep arms at hips, handle position correct
- Lean back moderately against the pull

**Best checkpoint**: `wakeboard-ckpts:/wakeboard_stage1/model_latest.pt` (Modal volume)
Also useful: `model_980.pt` (0% fell, cleanest stability), `model_5700.pt` (5% fell, 81% timeout)

## Architecture

- **Method:** PPO (RSL-RL ≥5.0) — two-stage curriculum (Stage I discovery → Stage II speed ramp)
- **Env:** `WakeboardStartEnv` (manager-based), 16 parallel envs, 10 km/h fixed pull (Stage I)
- **Rope model:** virtual spring anchor (`rope_model.py`) — `F = kp·(x_anchor − x_handle) + kd·(v_anchor − v_handle)`, capped at 600N
- **Board:** G1 feet welded to board via `UsdPhysics.FixedJoint`, PhysX solver 16/4 iters
- **Init pose:** cannonball crouch — deep hip/knee flex, torso reclined, `CANNONBALL_ROOT_Z=0.55`
- **Obs:** joint pos/vel, projected gravity, root vel, board pitch, rope force, last action (~70 dims)
- **Act:** 23 DOF joint position targets (PD control, scale=0.5)

## Key Bugs Fixed

1. **rope.reset() double-indexing** — root cause of all Modal CUDA crashes. `rope.reset(nan_ids, self._handle_pos)` — pass full tensor, not pre-sliced `[nan_ids]`
2. **RewardManager timing** — `apply_reward_weights()` MUST be called BEFORE `WakeboardStartEnv()`
3. **Isaac image entrypoint** — Modal needs `.entrypoint([])` to clear `runheadless.sh`, then full Isaac env setup in bash shell_cmd
4. **rsl-rl 5.0 deprecated kwargs** — strip `stochastic`, `init_noise_std` etc from `runner.to_dict()`
5. **Cannonball pose alignment** — spawn=weld=reset must all use `CANNONBALL_ROOT_Z=0.55`

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
